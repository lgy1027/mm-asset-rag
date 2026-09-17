"""Keyword extraction for text chunks.

Two extractors that the parser / ingest path can attach to chunks as
metadata (``keywords=[...]``) so the BM25 channel has more signal for
short queries like "联宝 ESG" or "Codex 全景指南":

- :func:`extract_keywords_zh` — jieba ``analyse.textrank`` for
  Chinese text (top-K by PageRank score). Falls back to a naive
  noun-phrase frequency count when jieba is unavailable.
- :func:`extract_keywords_en` — simple stopword-filtered frequency
  count for Latin-script text. Stems by lowercasing only (no Porter
  stemmer; CLIR results on the bundled corpus are within noise
  without stemming).

Both functions are pure (no I/O) and accept any text length. Output
is a list of unique tokens, capped at ``top_k`` (default 8). When
``text`` is empty / whitespace-only the result is ``[]`` so the
caller can safely forward it into chunk metadata.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

# Cached stopword sets. Small language-agnostic lists — the goal is
# to filter "the" / "的" / "了" rather than build a complete lexical
# resource. Deployments that need richer stopword lists can extend
# these in place.

_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "are",
        "was",
        "were",
        "but",
        "not",
        "you",
        "your",
        "our",
        "all",
        "any",
        "can",
        "has",
        "have",
        "had",
        "one",
        "two",
        "may",
        "use",
        "used",
        "via",
        "of",
        "in",
        "on",
        "at",
        "to",
        "is",
        "it",
        "by",
        "as",
        "or",
        "an",
        "be",
        "if",
        "we",
        "do",
    }
)
_ZH_STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "和",
        "是",
        "在",
        "我",
        "有",
        "与",
        "也",
        "就",
        "都",
        "而",
        "及",
        "等",
        "着",
        "或",
        "一个",
        "没有",
        "我们",
        "你们",
        "他们",
        "这",
        "那",
        "为",
        "以",
        "从",
        "到",
        "把",
        "被",
        "对",
        "于",
        "上",
        "下",
        "中",
    }
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]+")

# Markdown inline image reference: ``![alt](url)``. Used to strip embedded
# image refs before keyword extraction. Matches any target — relative path
# (``images/p0_i0.png``), data URL, or http(s) URL — so it is corpus-agnostic
# and works for every parser that emits markdown (markitdown / docling /
# PaddleOCR-VL). PyMuPDF bodies carry no such refs (images are associated by
# bbox, not inline syntax), so the strip is a no-op there.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def _tokenize(text: str) -> list[str]:
    """Split on Latin alphanumeric runs and contiguous CJK blocks.

    CJK blocks are kept whole — the next stage (``analyse.textrank``
    for Chinese, frequency for English) does the segmentation.
    """
    return _WORD_RE.findall(text or "")


@lru_cache(maxsize=1)
def _jieba_textrank():
    """Lazy-load jieba's TextRank analyser. First call imports jieba.

    We import on first use because the project already has jieba as
    a hard dependency (``pyproject.toml``), but we don't want to
    pay the import cost on every ``extract_keywords_zh`` call.
    """
    try:
        import jieba.analyse  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — jieba is a hard dep
        return None
    return jieba.analyse


def _strip_markdown_images(text: str) -> str:
    """Remove ``![alt](url)`` refs so they don't leak into keywords.

    An embedded-image ref is layout, not semantics: ``![](images/p0_i0.png)``
    carries no searchable meaning, but ``extract_keywords_zh``'s bigram
    fallback (and jieba, on path-like tokens) would otherwise surface
    ``images`` / ``markitdown`` / ``png`` / a hash as keywords. Those tokens
    then get injected as a "关键词: ..." footer and pollute the BM25 channel
    for any chunk whose body is mostly an image ref — a common shape in
    image-heavy office decks where a slide's text reduces to a lone figure.

    Stripping before extraction is corpus- and parser-agnostic: bodies with
    no refs are unchanged; the image↔chunk association in ``ir_to_documents``
    reads the original ``body`` (not this stripped form), so this never drops
    an association.
    """
    return _MD_IMAGE_RE.sub(" ", text or "")


def extract_keywords_zh(text: str, top_k: int = 8) -> list[str]:
    """Top-K Chinese keywords by jieba TextRank.

    Falls back to a stopword-filtered character-bigram frequency
    count when jieba is unavailable. The bigram fallback captures
    short noun phrases (``联宝``, ``Codex``) which is enough for
    the bundled corpus.
    """
    if not text or not text.strip():
        return []
    textrank = _jieba_textrank()
    if textrank is not None:
        try:
            return list(textrank.textrank(text, topK=top_k))
        except Exception:  # pragma: no cover — TextRank on empty is rare
            pass
    # Fallback: CJK characters only, bigrams, stopword filter.
    from itertools import pairwise

    cjk_chars = [c for c in text if "一" <= c <= "鿿"]
    bigrams = ["".join(pair) for pair in pairwise(cjk_chars)]
    counter = Counter(
        bg for bg in bigrams if bg not in _ZH_STOPWORDS and not all(c in _ZH_STOPWORDS for c in bg)
    )
    return [bg for bg, _ in counter.most_common(top_k)]


_EN_MIN_LEN = 3


def extract_keywords_en(text: str, top_k: int = 8) -> list[str]:
    """Top-K English keywords by stopword-filtered frequency.

    No stemming — the bundled corpus (English arxiv papers) has
    consistent lemma forms. We tokenise on ``[A-Za-z0-9]+`` and
    drop tokens shorter than ``_EN_MIN_LEN=3`` plus the small
    stopword list. Numbers are kept because they often appear in
    paper titles (``Resnet50``, ``GPT3``). The result preserves
    original casing — multilingual embeddings are case-aware and
    the BM25 side lowercases at the query preprocessor.
    """
    if not text or not text.strip():
        return []
    counter: Counter[str] = Counter()
    for tok in _tokenize(text):
        if len(tok) < _EN_MIN_LEN:
            continue
        if tok.lower() in _EN_STOPWORDS:
            continue
        counter[tok] += 1
    return [tok for tok, _ in counter.most_common(top_k)]


def extract_keywords(text: str, top_k: int = 8, *, language: str = "zh") -> list[str]:
    """Single-entry dispatcher that picks the right extractor.

    ``language="zh"`` runs jieba TextRank (with bigram fallback);
    ``language="en"`` runs the simple stopword-frequency extractor.
    The dispatcher falls back to whichever language yields a
    non-empty result when ``language="auto"`` (try jieba first,
    then English).
    """
    if language == "zh":
        return extract_keywords_zh(text, top_k=top_k)
    if language == "en":
        return extract_keywords_en(text, top_k=top_k)
    if language == "auto":
        zh = extract_keywords_zh(text, top_k=top_k)
        if zh:
            return zh
        return extract_keywords_en(text, top_k=top_k)
    raise NotImplementedError(f"keyword extractor not implemented for language={language!r}")


def enrich_chunk_text(text: str, keywords: list[str], *, separator: str = "\n\n关键词: ") -> str:
    """Append the keyword list to a chunk's text under a labelled header.

    The header makes it visible in retrieval debug output and gives
    the BM25 channel a deterministic injection point for keywords
    that the dense channel might under-weight. Returns ``text``
    unchanged when ``keywords`` is empty so callers don't add empty
    header lines to the index.
    """
    if not keywords:
        return text
    return text + separator + " ".join(keywords)
