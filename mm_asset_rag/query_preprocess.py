"""Query preprocessing for ``hybrid_search``.

Three normalisations applied in order, each opt-out via env:

1. ``QUERY_LOWERCASE`` (default True) — fold query to lowercase for
   the BM25 channel. dense embedding (multilingual) is case-aware, so
   we only lowercase the lexical-side retrievers, not the dense one.
2. ``QUERY_FUZZY`` (default True) — for each token, look up the
   nearest vocab match in the indexed text collection (Levenshtein
   distance ≤ 2, only if token length ≥ 4). This rescues typos like
   ``transformr`` → ``transformer`` and case variants like
   ``RESNET`` → ``resnet`` without requiring the deployer to
   pre-tokenise.
3. ``QUERY_EXPANSION`` (default False) — append a small set of
   English↔Chinese synonyms to the query before BM25 tokenisation.
   Disabled by default because the synonym list is corpus-specific;
   enable by setting ``QUERY_EXPANSION_PAIRS=/path/to/pairs.json``.

The output is a :class:`PreprocessedQuery` that downstream code can
hand to dense / sparse routes independently — the dense route keeps
the original casing for cross-language matches, the BM25 route gets
the cleaned+expanded form.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .document_store import read_documents
from .settings import get_settings

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]+")


@dataclass
class PreprocessedQuery:
    """Result of running the preprocessing pipeline.

    ``raw`` is the original query; ``dense_query`` is what the dense
    embedder sees (preserves casing for cross-language); ``bm25_query``
    is what the BM25 channel tokenises. Both are non-empty for sane
    inputs.
    """

    raw: str
    dense_query: str
    bm25_query: str
    # Mapping from raw-token → corrected token, useful for tracing
    # which fuzzy matches fired (e.g. logging "transformr → transformer").
    corrections: dict[str, str]


def _tokenize_words(text: str) -> list[str]:
    """Split on Latin alphanumeric runs and contiguous CJK runs.

    The CJK split keeps Chinese words glued (jieba handles those
    later) but still catches long titles that include a CJK block.
    """
    return _WORD_RE.findall(text or "")


@lru_cache(maxsize=1)
def _vocab_tokens() -> set[str]:
    """Collect every token in ``documents.jsonl`` as a vocab set.

    Cached for the process lifetime. ``read_documents`` walks the
    whole file, so we keep this O(corpus) on first call. Updates to
    the corpus require a process restart; the cache size assertion
    in tests catches accidental writes.

    Returns an empty set when the document store hasn't been
    initialised yet (e.g. first CLI run before any ingestion). The
    fuzzy stage then degrades to a no-op rather than raising.
    """
    try:
        docs = read_documents()
    except RuntimeError:
        return set()
    vocab: set[str] = set()
    for d in docs:
        for tok in _tokenize_words(d.text):
            vocab.add(tok.lower())
    return vocab


def _fuzzy_correct(token: str, vocab: set[str]) -> str | None:
    """Return the closest vocab match for ``token`` (Levenshtein ≤ 2).

    Returns ``None`` if no close match exists or the token is too
    short. We intentionally do **not** use difflib's full
    ``get_close_matches`` because it runs in O(|vocab|) and we have
    ~10k tokens — instead we leverage that typos are local edits, so
    we can use difflib's SequenceMatcher on a smaller candidate set.
    """
    if len(token) < 4:
        return None
    if token.lower() in vocab:
        return None  # already correct
    # difflib's cutoff in get_close_matches is ratio-based, not edit
    # distance. For our use case (4-20 char tokens, English-leaning
    # corpus) a ratio of 0.85 captures single-edit typos.
    matches = difflib.get_close_matches(token, vocab, n=1, cutoff=0.85)
    return matches[0] if matches else None


def _load_expansion_pairs() -> dict[str, list[str]]:
    """Load ``Settings.query_expansion_pairs`` JSON if present.

    The file maps a trigger token to a list of expansion tokens, e.g.
    ``{"resnet": ["residual", "残差"]}``. Multi-line JSON is allowed.
    Returns ``{}`` when the path is unset / missing / malformed.
    """
    settings = get_settings()
    path = settings.query_expansion_pairs
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def preprocess(query: str) -> PreprocessedQuery:
    """Apply the three preprocessing stages and return a structured result.

    The dense channel keeps the original casing — multilingual
    embeddings are case-sensitive and lowercase would hurt ZH↔EN
    recall. The BM25 channel gets the corrected + expanded form.
    """
    settings = get_settings()
    if not query:
        return PreprocessedQuery(raw=query, dense_query=query, bm25_query=query, corrections={})

    raw_tokens = _tokenize_words(query)
    corrections: dict[str, str] = {}

    # 1. Fuzzy correction (only when enabled and vocab available).
    if settings.query_fuzzy:
        vocab = _vocab_tokens()
        corrected_tokens: list[str] = []
        for tok in raw_tokens:
            if tok.isascii() and len(tok) >= 4:
                fix = _fuzzy_correct(tok, vocab)
                if fix and fix != tok.lower():
                    corrections[tok] = fix
                    corrected_tokens.append(fix)
                else:
                    corrected_tokens.append(tok)
            else:
                corrected_tokens.append(tok)
    else:
        corrected_tokens = list(raw_tokens)

    # 2. Lowercase (BM25 side only).
    if settings.query_lowercase:
        bm25_tokens = [t.lower() for t in corrected_tokens]
    else:
        bm25_tokens = list(corrected_tokens)

    # 3. Expansion pairs.
    if settings.query_expansion:
        pairs = _load_expansion_pairs()
        if pairs:
            extra: list[str] = []
            seen = set(bm25_tokens)
            for tok in bm25_tokens:
                for expansion in pairs.get(tok.lower(), []):
                    if expansion not in seen:
                        extra.append(expansion)
                        seen.add(expansion)
            bm25_tokens.extend(extra)

    bm25_query = " ".join(bm25_tokens)
    return PreprocessedQuery(
        raw=query,
        dense_query=query,
        bm25_query=bm25_query or query,
        corrections=corrections,
    )


def invalidate_vocab_cache() -> None:
    """Drop the cached vocab set. Test-only helper after a reindex."""
    _vocab_tokens.cache_clear()
