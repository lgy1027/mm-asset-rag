"""Chinese-aware BM25 sparse encoder (jieba + Okapi).

Companion to :mod:`mm_asset_rag.backends.qdrant_backend` which hosts the
English fastembed BM25 vector. ``bm25_zh`` produces a parallel sparse
vector (``bm25_zh``) so Chinese documents and queries get token-level
recall instead of relying on dense embeddings alone.

Design notes:

- **Tokeniser**: ``jieba.cut`` for CJK + a Latin/number mask applied
  before jieba so terms like ``BERT`` / ``LayoutLM`` / ``V100`` survive
  intact. The Latin mask runs first, replaces matched spans with
  unique placeholder tokens, jieba runs on the masked text, then
  placeholders are substituted back. This keeps multi-script documents
  (English code-switching in Chinese text) clean.
- **BM25 math**: Robertson–Walker IDF + Okapi BM25, same shape as the
  in-tree ``_bm25_okapi_scores`` helper used by the per-asset chunk
  selector. ``k1=1.5`` and ``b=0.75`` are the standard defaults.
- **Hash-to-int mapping**: a term's sparse-vector index is
  ``sha1(term)[:4] -> uint32``. SHA1 is deterministic across processes
  so an ``indexing`` Python and a ``querying`` Python agree on the
  same integer for the same term, unlike Python's built-in ``hash()``
  which is salted. 32 bits gives 4.3B slots — collisions are
  negligible for the corpus sizes we deal with.
- **Query side**: the query vector is just ``idf[term]`` for every
  present term (tf=1 by definition). ``Qdrant`` accepts a sparse
  vector directly without needing per-query scores.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from typing import Iterable

import jieba
from qdrant_client import models

_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
_NUM_RE = re.compile(r"\d+")
# Placeholder characters use the Unicode Private Use Area (U+E000..U+F8FF).
# These codepoints are guaranteed not to appear in real text, jieba
# leaves them as single tokens, and crucially they contain NO decimal
# digits — so ``_NUM_RE`` won't re-match inside the Latin placeholder
# and glue the two masks together.
_PUA_LATIN_BASE = 0xE000
_PUA_NUMBER_BASE = 0xF000


def _placeholder(index: int, base: int) -> str:
    """Build a single-codepoint PUA placeholder."""
    return chr(base + index)


def tokenize_zh(text: str | None) -> list[str]:
    """Tokenise ``text`` for the Chinese BM25 sparse vector.

    CJK text goes through ``jieba.cut``; Latin words and numbers are
    captured before jieba runs and re-inserted as their original form
    (lowercased) after segmentation. Whitespace-only and ``None``
    inputs return an empty list.
    """
    if not text:
        return []
    _ensure_jieba()
    text = str(text)

    # Mask Latin tokens and numbers so jieba doesn't break them apart.
    latin_placeholders: dict[str, str] = {}
    num_placeholders: dict[str, str] = {}

    def _latin_sub(match: re.Match[str]) -> str:
        placeholder = _placeholder(len(latin_placeholders), _PUA_LATIN_BASE)
        latin_placeholders[placeholder] = match.group(0).lower()
        return placeholder

    def _num_sub(match: re.Match[str]) -> str:
        placeholder = _placeholder(len(num_placeholders), _PUA_NUMBER_BASE)
        num_placeholders[placeholder] = match.group(0)
        return placeholder

    masked = _LATIN_RE.sub(_latin_sub, text)
    masked = _NUM_RE.sub(_num_sub, masked)

    out: list[str] = []
    for tok in jieba.cut(masked):
        if not tok or tok.isspace():
            continue
        if tok in latin_placeholders:
            out.append(latin_placeholders[tok])
        elif tok in num_placeholders:
            out.append(num_placeholders[tok])
        else:
            out.append(tok)
    return out

_JIEBA_LOCK = threading.Lock()
_JIEBA_INITIALISED = False


def _ensure_jieba() -> None:
    """Trigger jieba's first-time dict load exactly once per process."""
    global _JIEBA_INITIALISED
    with _JIEBA_LOCK:
        if not _JIEBA_INITIALISED:
            # ``initialize`` is a no-op after the first call but cheap to
            # call again — keeps the flag-based single-init logic simple.
            jieba.initialize()
            _JIEBA_INITIALISED = True


def tokenize_zh(text: str | None) -> list[str]:
    """Tokenise ``text`` for the Chinese BM25 sparse vector.

    CJK text goes through ``jieba.cut``; Latin words and numbers are
    captured before jieba runs and re-inserted as their original form
    (lowercased) after segmentation. Whitespace-only and ``None``
    inputs return an empty list.
    """
    if not text:
        return []
    _ensure_jieba()
    text = str(text)

    # Mask Latin tokens and numbers so jieba doesn't break them apart.
    latin_placeholders: dict[str, str] = {}
    num_placeholders: dict[str, str] = {}

    def _latin_sub(match: re.Match[str]) -> str:
        placeholder = _placeholder(len(latin_placeholders), _PUA_LATIN_BASE)
        latin_placeholders[placeholder] = match.group(0).lower()
        return placeholder

    def _num_sub(match: re.Match[str]) -> str:
        placeholder = _placeholder(len(num_placeholders), _PUA_NUMBER_BASE)
        num_placeholders[placeholder] = match.group(0)
        return placeholder

    masked = _LATIN_RE.sub(_latin_sub, text)
    masked = _NUM_RE.sub(_num_sub, masked)

    out: list[str] = []
    for tok in jieba.cut(masked):
        if not tok or tok.isspace():
            continue
        if tok in latin_placeholders:
            out.append(latin_placeholders[tok])
        elif tok in num_placeholders:
            out.append(num_placeholders[tok])
        else:
            out.append(tok)
    return out


def _term_to_index(term: str) -> int:
    """Deterministic 32-bit hash so indexing and querying agree."""
    digest = hashlib.sha1(term.encode("utf-8")).digest()[:4]
    return int.from_bytes(digest, "big", signed=False)


def compute_idf(
    documents_tokens: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Build the BM25 Okapi IDF table for a tokenised corpus.

    The returned dict also carries ``_avgdl``, ``_k1``, ``_b`` so the
    score function can reconstruct everything from a single map.
    """
    n = len(documents_tokens)
    avgdl = (
        sum(len(d) for d in documents_tokens) / n if n else 0.0
    )
    df: Counter[str] = Counter()
    for d in documents_tokens:
        for term in set(d):
            df[term] += 1
    idf: dict[str, float] = {"_avgdl": avgdl, "_k1": k1, "_b": b}
    for term, freq in df.items():
        # Robertson–Walker IDF, smoothed so very common terms still score.
        idf[term] = math.log(1 + (n - freq + 0.5) / (freq + 0.5))
    return idf


def bm25_zh_score(
    query_tokens: Iterable[str],
    doc_tokens: list[str],
    idf: dict[str, float],
) -> float:
    """Score a single query against a single doc under the stored IDF."""
    if not query_tokens or not doc_tokens:
        return 0.0
    avgdl = max(idf.get("_avgdl", 1.0), 1.0)
    k1 = idf.get("_k1", 1.5)
    b = idf.get("_b", 0.75)
    dl = max(len(doc_tokens), 1)
    tf = Counter(doc_tokens)
    score = 0.0
    for q in query_tokens:
        if q not in idf:
            continue
        f = tf.get(q, 0)
        if f == 0:
            continue
        score += idf[q] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return score


def bm25_zh_encode_query(
    query_tokens: Iterable[str],
    idf: dict[str, float],
) -> models.SparseVector:
    """Encode a tokenised query as a Qdrant sparse vector.

    The query-side BM25 score reduces to ``idf[term] * (k1+1)/(k1+1) =
    idf[term]`` because tf=1 on the query side; the vector is one entry
    per distinct present-in-idf term.
    """
    seen: dict[str, float] = {}
    for t in query_tokens:
        if t in idf and t not in seen:
            seen[t] = idf[t]
    if not seen:
        return models.SparseVector(indices=[], values=[])
    # Sort by term to make the output order deterministic.
    items = sorted(seen.items())
    indices = [_term_to_index(t) for t, _ in items]
    values = [v for _, v in items]
    return models.SparseVector(indices=indices, values=values)


def build_bm25_zh_index(
    documents: list,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[list[models.SparseVector], dict[str, float]]:
    """Tokenise all documents, compute IDF, and emit per-doc sparse vectors.

    Each returned vector is the document-side BM25 score for the
    document's own tokens — a standard sparse representation that
    pairs with the query-side vector above. Empty texts produce empty
    sparse vectors (Qdrant accepts these as null contributions).
    """
    docs_tokens = [tokenize_zh(getattr(d, "text", "") or "") for d in documents]
    idf = compute_idf(docs_tokens, k1=k1, b=b)
    avgdl = max(idf["_avgdl"], 1.0)
    k1 = idf["_k1"]
    b = idf["_b"]

    out: list[models.SparseVector] = []
    for tokens in docs_tokens:
        if not tokens:
            out.append(models.SparseVector(indices=[], values=[]))
            continue
        dl = max(len(tokens), 1)
        tf = Counter(tokens)
        scores: dict[int, float] = {}
        for term, f in tf.items():
            term_idf = idf.get(term)
            if term_idf is None or f == 0:
                continue
            score = term_idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            idx = _term_to_index(term)
            scores[idx] = scores.get(idx, 0.0) + score
        if not scores:
            out.append(models.SparseVector(indices=[], values=[]))
            continue
        indices = sorted(scores.keys())
        values = [scores[i] for i in indices]
        out.append(models.SparseVector(indices=indices, values=values))
    return out, idf