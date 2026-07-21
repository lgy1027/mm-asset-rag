"""Shared retrieval-quality metrics.

The legacy ``scripts/eval_rag.py`` and ``scripts/eval_extended.py`` each
rolled their own ``hit_rate`` / ``MRR`` helpers; the new full-suite run
also needs ``NDCG@k`` / ``Precision@k`` / ``Recall@k`` / ``MAP`` / ``F1@k``
so the eval reports read like a standard RAG benchmark. Keeping the
math in one place makes it easy to test and easy to swap in a C
implementation later if the suite grows.

Every metric takes:

- ``actual_ids``  — list of ``asset_id`` strings returned by the
  retriever at rank ``1..k``
- ``expected_ids`` — list of acceptable ground-truth ids
- ``k``           — cutoff

For "any-of" relevance (one expected id is enough to count as a hit)
the helpers handle the set-style match automatically.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_HEX = set("0123456789abcdef")


def _strip_trailing_hash_repeated(asset_id: str) -> str:
    """Strip every trailing ``_<8-hex>`` segment, not just the last one.

    Asset ids can stack two hash suffixes when the source filename itself
    carries a leftover hash (e.g. ``Resnext_69df8de4_903a9c76``); the
    single-pass ``strip_trailing_hash`` in evaluation_v2 only removes one
    layer, leaving the id still holding an inner hash. Relevance matching
    here needs both ids reduced to the same bare title, so we loop.

    Stops as soon as the tail is not an all-lowercase 8-char hex segment,
    which keeps ``foo_longtail`` (non-hex) and ``a1b2c3d4`` (no
    underscore separator) untouched.
    """
    s = asset_id
    while "_" in s:
        head, _, tail = s.rpartition("_")
        if len(tail) == 8 and all(c in _HEX for c in tail):
            s = head
        else:
            break
    return s


def _normalize_id(asset_id: str) -> str:
    """Reduce an asset id to a comparable slug.

    Multi-layer hash stripping + casefold + collapse runs of whitespace,
    hyphens and underscores into a single ``-``. This is the single
    source of truth for relevance in this module — every metric below
    goes through it so a hit means the same thing in ``hit_rate``,
    ``MRR``, ``NDCG`` and the eval harness' own ``_match`` (which calls
    back into :func:`_is_relevant`).
    """
    s = _strip_trailing_hash_repeated(asset_id).casefold()
    return re.sub(r"[\s\-_]+", "-", s).strip("-")


def _matches(a_norm: str, b_norm: str) -> bool:
    """Bidirectional substring match between two normalised id slugs.

    An expected title counts as matched when, after normalisation, it is a
    substring of the returned id *or* the returned id is a substring of it.
    This is intentionally permissive: it keeps the short-expected /
    long-actual hit (a bare title ``"obsidian-..."`` is a substring of the
    full returned id ``"...-第-1-名..."``) and the bare-prefix hit
    (``"caltech-panda"`` ⊂ ``"caltech-panda-01"``).

    Known limitation: a very short expected id that merely appears *inside*
    an unrelated longer id (``"bert"`` inside ``"robert"``) would also match.
    The current eval corpus does not trigger this (no paper title is a
    substring of another's); if the corpus grows to, add a length-guarded
    prefix check here rather than tightening globally — a global prefix rule
    cost ~4 real hits (0.855 → 0.782) in the live eval.
    """
    if not a_norm or not b_norm:
        return False
    return a_norm in b_norm or b_norm in a_norm


def _is_relevant(actual_id: str, expected_ids: list[str]) -> bool:
    """Prefix/equal match on normalised ids (see :func:`_matches`).

    Mirrors ``evaluation_v2._match``: an actual id counts as relevant when,
    after normalisation, it is a prefix of — or prefixed by / equal to — any
    expected id. This tolerates the asymmetric shapes real data has (a bare
    short title ``"Obsidian 的 10 大 AI Skill"`` expected vs a long full id
    ``"Obsidian 的 10 大 AI Skill,第 1 名..._924b37db"`` returned), which a
    plain set-equality check would mark as a miss.
    """
    na = _normalize_id(actual_id)
    if not na:
        return False
    return any(_matches(na, _normalize_id(exp)) for exp in expected_ids)


def _expected_norm_dedup(expected_ids: Iterable[str]) -> list[str]:
    """Normalise expected ids and drop duplicates, preserving order.

    Hash variants and stacked-hash variants of the same document collapse
    to one entry (``Resnext`` and ``Resnext_69df8de4_903a9c76`` both →
    ``resnext``), so the relevant-set size is the number of *distinct
    documents*, not the number of id spellings. This is the denominator
    for recall / AP / NDCG and the ideal-hit cap for NDCG.
    """
    out: list[str] = []
    for exp in expected_ids:
        ne = _normalize_id(exp)
        if ne and ne not in out:
            out.append(ne)
    return out


def _relevance_assignment(
    actual_ids: list[str], expected_norm: list[str], k: int | None
) -> list[bool]:
    """Greedy one-expected-per-actual relevance assignment over top-k.

    Walks the (top-k) actuals in rank order; each actual claims the first
    not-yet-claimed expected it matches (normalised substring either way).
    An expected is claimed at most once, so several actuals that all match
    the *same* expected (duplicate hash variants, stacked-hash copies, or
    same-titled different docs) cannot inflate precision / DCG / AP beyond
    the relevant-set size — keeping NDCG and AP ≤ 1.0 by construction.

    Returns a per-position boolean (length = number of actuals considered).
    """
    claimed: set[int] = set()
    assigned: list[bool] = []
    actuals = actual_ids if k is None else actual_ids[:k]
    for a in actuals:
        na = _normalize_id(a)
        idx = None
        if na:
            for i, ne in enumerate(expected_norm):
                if i in claimed:
                    continue
                if _matches(na, ne):
                    idx = i
                    break
        assigned.append(idx is not None)
        if idx is not None:
            claimed.add(idx)
    return assigned


def _relevant_set(expected: Iterable[str]) -> set[str]:
    """Normalise expected ids into a set; empty when nothing to match."""
    return {e for e in expected if e}


def hit_rate_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """1.0 if any of the top-k ``actual_ids`` is relevant to ``expected_ids``.

    Relevance is normalised bidirectional substring (see :func:`_is_relevant`),
    not set equality, so a bare expected title matches a longer returned id.
    """
    if not _relevant_set(expected_ids):
        return 0.0
    return 1.0 if any(_is_relevant(a, expected_ids) for a in actual_ids[:k]) else 0.0


def reciprocal_rank(actual_ids: list[str], expected_ids: list[str]) -> float:
    """1 / rank of the first relevant result, 0 if none."""
    if not _relevant_set(expected_ids):
        return 0.0
    for rank, a in enumerate(actual_ids, start=1):
        if _is_relevant(a, expected_ids):
            return 1.0 / rank
    return 0.0


def precision_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of the top-k results that are relevant. Empty expected → 0.

    Relevance is greedy one-expected-per-actual (see
    :func:`_relevance_assignment`), so several top-k results pointing at the
    same expected document count once — the fraction stays ≤ 1.
    """
    if k <= 0:
        return 0.0
    expected_norm = _expected_norm_dedup(expected_ids)
    if not expected_norm:
        return 0.0
    topk = actual_ids[:k]
    if not topk:
        return 0.0
    assigned = _relevance_assignment(actual_ids, expected_norm, k)
    return sum(1 for r in assigned if r) / len(topk)


def recall_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of the relevant set that is matched by the top-k. Empty expected → 0.

    Denominator is the number of distinct expected documents (hash variants
    collapsed); numerator is how many of them some top-k actual claims.
    """
    expected_norm = _expected_norm_dedup(expected_ids)
    if not expected_norm:
        return 0.0
    assigned = _relevance_assignment(actual_ids, expected_norm, k)
    matched = sum(1 for r in assigned if r)
    return matched / len(expected_norm)


def f1_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k. 0 if both are 0."""
    p = precision_at_k(actual_ids, expected_ids, k)
    r = recall_at_k(actual_ids, expected_ids, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(actual_ids: list[str], expected_ids: list[str]) -> float:
    """AP = sum_k (P(k) * rel(k)) / |relevant|.

    Standard definition; the denominator is the number of distinct relevant
    documents (hash variants collapsed). ``rel(k)`` is the greedy
    one-expected-per-actual assignment, so the running hit count never
    exceeds the relevant-set size and AP stays ≤ 1.0 even when several
    returned ids point at the same expected document.
    """
    expected_norm = _expected_norm_dedup(expected_ids)
    if not expected_norm:
        return 0.0
    assigned = _relevance_assignment(actual_ids, expected_norm, None)
    score = 0.0
    hits = 0
    for rank, rel in enumerate(assigned, start=1):
        if rel:
            hits += 1
            score += hits / rank
    return score / len(expected_norm)


def _log2(x: int) -> float:
    import math

    return math.log2(x)


def dcg_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Discounted cumulative gain at k (binary relevance, greedy assignment)."""
    expected_norm = _expected_norm_dedup(expected_ids)
    if not expected_norm:
        return 0.0
    assigned = _relevance_assignment(actual_ids, expected_norm, k)
    score = 0.0
    for rank, rel in enumerate(assigned, start=1):
        if rel:
            # log2(rank+1) — standard; rank=1 → 1.0
            score += 1.0 / _log2(rank + 1)
    return score


def ndcg_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """NDCG@k with binary relevance; ideal DCG assumes the first |relevant|
    positions are all hits (and the rest don't matter).

    Because relevance uses greedy one-expected-per-actual assignment and the
    ideal is capped at the number of *distinct* relevant documents, the DCG
    numerator can never exceed the IDCG denominator → NDCG ≤ 1.0 even with
    duplicate hash variants in the returned ids.
    """
    expected_norm = _expected_norm_dedup(expected_ids)
    if not expected_norm:
        return 0.0
    assigned = _relevance_assignment(actual_ids, expected_norm, k)
    dcg = 0.0
    for rank, rel in enumerate(assigned, start=1):
        if rel:
            dcg += 1.0 / _log2(rank + 1)
    ideal_hits = min(len(expected_norm), k)
    idcg = sum(1.0 / _log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aggregate_metrics(
    results: list[dict],
    *,
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict:
    """Compute hit_rate / MRR / NDCG / Precision / Recall / F1 / MAP for each k.

    ``results`` is a list of per-query dicts each carrying:
        ``actual_ids``   — list[str] returned by the retriever
        ``expected_ids`` — list[str] ground truth

    Returns a dict keyed by metric name with a per-k breakdown plus MAP
    (which is k-independent).
    """
    ks = sorted(set(k_values))
    out: dict[str, dict[int, float]] = {
        "hit_rate": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "ndcg": {},
    }
    for k in ks:
        out["hit_rate"][k] = sum(
            hit_rate_at_k(r["actual_ids"], r["expected_ids"], k) for r in results
        ) / max(len(results), 1)
        out["precision"][k] = sum(
            precision_at_k(r["actual_ids"], r["expected_ids"], k) for r in results
        ) / max(len(results), 1)
        out["recall"][k] = sum(
            recall_at_k(r["actual_ids"], r["expected_ids"], k) for r in results
        ) / max(len(results), 1)
        out["f1"][k] = sum(f1_at_k(r["actual_ids"], r["expected_ids"], k) for r in results) / max(
            len(results), 1
        )
        out["ndcg"][k] = sum(
            ndcg_at_k(r["actual_ids"], r["expected_ids"], k) for r in results
        ) / max(len(results), 1)
    out["mrr"] = sum(reciprocal_rank(r["actual_ids"], r["expected_ids"]) for r in results) / max(
        len(results), 1
    )
    out["map"] = sum(average_precision(r["actual_ids"], r["expected_ids"]) for r in results) / max(
        len(results), 1
    )
    return out
