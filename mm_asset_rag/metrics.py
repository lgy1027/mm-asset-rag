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

from collections.abc import Iterable


def _relevant_set(expected: Iterable[str]) -> set[str]:
    """Normalise expected ids into a set; empty when nothing to match."""
    return {e for e in expected if e}


def hit_rate_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """1.0 if any of the top-k ``actual_ids`` is in ``expected_ids`` else 0.0."""
    expected = _relevant_set(expected_ids)
    if not expected:
        return 0.0
    return 1.0 if any(a in expected for a in actual_ids[:k]) else 0.0


def reciprocal_rank(actual_ids: list[str], expected_ids: list[str]) -> float:
    """1 / rank of the first relevant result, 0 if none."""
    expected = _relevant_set(expected_ids)
    for rank, a in enumerate(actual_ids, start=1):
        if a in expected:
            return 1.0 / rank
    return 0.0


def precision_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of the top-k results that are relevant. Empty expected → 0."""
    if k <= 0:
        return 0.0
    expected = _relevant_set(expected_ids)
    if not expected:
        return 0.0
    topk = actual_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for a in topk if a in expected) / len(topk)


def recall_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of the relevant set that appears in the top-k. Empty expected → 0."""
    expected = _relevant_set(expected_ids)
    if not expected:
        return 0.0
    topk = set(actual_ids[:k])
    return len(expected & topk) / len(expected)


def f1_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k. 0 if both are 0."""
    p = precision_at_k(actual_ids, expected_ids, k)
    r = recall_at_k(actual_ids, expected_ids, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(actual_ids: list[str], expected_ids: list[str]) -> float:
    """AP = sum_k (P(k) * rel(k)) / |relevant|.

    Standard definition; the denominator is the size of the relevant set.
    This is a non-truncated variant (uses every returned id, not just top-k),
    matching sklearn's ``average_precision_score``.
    """
    expected = _relevant_set(expected_ids)
    if not expected:
        return 0.0
    score = 0.0
    hits = 0
    for rank, a in enumerate(actual_ids, start=1):
        if a in expected:
            hits += 1
            score += hits / rank
    return score / len(expected)


def _log2(x: int) -> float:
    import math

    return math.log2(x)


def dcg_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Discounted cumulative gain at k (binary relevance)."""
    expected = _relevant_set(expected_ids)
    if not expected:
        return 0.0
    score = 0.0
    for rank, a in enumerate(actual_ids[:k], start=1):
        if a in expected:
            # log2(rank+1) — standard; rank=1 → 1.0
            score += 1.0 / _log2(rank + 1)
    return score


def ndcg_at_k(actual_ids: list[str], expected_ids: list[str], k: int) -> float:
    """NDCG@k with binary relevance; ideal DCG assumes the first |expected|
    positions are all hits (and the rest don't matter)."""
    expected = _relevant_set(expected_ids)
    if not expected:
        return 0.0
    dcg = 0.0
    for rank, a in enumerate(actual_ids[:k], start=1):
        if a in expected:
            dcg += 1.0 / _log2(rank + 1)
    ideal_hits = min(len(expected), k)
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
