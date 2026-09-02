"""Exact document-level retrieval metrics over graded qrels.

The retrieval boundary is a logical ``document_id``.  No filename, title,
case, hash, or substring normalization participates in relevance.  Qrels use
the compact shape ``{document_id: relevance}``; only positive integer grades
are relevant, and NDCG uses the standard exponential gain ``2**grade - 1``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

_HEX = set("0123456789abcdef")


def _normalize_id(value: str) -> str:
    """Normalize legacy answer-evaluation citation labels.

    Retrieval metrics deliberately do not call this helper.  It remains only
    because answer-quality evaluation has a separate, asset-citation contract.
    """
    normalized = value
    while "_" in normalized:
        head, _, tail = normalized.rpartition("_")
        if len(tail) == 8 and all(char in _HEX for char in tail):
            normalized = head
        else:
            break
    return re.sub(r"[\s\-_]+", "-", normalized.casefold()).strip("-")


def _positive_qrels(qrels: Mapping[str, int]) -> dict[str, int]:
    return {
        document_id: relevance
        for document_id, relevance in qrels.items()
        if isinstance(document_id, str)
        and document_id
        and isinstance(relevance, int)
        and not isinstance(relevance, bool)
        and relevance > 0
    }


def _is_relevant(actual_document_id: str, qrels: Mapping[str, int]) -> bool:
    """Return whether an exact document ID has a positive qrel."""
    return actual_document_id in _positive_qrels(qrels)


def recall_at_k(actual_document_ids: list[str], qrels: Mapping[str, int], k: int) -> float:
    """Fraction of positively judged documents retrieved in the top ``k``."""
    relevant = set(_positive_qrels(qrels))
    if k <= 0 or not relevant:
        return 0.0
    retrieved = set(actual_document_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank(actual_document_ids: list[str], qrels: Mapping[str, int]) -> float:
    """Reciprocal rank of the first exact positively judged document."""
    relevant = set(_positive_qrels(qrels))
    if not relevant:
        return 0.0
    for rank, document_id in enumerate(actual_document_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(actual_document_ids: list[str], qrels: Mapping[str, int]) -> float:
    """Binary AP over positive qrels, with duplicate results counted once."""
    relevant = set(_positive_qrels(qrels))
    if not relevant:
        return 0.0
    seen: set[str] = set()
    hits = 0
    score = 0.0
    for rank, document_id in enumerate(actual_document_ids, start=1):
        if document_id in seen:
            continue
        seen.add(document_id)
        if document_id in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


def dcg_at_k(actual_document_ids: list[str], qrels: Mapping[str, int], k: int) -> float:
    """Graded DCG@k using exact IDs and exponential relevance gain."""
    if k <= 0:
        return 0.0
    relevant = _positive_qrels(qrels)
    seen: set[str] = set()
    score = 0.0
    for rank, document_id in enumerate(actual_document_ids[:k], start=1):
        if document_id in seen:
            continue
        seen.add(document_id)
        relevance = relevant.get(document_id, 0)
        if relevance:
            score += (2**relevance - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(actual_document_ids: list[str], qrels: Mapping[str, int], k: int) -> float:
    """Graded normalized DCG@k against the ideal qrel ordering."""
    if k <= 0:
        return 0.0
    grades = sorted(_positive_qrels(qrels).values(), reverse=True)[:k]
    if not grades:
        return 0.0
    ideal = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
    return dcg_at_k(actual_document_ids, qrels, k) / ideal if ideal else 0.0


def aggregate_metrics(
    results: list[dict[str, object]], *, k_values: Iterable[int] = (1, 3, 5, 10)
) -> dict[str, object]:
    """Macro-average Recall, MRR, MAP, and graded NDCG over query results."""
    ks = sorted(set(k_values))
    count = len(results)
    denominator = max(count, 1)

    def row(result: dict[str, object]) -> tuple[list[str], Mapping[str, int]]:
        actual = result.get("actual_document_ids", [])
        qrels = result.get("qrels", {})
        if not isinstance(actual, list) or not all(isinstance(item, str) for item in actual):
            raise TypeError("actual_document_ids must be a list of strings")
        if not isinstance(qrels, Mapping):
            raise TypeError("qrels must be a mapping of document_id to relevance")
        return actual, qrels  # type: ignore[return-value]

    rows = [row(result) for result in results]
    return {
        "recall": {
            k: sum(recall_at_k(actual, qrels, k) for actual, qrels in rows) / denominator
            for k in ks
        },
        "mrr": sum(reciprocal_rank(actual, qrels) for actual, qrels in rows) / denominator,
        "map": sum(average_precision(actual, qrels) for actual, qrels in rows) / denominator,
        "ndcg": {
            k: sum(ndcg_at_k(actual, qrels, k) for actual, qrels in rows) / denominator for k in ks
        },
    }
