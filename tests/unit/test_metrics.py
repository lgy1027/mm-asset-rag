"""Document-qrels retrieval metric tests."""

from __future__ import annotations

import pytest

from mm_asset_rag.metrics import (
    aggregate_metrics,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_uses_exact_document_ids_and_deduplicates_results() -> None:
    qrels = {"doc-a": 2, "doc-b": 1}

    assert recall_at_k(["doc-a", "doc-a", "doc-b"], qrels, 2) == 0.5
    assert recall_at_k(["DOC-A", "doc-b"], qrels, 2) == 0.5


def test_reciprocal_rank_uses_first_exact_relevant_document() -> None:
    assert reciprocal_rank(["guide", "doc-guide"], {"doc-guide": 3}) == 0.5
    assert reciprocal_rank(["guide"], {"doc-guide": 3}) == 0.0


def test_average_precision_counts_each_relevant_document_once() -> None:
    score = average_precision(["doc-a", "doc-a", "miss", "doc-b"], {"doc-a": 2, "doc-b": 1})

    assert score == pytest.approx((1.0 + 2 / 4) / 2)


def test_dcg_uses_graded_exponential_gain() -> None:
    score = dcg_at_k(["doc-low", "doc-high"], {"doc-high": 3, "doc-low": 1}, 2)

    assert score == pytest.approx(1 + 7 / 1.584962500721156)


def test_ndcg_rewards_higher_relevance_earlier() -> None:
    qrels = {"doc-high": 3, "doc-low": 1}
    ideal = ndcg_at_k(["doc-high", "doc-low"], qrels, 2)
    reversed_order = ndcg_at_k(["doc-low", "doc-high"], qrels, 2)

    assert ideal == 1.0
    assert reversed_order == pytest.approx(
        (1 + 7 / 1.584962500721156) / (7 + 1 / 1.584962500721156)
    )
    assert reversed_order < ideal


def test_metrics_ignore_non_positive_qrels_and_invalid_cutoffs() -> None:
    qrels = {"relevant": 1, "zero": 0, "negative": -1}

    assert recall_at_k(["zero", "relevant"], qrels, 1) == 0.0
    assert recall_at_k(["relevant"], qrels, 0) == 0.0
    assert dcg_at_k(["zero", "relevant"], qrels, 2) == pytest.approx(1 / 1.584962500721156)
    assert ndcg_at_k(["relevant"], {}, 5) == 0.0


def test_aggregate_metrics_returns_required_document_metrics() -> None:
    results = [
        {"actual_document_ids": ["doc-a", "miss"], "qrels": {"doc-a": 3}},
        {"actual_document_ids": ["miss", "doc-b"], "qrels": {"doc-b": 1}},
    ]

    out = aggregate_metrics(results, k_values=(1, 2))

    assert set(out) == {"recall", "mrr", "map", "ndcg"}
    assert out["recall"] == {1: 0.5, 2: 1.0}
    assert out["mrr"] == pytest.approx(0.75)
    assert out["map"] == pytest.approx(0.75)
    assert out["ndcg"][1] == pytest.approx(0.5)
    assert out["ndcg"][2] == pytest.approx((1 + 1 / 1.584962500721156) / 2)


def test_aggregate_metrics_handles_empty_results() -> None:
    assert aggregate_metrics([], k_values=(1, 5)) == {
        "recall": {1: 0.0, 5: 0.0},
        "mrr": 0.0,
        "map": 0.0,
        "ndcg": {1: 0.0, 5: 0.0},
    }
