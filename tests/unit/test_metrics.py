"""Tests for ``mm_asset_rag.metrics``.

Pure-Python; no fixture or external service required.
"""

from __future__ import annotations

import pytest

from mm_asset_rag.metrics import (
    _log2,
    aggregate_metrics,
    average_precision,
    dcg_at_k,
    f1_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ─── _log2 helper ────────────────────────────────────────────────────────


def test_log2_matches_math_log2() -> None:
    import math

    for x in (1, 2, 3, 4, 5, 10, 100):
        assert _log2(x) == pytest.approx(math.log2(x))


# ─── hit_rate_at_k ──────────────────────────────────────────────────────


def test_hit_rate_full_match() -> None:
    assert hit_rate_at_k(["a", "b", "c"], ["a"], 5) == 1.0


def test_hit_rate_zero_when_missed() -> None:
    assert hit_rate_at_k(["x", "y", "z"], ["a"], 5) == 0.0


def test_hit_rate_k_cutoff() -> None:
    """Expected at rank > k is a miss."""
    assert hit_rate_at_k(["x", "y", "a"], ["a"], 2) == 0.0
    assert hit_rate_at_k(["x", "a", "y"], ["a"], 2) == 1.0


def test_hit_rate_empty_expected_is_zero() -> None:
    assert hit_rate_at_k(["a", "b"], [], 5) == 0.0


# ─── reciprocal_rank ────────────────────────────────────────────────────


def test_rr_first_position() -> None:
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_rr_third_position() -> None:
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_rr_no_match_returns_zero() -> None:
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


# ─── precision_at_k ────────────────────────────────────────────────────


def test_precision_at_k_full() -> None:
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_precision_at_k_half() -> None:
    """Top-2 = 1 hit / 2 = 0.5."""
    assert precision_at_k(["a", "x"], ["a"], 2) == 0.5


def test_precision_at_k_short_result() -> None:
    assert precision_at_k(["a"], ["a", "b"], 5) == 1.0


def test_precision_at_k_zero_expected() -> None:
    assert precision_at_k(["a"], [], 5) == 0.0


# ─── recall_at_k ────────────────────────────────────────────────────────


def test_recall_at_k_full() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0


def test_recall_at_k_partial() -> None:
    assert recall_at_k(["a", "x"], ["a", "b"], 5) == 0.5


def test_recall_at_k_zero_expected() -> None:
    assert recall_at_k(["a"], [], 5) == 0.0


# ─── f1_at_k ───────────────────────────────────────────────────────────


def test_f1_harmonic_mean() -> None:
    """P=0.5 (1/2 hit), R=1.0 (1/1 expected) → F1 = 2/3 ≈ 0.667."""
    assert f1_at_k(["a", "x"], ["a"], 2) == pytest.approx(2 / 3)


def test_f1_zero_when_both_zero() -> None:
    assert f1_at_k(["x", "y"], ["a"], 2) == 0.0


# ─── average_precision ──────────────────────────────────────────────────


def test_ap_perfect_ranking() -> None:
    """AP with 2 relevant at ranks 1 and 2 → (1/1 + 2/2)/2 = 1.0."""
    assert average_precision(["a", "b", "x"], ["a", "b"]) == 1.0


def test_ap_mixed_ranking() -> None:
    """1 hit at rank 1, miss at rank 2, hit at rank 3 → (1/1 + 2/3)/2 ≈ 0.833."""
    score = average_precision(["a", "x", "b"], ["a", "b"])
    assert score == pytest.approx((1 + 2 / 3) / 2)


# ─── dcg / ndcg ─────────────────────────────────────────────────────────


def test_dcg_perfect() -> None:
    score = dcg_at_k(["a", "b"], ["a", "b"], 2)
    assert score == pytest.approx(1 + 1 / _log2(3))


def test_ndcg_perfect_is_one() -> None:
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)


def test_ndcg_miss_is_zero() -> None:
    assert ndcg_at_k(["x", "y"], ["a"], 5) == 0.0


def test_ndcg_half() -> None:
    """1 hit at rank 2 (vs ideal rank 1) → 0.5 NDCG."""
    # dcg  = 1 / log2(3) ≈ 0.631
    # idcg = 1 (one ideal hit at rank 1)
    # ndcg = 0.631
    score = ndcg_at_k(["x", "a"], ["a"], 5)
    assert score == pytest.approx(1 / _log2(3))


# ─── aggregate_metrics ─────────────────────────────────────────────────


def test_aggregate_returns_all_metrics() -> None:
    results = [
        {"actual_ids": ["a", "b"], "expected_ids": ["a"]},
        {"actual_ids": ["x", "y"], "expected_ids": ["a"]},
    ]
    out = aggregate_metrics(results, k_values=(1, 5))
    for metric in ("hit_rate", "precision", "recall", "f1", "ndcg"):
        assert metric in out
        for k in (1, 5):
            assert k in out[metric]
    assert "mrr" in out
    assert "map" in out


def test_aggregate_handles_empty_results() -> None:
    out = aggregate_metrics([], k_values=(1, 5))
    # All metrics are 0 when there's nothing to measure.
    for metric in ("hit_rate", "precision", "recall", "f1", "ndcg"):
        for k in (1, 5):
            assert out[metric][k] == 0.0
    assert out["mrr"] == 0.0
    assert out["map"] == 0.0


def test_aggregate_averages_across_results() -> None:
    results = [
        {"actual_ids": ["a", "x"], "expected_ids": ["a"]},  # P=0.5, R=1
        {"actual_ids": ["x", "y"], "expected_ids": ["a"]},  # P=0, R=0
    ]
    out = aggregate_metrics(results, k_values=(2,))
    assert out["hit_rate"][2] == pytest.approx(0.5)  # 1 hit / 2 cases
    assert out["precision"][2] == pytest.approx(0.25)
    assert out["recall"][2] == pytest.approx(0.5)
    assert out["f1"][2] == pytest.approx(1 / 3)
    assert out["mrr"] == pytest.approx(0.5)  # first case hit at rank 1
    assert out["map"] == pytest.approx(0.5)  # AP = 1/1 for case 1
