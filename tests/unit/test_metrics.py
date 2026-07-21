"""Tests for ``mm_asset_rag.metrics``.

Pure-Python; no fixture or external service required.
"""

from __future__ import annotations

import pytest

from mm_asset_rag.metrics import (
    _is_relevant,
    _log2,
    _normalize_id,
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


# ─── normalisation / substring relevance ────────────────────────────────


def test_normalize_id_strips_single_and_stacked_hash() -> None:
    """One layer and two stacked hash suffixes both collapse to the bare
    title; casefold + hyphen/space/underscore runs folded to ``-``."""
    assert _normalize_id("Alexnet_0c1c2b23") == "alexnet"
    # stacked: filename stem itself carried a hash, then upload added another
    assert _normalize_id("Resnext_69df8de4_903a9c76") == "resnext"
    assert _normalize_id("Attention Is All You Need_79e328a2") == "attention-is-all-you-need"
    assert _normalize_id("attention-is-all-you-need_0c713762") == "attention-is-all-you-need"


def test_normalize_id_keeps_non_hash_tail_intact() -> None:
    """A tail that isn't an all-lowercase 8-hex segment is not stripped —
    ``foo_longtail`` and ``a1b2c3d4`` (no separator) stay as the title."""
    assert _normalize_id("foo_longtail") == "foo-longtail"
    assert _normalize_id("CLIP") == "clip"
    assert _normalize_id("") == ""


def test_is_relevant_long_actual_matches_short_expected() -> None:
    """The data shape that broke self-consistency: a bare short expected
    title vs a long full returned id with an extra suffix. Must be a hit
    via bidirectional substring after normalisation."""
    actual = "Obsidian 的 10 大 AI Skill，第 1 名安装量居然 37 万！_924b37db"
    assert _is_relevant(actual, ["Obsidian 的 10 大 AI Skill"]) is True


def test_is_relevant_hyphenated_matches_spaced_title() -> None:
    """A filename-style hyphenated id matches the spaced paper title."""
    assert _is_relevant("attention-is-all-you-need_0c713762", ["Attention Is All You Need"]) is True


def test_is_relevant_no_false_positive_on_unrelated() -> None:
    """Substring relevance does not create false positives between the
    unrelated ids the current corpus actually contains."""
    assert _is_relevant("banana_split", ["kiwi"]) is False
    assert _is_relevant("resnext_0d7aafb7", ["Alexnet"]) is False
    assert _is_relevant("caltech-panda-01_3443a5d5", ["caltech-dolphin"]) is False


def test_is_relevant_known_short_token_limitation() -> None:
    """Documented limitation: a very short expected id that happens to appear
    *inside* an unrelated longer id (``bert`` inside ``robert``) matches
    under bidirectional substring. The current eval corpus has no such pair
    (no paper title is a substring of another's); this test pins the behaviour
    so a future tightening is a conscious decision, not a silent regression.
    If the corpus ever adds a title that contains a short expected token,
    add a length-guarded prefix check in ``_matches``."""
    assert _is_relevant("robert_abc12345", ["Bert"]) is True  # known limitation


def test_is_relevant_short_expected_substring_match_hits() -> None:
    """The legitimate short-expected cases: a bare expected title that is a
    substring (incl. prefix) of the returned full id still hits."""
    assert (
        _is_relevant(
            "Obsidian 的 10 大 AI Skill，第 1 名安装量居然 37 万！_924b37db",
            ["Obsidian 的 10 大 AI Skill"],
        )
        is True
    )
    assert _is_relevant("Caltech Panda 01_3443a5d5", ["Caltech Panda"]) is True


def test_hit_rate_substring_long_actual() -> None:
    """hit_rate honours the substring relevance, not just set equality."""
    long_id = "Obsidian 的 10 大 AI Skill，第 1 名安装量居然 37 万！_924b37db"
    assert hit_rate_at_k([long_id], ["Obsidian 的 10 大 AI Skill"], 5) == 1.0


def test_is_relevant_pair_hits_on_title_when_id_misses() -> None:
    """An ``(asset_id, title)`` pair hits when the title matches an expected
    paper-title id even though the filename-stem asset_id doesn't — the CLIP
    case: id ``clip_b14b418e`` vs expected ``Learning Transferable ...``."""
    pair = (
        "clip_b14b418e",
        "Learning Transferable Visual Models From Natural Language Supervision",
    )
    assert (
        _is_relevant(
            pair, ["Learning Transferable Visual Models From Natural Language Supervision"]
        )
        is True
    )
    # the pair also still matches when expected aligns with the asset_id
    assert _is_relevant(("clip_b14b418e", "Learning Transferable Visual Models"), ["clip"]) is True


def test_hit_rate_pair_uses_title() -> None:
    """hit_rate honours the (asset_id, title) pair — a title-only match counts."""
    pair = (
        "clip_b14b418e",
        "Learning Transferable Visual Models From Natural Language Supervision",
    )
    assert (
        hit_rate_at_k(
            [pair], ["Learning Transferable Visual Models From Natural Language Supervision"], 5
        )
        == 1.0
    )


# ─── bounds: metrics must stay ≤ 1.0 under multi-match ──────────────────


def test_ndcg_capped_at_one_with_duplicate_hash_variants() -> None:
    """Several returned ids that all match the same single expected must not
    push NDCG above 1.0 (the v0 regression: stacked-hash variants matched the
    same expected and inflated DCG past IDCG)."""
    actuals = ["Resnext_69df8de4_903a9c76", "Resnext_abc12345", "Resnext_def67890"]
    assert ndcg_at_k(actuals, ["Resnext"], 5) == pytest.approx(1.0)


def test_ap_capped_at_one_with_duplicate_hash_variants() -> None:
    actuals = ["Resnext_69df8de4_903a9c76", "Resnext_abc12345", "Resnext_def67890"]
    assert average_precision(actuals, ["Resnext"]) == pytest.approx(1.0)


def test_precision_counts_each_expected_once() -> None:
    """Two actuals both matching the only expected → precision 1/2, not 2/2."""
    assert precision_at_k(["Resnext_69df8de4_903a9c76", "Resnext_abc12345"], ["Resnext"], 5) == 0.5


def test_ndcg_two_distinct_expecteds_can_reach_one() -> None:
    """Sanity: when actuals match two *distinct* expecteds, NDCG can still be 1."""
    assert ndcg_at_k(["YOLO_abc12345", "SSD_def67890"], ["YOLO", "SSD"], 5) == 1.0


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
