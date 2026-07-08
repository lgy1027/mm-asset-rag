"""Tests for mm_asset_rag.retrieval.

The Qdrant client is replaced with a MagicMock that returns deterministic
hits so the merge / RRF fusion logic can be exercised offline.
"""

from __future__ import annotations

import pytest

from mm_asset_rag import retrieval
from mm_asset_rag.backends.qdrant_backend import RRF_K
from mm_asset_rag.schema import SearchHit


def _make_hit(asset_id: str, route: str, score: float, source_type: str = "pdf") -> SearchHit:
    return SearchHit(
        route=route,
        score=score,
        asset_id=asset_id,
        title=asset_id,
        source_type=source_type,
        source_path=f"{asset_id}.pdf",
        evidence=f"evidence-for-{asset_id}",
    )


# ─── RRF fusion (rank-based) ──────────────────────────────────────────────


def test_merge_hits_rrf_combines_routes_for_same_asset() -> None:
    """An asset surfacing in two routes sums the per-route RRF contributions."""
    groups = [
        [_make_hit("a", "text", 1.0)],
        [_make_hit("a", "text_to_image", 1.0)],
    ]
    weights = [0.6, 0.4]
    merged = retrieval.merge_hits(groups, weights, top_k=5)
    assert len(merged) == 1
    assert merged[0].asset_id == "a"
    assert sorted(merged[0].metadata["routes"]) == ["text", "text_to_image"]
    # rank=1 in each route: 0.6/(60+1) + 0.4/(60+1) = 1.0/61
    assert merged[0].score == pytest.approx(1.0 / (RRF_K + 1))


def test_merge_hits_rrf_top_rank_scores_higher_than_second() -> None:
    """Within a single route the rank-1 hit scores above the rank-2 hit."""
    groups = [[_make_hit(f"id{i}", "text", 1.0 / (i + 1)) for i in range(5)]]
    merged = retrieval.merge_hits(groups, [1.0], top_k=5)
    assert len(merged) == 5
    # highest raw score -> rank 1 -> highest RRF contribution
    assert merged[0].asset_id == "id0"
    assert merged[0].score > merged[1].score
    assert merged[0].score == pytest.approx(1.0 / (RRF_K + 1))
    assert merged[1].score == pytest.approx(1.0 / (RRF_K + 2))


def test_merge_hits_top_k() -> None:
    groups = [[_make_hit(f"id{i}", "text", 1.0 / (i + 1)) for i in range(5)]]
    merged = retrieval.merge_hits(groups, [1.0], top_k=2)
    assert len(merged) == 2
    # highest score first
    assert merged[0].score >= merged[1].score


def test_merge_hits_rrf_decouples_score_scales() -> None:
    """A route with small raw scores can still contribute high ranks.

    The old ``score/max`` normalisation coupled routes' scales: a route
    whose max was 0.3 would be normalised to 1.0 and dominate. RRF uses
    rank only, so a hit at rank 1 in a low-score route contributes the
    same as rank 1 in a high-score route.
    """
    groups = [
        [_make_hit("low_scale", "text", 0.05)],  # tiny score but rank 1
        [_make_hit("high_scale", "text_to_image", 0.99)],  # rank 1 too
    ]
    weights = [1.0, 1.0]
    merged = retrieval.merge_hits(groups, weights, top_k=5)
    # Both rank-1 hits get the same RRF contribution; two distinct assets.
    assert {h.asset_id for h in merged} == {"low_scale", "high_scale"}
    assert merged[0].score == merged[1].score  # equal RRF contributions


def test_merge_hits_same_asset_across_routes_sums_contributions() -> None:
    """An asset at rank 1 in both routes sums two 1/61 contributions."""
    groups = [
        [_make_hit("shared", "text", 0.8), _make_hit("other", "text", 0.7)],
        [_make_hit("shared", "text_to_image", 0.5)],
    ]
    merged = retrieval.merge_hits(groups, [0.5, 0.5], top_k=5)
    # shared: rank1 route1 + rank1 route2 = 0.5/61 + 0.5/61 = 1/61
    # other: rank2 route1 = 0.5/62
    shared = next(h for h in merged if h.asset_id == "shared")
    other = next(h for h in merged if h.asset_id == "other")
    assert shared.score == pytest.approx(0.5 / (RRF_K + 1) + 0.5 / (RRF_K + 1))
    assert other.score == pytest.approx(0.5 / (RRF_K + 2))
    assert shared.score > other.score


def test_merge_hits_skips_zero_score_hits() -> None:
    """A zero-score hit is skipped so it doesn't pollute the rank space."""
    groups = [[_make_hit("zero", "text", 0.0), _make_hit("real", "text", 0.5)]]
    merged = retrieval.merge_hits(groups, [1.0], top_k=5)
    # zero is skipped; real becomes rank 1
    assert {h.asset_id for h in merged} == {"real"}
    assert merged[0].score == pytest.approx(1.0 / (RRF_K + 1))


def test_merge_hits_skips_zero_weight_routes() -> None:
    """A route with weight 0 contributes nothing."""
    groups = [
        [_make_hit("a", "text", 1.0)],
        [_make_hit("b", "text_to_image", 1.0)],
    ]
    weights = [1.0, 0.0]
    merged = retrieval.merge_hits(groups, weights, top_k=5)
    # b's route is skipped entirely
    assert {h.asset_id for h in merged} == {"a"}


def test_merge_hits_min_score_soft_floor() -> None:
    """``min_score`` is a soft low-end guard on the tiny RRF score.

    RRF scores are ~0.0164, so a floor of 0.001 trims only the
    smallest contributions. The default 0.0 keeps everything.
    """
    groups = [
        [
            _make_hit("top", "text", 1.0),
            _make_hit("tail", "text", 0.0001),
        ]
    ]
    # default 0.0 keeps both
    merged_default = retrieval.merge_hits(groups, [1.0], top_k=5)
    assert {h.asset_id for h in merged_default} == {"top", "tail"}

    # floor above tail's contribution drops only tail
    tail_score = 1.0 / (RRF_K + 2)
    merged_floored = retrieval.merge_hits(
        groups, [1.0], top_k=5, min_score=tail_score + 1e-9
    )
    assert {h.asset_id for h in merged_floored} == {"top"}


def test_merge_hits_empty_groups() -> None:
    assert retrieval.merge_hits([], [], top_k=5) == []


def test_merge_hits_does_not_mutate_input() -> None:
    hit = _make_hit("a", "text", 1.0)
    groups = [[hit]]
    retrieval.merge_hits(groups, [1.0], top_k=5)
    assert hit.score == 1.0  # unchanged


def test_merge_hits_deterministic_tie_break() -> None:
    """Ties on raw score are broken by asset_id so ranking is stable."""
    groups = [[_make_hit("b", "text", 1.0), _make_hit("a", "text", 1.0)]]
    merged = retrieval.merge_hits(groups, [1.0], top_k=5)
    # Both rank 1 by score; asset_id "a" wins the tie-break.
    assert merged[0].asset_id == "a"


# ─── hybrid_search wiring ─────────────────────────────────────────────────


def test_hybrid_search_forwards_min_score(monkeypatch, fixed_vector) -> None:
    """``hybrid_search`` reads ``Settings.min_score`` and passes it to ``merge_hits``.

    RRF scores are tiny (~0.0164 for rank 1). The default ``min_score=0.0``
    keeps everything; a floor above the top hit's RRF score drops all.
    """
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_search",
        lambda query, top_k=5, **_: [
            _make_hit("a", "qdrant_text", 1.0),
            _make_hit("b", "qdrant_text", 0.3),
        ],
    )
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_to_image_search",
        lambda query, top_k=5, **_: [],
    )
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "hybrid_weight_text", 0.8)
    monkeypatch.setattr(settings, "hybrid_weight_text_to_image", 0.2)
    # Disable reranker so we test the merge path directly.
    monkeypatch.setattr(settings, "reranker_enabled", False)

    # Default 0.0 keeps both a (rank 1) and b (rank 2).
    monkeypatch.setattr(settings, "min_score", 0.0)
    hits = retrieval.hybrid_search("anything")
    assert {h.asset_id for h in hits} == {"a", "b"}

    # Floor above a's RRF score drops everything.
    a_score = 0.8 / (RRF_K + 1)
    monkeypatch.setattr(settings, "min_score", a_score + 0.001)
    hits = retrieval.hybrid_search("anything")
    assert hits == []


def test_hybrid_search_uses_qdrant_backend(monkeypatch, fixed_vector) -> None:
    text_hits = [_make_hit("a", "qdrant_text", 0.9)]
    text_to_image_hits = [_make_hit("b", "qdrant_text_to_image", 0.7)]

    # Disable reranker so the merge path is the only thing exercised.
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)

    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_search",
        lambda query, top_k=5: text_hits,
    )
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_to_image_search",
        lambda query, top_k=5: text_to_image_hits,
    )
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_image_to_image_search",
        lambda path, top_k=5: [],
    )

    hits = retrieval.hybrid_search("anything")
    assert {hit.asset_id for hit in hits} == {"a", "b"}


def test_hybrid_search_uses_settings_weights(monkeypatch, fixed_vector) -> None:
    """Weights passed to merge_hits must come from ``Settings``."""
    text_hits = [_make_hit("a", "qdrant_text", 1.0)]
    text_to_image_hits = [_make_hit("b", "qdrant_text_to_image", 1.0)]

    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)

    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_search",
        lambda query, top_k=5: text_hits,
    )
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_to_image_search",
        lambda query, top_k=5: text_to_image_hits,
    )

    captured: dict[str, list[float]] = {}

    # Capture the real merge_hits before monkeypatching it so the fake
    # wrapper doesn't recurse into itself.
    real_merge = retrieval.merge_hits

    def _fake_merge(groups, weights, top_k, **kwargs):
        captured["weights"] = list(weights)
        return real_merge(groups, weights, top_k, **kwargs)

    monkeypatch.setattr("mm_asset_rag.retrieval.merge_hits", _fake_merge)

    # Tighten text-to-image so a different score from the default is observable.
    monkeypatch.setattr(settings, "hybrid_weight_text", 0.70)
    monkeypatch.setattr(settings, "hybrid_weight_text_to_image", 0.30)

    retrieval.hybrid_search("anything")

    assert captured["weights"] == [0.70, 0.30]


def test_hybrid_search_skips_image_route_when_weight_zero(monkeypatch, fixed_vector) -> None:
    """Image-to-image route must not be called when its weight is <= 0."""
    text_hits = [_make_hit("a", "qdrant_text", 1.0)]
    text_to_image_hits = [_make_hit("b", "qdrant_text_to_image", 1.0)]

    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)

    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_search",
        lambda query, top_k=5: text_hits,
    )
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_to_image_search",
        lambda query, top_k=5: text_to_image_hits,
    )

    called = {"i2i": 0}

    def _track(path, top_k=5):
        called["i2i"] += 1
        return []

    monkeypatch.setattr("mm_asset_rag.retrieval.qdrant_image_to_image_search", _track)

    from pathlib import Path

    # Set image-to-image weight to 0.0; even when an image_path is
    # supplied, the route should be skipped to avoid wasted round-trips.
    monkeypatch.setattr(settings, "hybrid_weight_image_to_image", 0.0)

    retrieval.hybrid_search("q", image_path=Path("/tmp/nonexistent.png"))

    assert called["i2i"] == 0


def test_hybrid_search_calls_image_route_when_weight_positive(monkeypatch, fixed_vector) -> None:
    """Image-to-image route is consulted when its weight is > 0 and an image_path is given."""
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)

    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_search",
        lambda query, top_k=5: [_make_hit("a", "qdrant_text", 1.0)],
    )
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_to_image_search",
        lambda query, top_k=5: [_make_hit("b", "qdrant_text_to_image", 1.0)],
    )

    called = {"i2i": 0}

    def _track(path, top_k=5):
        called["i2i"] += 1
        return [_make_hit("c", "qdrant_image_to_image", 1.0)]

    monkeypatch.setattr("mm_asset_rag.retrieval.qdrant_image_to_image_search", _track)

    from pathlib import Path

    monkeypatch.setattr(settings, "hybrid_weight_image_to_image", 0.10)

    retrieval.hybrid_search("q", image_path=Path("/tmp/nonexistent.png"))

    assert called["i2i"] == 1


def test_hybrid_search_default_image_to_image_weight_is_positive(monkeypatch, fixed_vector) -> None:
    """The default image-to-image weight is now 0.15 so the route is consulted."""
    settings = retrieval.get_settings()
    assert settings.hybrid_weight_image_to_image == 0.15
    monkeypatch.setattr(settings, "reranker_enabled", False)

    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_search",
        lambda query, top_k=5: [],
    )
    monkeypatch.setattr(
        "mm_asset_rag.retrieval.qdrant_text_to_image_search",
        lambda query, top_k=5: [],
    )
    called = {"i2i": 0}

    def _track(path, top_k=5):
        called["i2i"] += 1
        return []

    monkeypatch.setattr("mm_asset_rag.retrieval.qdrant_image_to_image_search", _track)

    from pathlib import Path

    # Use the default weight (don't monkeypatch it).
    retrieval.hybrid_search("q", image_path=Path("/tmp/nonexistent.png"))
    assert called["i2i"] == 1
