"""Tests for mm_asset_rag.retrieval.

The Qdrant client is replaced with a MagicMock that returns deterministic
hits so the merge / normalize logic can be exercised offline.
"""

from __future__ import annotations

import pytest

from mm_asset_rag import retrieval
from mm_asset_rag.schema import SearchHit


def _make_hit(asset_id: str, route: str, score: float) -> SearchHit:
    return SearchHit(
        route=route,
        score=score,
        asset_id=asset_id,
        title=asset_id,
        source_type="pdf",
        source_path=f"{asset_id}.pdf",
        evidence=f"evidence-for-{asset_id}",
    )


def test_normalize_scores_empty() -> None:
    assert retrieval.normalize_scores([]) == []


def test_normalize_scores_divides_by_max() -> None:
    hits = [_make_hit("a", "text", 0.8), _make_hit("b", "text", 0.2)]
    normalized = retrieval.normalize_scores(hits)
    assert normalized[0].score == pytest.approx(1.0)
    assert normalized[1].score == pytest.approx(0.25)


def test_merge_hits_combines_routes_for_same_asset() -> None:
    groups = [
        [_make_hit("a", "text", 1.0)],
        [_make_hit("a", "text_to_image", 1.0)],
    ]
    weights = [0.6, 0.4]
    merged = retrieval.merge_hits(groups, weights, top_k=5)
    assert len(merged) == 1
    assert merged[0].asset_id == "a"
    assert sorted(merged[0].metadata["routes"]) == ["text", "text_to_image"]


def test_merge_hits_top_k() -> None:
    groups = [[_make_hit(f"id{i}", "text", 1.0 / (i + 1)) for i in range(5)]]
    merged = retrieval.merge_hits(groups, [1.0], top_k=2)
    assert len(merged) == 2
    # highest score first
    assert merged[0].score >= merged[1].score


def test_merge_hits_min_score_drops_low_confidence() -> None:
    """``min_score`` drops merged hits below the floor before ``top_k``."""
    groups = [
        [
            _make_hit("strong", "text", 1.0),
            _make_hit("weak", "text", 0.05),
        ]
    ]
    merged_default = retrieval.merge_hits(groups, [1.0], top_k=5)
    assert {h.asset_id for h in merged_default} == {"strong", "weak"}

    merged_floored = retrieval.merge_hits(groups, [1.0], top_k=5, min_score=0.10)
    assert {h.asset_id for h in merged_floored} == {"strong"}


def test_hybrid_search_forwards_min_score(monkeypatch, fixed_vector) -> None:
    """``hybrid_search`` reads ``Settings.min_score`` and passes it to ``merge_hits``."""
    # Use two hits at different raw scores so we can pick a floor
    # between their weighted scores. raw=1.0 -> normalised=1.0; raw=0.3
    # -> normalised=0.3; with weight 0.8 they become 0.80 and 0.24.
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

    # Floor at 0.30 keeps a (0.80) and b (0.24 is dropped, but b's
    # raw 0.3 * 0.8 = 0.24 < 0.30, so it should be filtered).
    monkeypatch.setattr(settings, "min_score", 0.30)
    hits = retrieval.hybrid_search("anything")
    assert {h.asset_id for h in hits} == {"a"}

    # Floor at 0.0 keeps both.
    monkeypatch.setattr(settings, "min_score", 0.0)
    hits = retrieval.hybrid_search("anything")
    assert {h.asset_id for h in hits} == {"a", "b"}

    # Floor past both weights drops everything.
    monkeypatch.setattr(settings, "min_score", 0.90)
    hits = retrieval.hybrid_search("anything")
    assert hits == []


def test_hybrid_search_uses_qdrant_backend(monkeypatch, fixed_vector) -> None:
    text_hits = [_make_hit("a", "qdrant_text", 0.9)]
    text_to_image_hits = [_make_hit("b", "qdrant_text_to_image", 0.7)]

    # This test exercises the *merge* path — explicitly disable the
    # min_score floor so the test scores survive the production default
    # of 0.30. (b's weighted score 0.7*0.2 = 0.14 would otherwise be
    # filtered.)
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "min_score", 0.0)

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

    settings = retrieval.get_settings()
    # Tighten text-to-image so a different score from the default is observable.
    monkeypatch.setattr(settings, "hybrid_weight_text", 0.70)
    monkeypatch.setattr(settings, "hybrid_weight_text_to_image", 0.30)

    retrieval.hybrid_search("anything")

    assert captured["weights"] == [0.70, 0.30]


def test_hybrid_search_skips_image_route_when_weight_zero(monkeypatch, fixed_vector) -> None:
    """Image-to-image route must not be called when its weight is <= 0."""
    text_hits = [_make_hit("a", "qdrant_text", 1.0)]
    text_to_image_hits = [_make_hit("b", "qdrant_text_to_image", 1.0)]

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

    settings = retrieval.get_settings()
    # Default image-to-image weight is 0.0; even when an image_path is
    # supplied, the route should be skipped to avoid wasted round-trips.
    monkeypatch.setattr(settings, "hybrid_weight_image_to_image", 0.0)

    retrieval.hybrid_search("q", image_path=Path("/tmp/nonexistent.png"))

    assert called["i2i"] == 0


def test_hybrid_search_calls_image_route_when_weight_positive(monkeypatch, fixed_vector) -> None:
    """Image-to-image route is consulted when its weight is > 0 and an image_path is given."""
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

    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "hybrid_weight_image_to_image", 0.10)

    retrieval.hybrid_search("q", image_path=Path("/tmp/nonexistent.png"))

    assert called["i2i"] == 1
