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


def test_hybrid_search_uses_qdrant_backend(monkeypatch, fixed_vector) -> None:
    text_hits = [_make_hit("a", "qdrant_text", 0.9)]
    text_to_image_hits = [_make_hit("b", "qdrant_text_to_image", 0.7)]

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

    def _fake_merge(groups, weights, top_k):
        captured["weights"] = list(weights)
        return real_merge(groups, weights, top_k)

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
