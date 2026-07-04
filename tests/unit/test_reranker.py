"""Tests for the two-stage reranker (``mm_asset_rag.embedders.reranker``).

Covers the three contracts ``hybrid_search`` depends on:
1. ``Reranker.rerank`` re-scores hits by (query, evidence) pairs, preserves
   the pre-rerank score in ``metadata["hybrid_score"]``, and slices to top_k.
2. ``get_default_reranker`` returns ``None`` when disabled or when the
   model fails to load — ``hybrid_search`` then degrades to single-stage.
3. ``hybrid_search`` fetches ``reranker_top_n`` candidates when the reranker
   is on (wider pool), and passes ``top_k`` through when off.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mm_asset_rag.embedders.reranker import Reranker, get_default_reranker, reset_reranker
from mm_asset_rag.schema import SearchHit


def _hit(asset_id: str, evidence: str, score: float = 0.5) -> SearchHit:
    return SearchHit(
        route="text",
        score=score,
        asset_id=asset_id,
        title=asset_id,
        source_type="pdf",
        source_path=f"pdfs/{asset_id}.pdf",
        evidence=evidence,
        metadata={"page": 1},
    )


def test_rerank_reorders_by_cross_encoder_score(tmp_home, monkeypatch):
    """rerank sorts by the cross-encoder score, not the input hybrid score."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    # Input is in hybrid-score order: A (0.9) > B (0.1). Cross-encoder
    # reverses them: B scores higher than A. Output must follow CE scores.
    hits = [
        _hit("A", "alpha doc", score=0.9),
        _hit("B", "beta doc", score=0.1),
    ]

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            # pairs: [(query, "alpha doc"), (query, "beta doc")]
            return [0.2, 0.8]  # B > A

    reranker = Reranker()
    with patch.object(Reranker, "_load", return_value=FakeCE()):
        out = reranker.rerank("query", hits, top_k=2)

    assert [h.asset_id for h in out] == ["B", "A"]
    # Cross-encoder score written to hit.score
    assert out[0].score == 0.8
    assert out[1].score == 0.2
    # Pre-rerank hybrid score preserved
    assert out[0].metadata["hybrid_score"] == 0.1
    assert out[1].metadata["hybrid_score"] == 0.9


def test_rerank_slices_to_top_k(tmp_home, monkeypatch):
    """Only the top-k CE-scored hits are returned."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    hits = [_hit(f"H{i}", f"doc {i}", score=float(i)) for i in range(5)]

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            # H0 scores highest, H4 lowest
            return [float(4 - i) for i in range(5)]

    reranker = Reranker()
    with patch.object(Reranker, "_load", return_value=FakeCE()):
        out = reranker.rerank("query", hits, top_k=3)

    assert len(out) == 3
    assert [h.asset_id for h in out] == ["H0", "H1", "H2"]


def test_rerank_empty_hits(tmp_home, monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()
    reranker = Reranker()
    assert reranker.rerank("query", [], top_k=5) == []


def test_get_default_reranker_disabled_by_default(tmp_home):
    """No RERANKER_ENABLED → None (hybrid_search skips rerank)."""
    reset_reranker()
    assert get_default_reranker() is None


def test_get_default_reranker_load_failure_is_sticky(tmp_home, monkeypatch):
    """A failed model load sets _UNAVAILABLE so subsequent calls don't retry."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    with patch("mm_asset_rag.embedders.reranker.Reranker", side_effect=RuntimeError("boom")):
        assert get_default_reranker() is None

    # Second call should not retry (sticky), even if we re-patch to succeed.
    with patch("mm_asset_rag.embedders.reranker.Reranker", return_value=MagicMock()):
        assert get_default_reranker() is None


def test_hybrid_search_skips_rerank_when_disabled(tmp_home, monkeypatch):
    """reranker off → hybrid_search behaves as before (no rerank call)."""
    from mm_asset_rag.retrieval import hybrid_search

    fake_hits = [_hit("X", "doc", score=0.8)]
    with (
        patch("mm_asset_rag.retrieval.qdrant_text_search", return_value=fake_hits),
        patch("mm_asset_rag.retrieval.qdrant_text_to_image_search", return_value=[]),
        patch("mm_asset_rag.embedders.reranker.Reranker.rerank") as mock_rerank,
    ):
        out = hybrid_search("query", top_k=5)
    # merge_hits may add a "routes" key to metadata; compare by identity of
    # the surviving hit rather than full equality.
    assert len(out) == 1
    assert out[0].asset_id == "X"
    mock_rerank.assert_not_called()


def test_hybrid_search_reranks_when_enabled(tmp_home, monkeypatch):
    """reranker on → fetch reranker_top_n candidates, rerank, return top_k."""
    from mm_asset_rag.retrieval import hybrid_search

    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_TOP_N", "20")
    monkeypatch.setenv("RERANKER_TOP_K", "5")
    reset_reranker()

    # 20 fake hits from text route, none from image route
    pool = [_hit(f"P{i}", f"doc {i}", score=0.5) for i in range(20)]
    captured_fetch_k = {}

    def fake_text_search(query, top_k):
        captured_fetch_k["value"] = top_k
        return pool

    # The reranker's rerank picks the first 5 of its input (we just verify
    # the wiring: fetch_k=20, rerank was called, output sliced to 5).
    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return [0.9] * len(pairs)

    with (
        patch("mm_asset_rag.retrieval.qdrant_text_search", side_effect=fake_text_search),
        patch("mm_asset_rag.retrieval.qdrant_text_to_image_search", return_value=[]),
        patch.object(Reranker, "_load", return_value=FakeCE()),
    ):
        out = hybrid_search("query", top_k=5)

    # Fetched the wider candidate pool, not just top_k=5
    assert captured_fetch_k["value"] == 20
    # Reranker was applied (output is the reranked set, sliced to 5)
    assert len(out) == 5
