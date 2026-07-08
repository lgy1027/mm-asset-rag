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

import pytest

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
    # Final score is the blended (norm(CE) + norm(hybrid)) value, not the
    # raw cross-encoder score. With blend=0.6: B = 0.6*1.0 + 0.4*0.0 = 0.6,
    # A = 0.6*0.0 + 0.4*1.0 = 0.4 (CE ranks B>A, hybrid ranks A>B; CE wins).
    assert out[0].score == pytest.approx(0.6)
    assert out[1].score == pytest.approx(0.4)
    # Raw cross-encoder score preserved in metadata.
    assert out[0].metadata["rerank_score"] == 0.8
    assert out[1].metadata["rerank_score"] == 0.2
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


def test_get_default_reranker_disabled_by_default(tmp_home, monkeypatch):
    """Reranker is now enabled by default; disable via RERANKER_ENABLED=false."""
    reset_reranker()
    # Explicitly disable.
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    assert get_default_reranker() is None


def test_get_default_reranker_enabled_by_default(tmp_home):
    """The default is now enabled (latency for precision). The Reranker
    instance is constructed lazily and returned here without loading the
    model (model load only happens on the first ``rerank`` call)."""
    reset_reranker()
    reranker = get_default_reranker()
    assert reranker is not None


def test_get_default_reranker_load_failure_is_sticky(tmp_home, monkeypatch):
    """A failed model load sets _UNAVAILABLE so subsequent calls don't retry."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    with patch("mm_asset_rag.embedders.reranker.Reranker", side_effect=RuntimeError("boom")):
        assert get_default_reranker() is None

    # Second call should not retry (sticky), even if we re-patch to succeed.
    with patch("mm_asset_rag.embedders.reranker.Reranker", return_value=MagicMock()):
        assert get_default_reranker() is None

    # Clean up the sticky flag so it doesn't leak into later tests now
    # that reranker is enabled by default.
    reset_reranker()


def test_hybrid_search_skips_rerank_when_disabled(tmp_home, monkeypatch):
    """reranker off → hybrid_search behaves as before (no rerank call)."""
    from mm_asset_rag.retrieval import hybrid_search

    # The default is now enabled; explicitly disable for this test.
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    reset_reranker()
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()

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


# ─── image rerank split ────────────────────────────────────────────────────


def _image_hit(asset_id: str, score: float = 0.5) -> SearchHit:
    return SearchHit(
        route="qdrant_text_to_image",
        score=score,
        asset_id=asset_id,
        title=asset_id,
        source_type="image",
        source_path=f"images/{asset_id}.jpg",
        evidence=f"image caption {asset_id}",
        metadata={"page": 1},
    )


def test_rerank_skips_cross_encoder_for_image_hits(tmp_home, monkeypatch):
    """Image-source hits keep their original CLIP score; the text cross-encoder
    is only called for text/PDF hits."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    text_hit = _hit("docA", "alpha doc", score=0.2)
    image_hit = _image_hit("imgB", score=0.35)

    class FakeCE:
        def __init__(self):
            self.calls = 0

        def predict(self, pairs, show_progress_bar=False):
            self.calls += 1
            # Only one pair should reach the cross-encoder (the text hit).
            assert len(pairs) == 1
            return [0.9]

    ce = FakeCE()
    reranker = Reranker()
    with patch.object(Reranker, "_load", return_value=ce):
        out = reranker.rerank("query", [text_hit, image_hit], top_k=5)

    # Cross-encoder saw only the text hit.
    assert ce.calls == 1
    # Image hit's raw CLIP score is preserved in metadata; the hit.score
    # is now the blended value, not the raw CLIP score.
    img_out = next(h for h in out if h.asset_id == "imgB")
    assert img_out.metadata["rerank_score"] == 0.35
    assert img_out.metadata["hybrid_score"] == 0.35
    # Text hit: CE 0.9 (norm 1.0, the only text hit) + hybrid 0.2 (norm 0.0
    # vs the image's 0.35) → 0.6*1.0 + 0.4*0.0 = 0.6.
    text_out = next(h for h in out if h.asset_id == "docA")
    assert text_out.score == pytest.approx(0.6)
    assert text_out.metadata["rerank_score"] == 0.9


def test_rerank_image_only_hits_keep_scores(tmp_home, monkeypatch):
    """When all hits are images, the cross-encoder is never loaded."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    image_hits = [_image_hit(f"img{i}", score=0.3 + 0.01 * i) for i in range(3)]

    load_called = {"n": 0}

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            load_called["n"] += 1
            return [0.5] * len(pairs)

    reranker = Reranker()
    with patch.object(Reranker, "_load", return_value=FakeCE()) as mock_load:
        out = reranker.rerank("query", image_hits, top_k=5)

    # Cross-encoder never invoked (no text hits).
    mock_load.assert_not_called()
    assert [h.asset_id for h in out] == ["img2", "img1", "img0"]
    # Scores are blended (norm(CLIP) + norm(hybrid)); with a single signal
    # family the order is unchanged. img2 = 0.6*1.0 + 0.4*1.0 = 1.0.
    assert out[0].score == pytest.approx(1.0)
    assert out[0].metadata["rerank_score"] == 0.32


def test_rerank_image_and_text_unified_sort(tmp_home, monkeypatch):
    """Image and text hits are blended onto a common [0,1] scale and sorted
    together. The blend is scale-free, so a CLIP score and a cross-encoder
    logit compete by *rank within their signal family*, not raw magnitude.

    Here the text hit's CE (0.95) and the image hit's CLIP (0.30) each top
    their own family (norm=1.0); the image also tops the hybrid RRF
    (0.30 vs 0.20), so its blend (1.0) edges the text hit's (0.6).
    """
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    text_hit = _hit("docA", "alpha doc", score=0.2)
    image_hit = _image_hit("imgB", score=0.30)

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return [0.95]

    reranker = Reranker()
    with patch.object(Reranker, "_load", return_value=FakeCE()):
        out = reranker.rerank("query", [text_hit, image_hit], top_k=5)

    # Image tops both its CLIP family and the hybrid RRF → blend 1.0;
    # text tops CE but is last in hybrid → blend 0.6.
    assert out[0].asset_id == "imgB"
    assert out[0].score == pytest.approx(1.0)
    assert out[1].asset_id == "docA"
    assert out[1].score == pytest.approx(0.6)
    assert out[1].metadata["rerank_score"] == 0.95


def test_rerank_image_hit_uses_raw_clip_not_rrf_score(tmp_home, monkeypatch):
    """After ``merge_hits``, ``hit.score`` is the RRF contribution (~0.016)
    while the original CLIP cosine is preserved in ``metadata['raw_score']``.
    The reranker must blend the CLIP score, not the RRF score — otherwise the
    blend fuses two RRF signals on image hits and the CLIP relevance is lost.
    """
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    image_hit = SearchHit(
        route="qdrant_text_to_image",
        score=0.016,  # RRF contribution after merge_hits
        asset_id="imgB",
        title="imgB",
        source_type="image",
        source_path="images/imgB.jpg",
        evidence="caption imgB",
        metadata={"raw_score": 0.40},  # original CLIP cosine
    )

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return []

    reranker = Reranker()
    with patch.object(Reranker, "_load", return_value=FakeCE()):
        out = reranker.rerank("query", [image_hit], top_k=5)

    img = out[0]
    # rerank_score records the CLIP raw score (0.40), not the RRF score.
    assert img.metadata["rerank_score"] == 0.40
    # hybrid_score records the RRF score.
    assert img.metadata["hybrid_score"] == 0.016
