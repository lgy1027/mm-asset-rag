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

from mm_asset_rag.embedders.reranker import (
    HttpRerankApiReranker,
    Reranker,
    RerankerError,
    get_default_reranker,
    reset_reranker,
)
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


def test_rerank_degrades_when_model_load_fails(tmp_home, monkeypatch):
    """Bug fix (degradation contract): if the cross-encoder fails to load or
    score at ``rerank`` time (corrupted cache, OOM, revoked HF weights), the
    call must not raise out of ``hybrid_search``. It returns the pre-rerank
    hits sorted by hybrid score (truncated to top_k) and marks the reranker
    unavailable for the rest of the process."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()
    import mm_asset_rag.embedders.reranker as mod

    hits = [
        _hit("A", "alpha", score=0.1),
        _hit("B", "beta", score=0.9),
    ]
    reranker = Reranker()
    with patch.object(Reranker, "_load", side_effect=RuntimeError("corrupted cache")):
        out = reranker.rerank("query", hits, top_k=5)
    # Degraded: returned in hybrid-score order, not crashed.
    assert [h.asset_id for h in out] == ["B", "A"]
    # Sticky-unavailable for the process.
    assert mod._UNAVAILABLE is True
    reset_reranker()


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
    with patch.object(Reranker, "_dep_available", return_value=True):
        reranker = get_default_reranker()
    assert reranker is not None


def test_get_default_reranker_none_when_dep_missing(tmp_home, monkeypatch):
    """Bug fix: when ``sentence_transformers`` is not importable, the reranker
    must report unavailable *before* ``hybrid_search`` commits to the two-stage
    path. Previously construction was import-free so a non-None Reranker was
    returned and ``rerank`` later raised ``ModuleNotFoundError`` out of the
    search call instead of degrading."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()
    with patch.object(Reranker, "_dep_available", return_value=False):
        assert get_default_reranker() is None
    # Sticky: a second call (dep still missing) does not retry the probe.
    with patch.object(Reranker, "_dep_available", return_value=True) as probe:
        assert get_default_reranker() is None
    probe.assert_not_called()
    reset_reranker()


def test_get_default_reranker_load_failure_is_sticky(tmp_home, monkeypatch):
    """A failed model load sets _UNAVAILABLE so subsequent calls don't retry."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()

    with (
        patch.object(Reranker, "_dep_available", return_value=True),
        patch("mm_asset_rag.embedders.reranker.Reranker", side_effect=RuntimeError("boom")),
    ):
        assert get_default_reranker() is None

    # Second call should not retry (sticky), even if we re-patch to succeed.
    with (
        patch.object(Reranker, "_dep_available", return_value=True),
        patch("mm_asset_rag.embedders.reranker.Reranker", return_value=MagicMock()),
    ):
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
        patch.object(Reranker, "_dep_available", return_value=True),
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


# ─── HTTP rerank API provider (siliconflow / dashscope) ────────────────────


def _fake_response(rows):
    """Build a fake requests.Response whose .json() returns a Cohere-form
    rerank payload (``results: [{index, relevance_score}, ...]``) — the flat
    shape SiliconFlow returns at top level."""
    resp = MagicMock()
    resp.json.return_value = {"results": rows}
    resp.raise_for_status.return_value = None
    return resp


def _fake_nested_response(rows):
    """Same payload wrapped under ``output.results`` — the DashScope-native
    (百炼) shape. Same row schema, different wrapper."""
    resp = MagicMock()
    resp.json.return_value = {"output": {"results": rows}, "request_id": "x"}
    resp.raise_for_status.return_value = None
    return resp


def test_http_rerank_reorders_results_by_index(tmp_home, monkeypatch):
    """The API returns results sorted by relevance, not by input position.
    The provider must reorder by ``index`` back to input order so the score
    aligns with the right hit — otherwise the blend fuses hit A's hybrid
    score with hit B's cross-encoder score."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()

    # Input order: A(0.9 hybrid), B(0.1 hybrid). Server returns B's index=1
    # first (higher relevance). Provider must hand back [score_A, score_B].
    hits = [_hit("A", "alpha", score=0.9), _hit("B", "beta", score=0.1)]

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        # Server ranks B (index 1) above A (index 0).
        return _fake_response(
            [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        )

    reranker = HttpRerankApiReranker()
    with patch("requests.post", side_effect=fake_post):
        out = reranker.rerank("query", hits, top_k=2)

    # Request shape is the Cohere form.
    assert captured["url"] == "https://example.test/v1/rerank"
    assert captured["json"]["query"] == "query"
    assert captured["json"]["documents"] == ["alpha", "beta"]
    assert captured["json"]["top_n"] == 2
    assert captured["json"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    # B's cross-encoder score (0.9) was aligned to hit B, not hit A — so B
    # now wins the blend (CE dominates at blend=0.6).
    assert out[0].asset_id == "B"
    assert out[0].metadata["rerank_score"] == 0.9
    assert out[1].asset_id == "A"
    assert out[1].metadata["rerank_score"] == 0.2


def test_http_rerank_missing_candidate_scores_zero(tmp_home, monkeypatch):
    """If the server drops a candidate (no result row for some index), the
    provider scores it 0.0 — min-max normalisation buries it instead of
    crashing the blend. Uses the 百炼 nested form (``output.results``)."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "dashscope")
    monkeypatch.setenv("RERANKER_API_MODEL", "qwen3-rerank")
    reset_reranker()

    hits = [_hit("A", "alpha", score=0.9), _hit("B", "beta", score=0.1)]
    reranker = HttpRerankApiReranker()
    with patch(
        "requests.post",
        return_value=_fake_nested_response([{"index": 1, "relevance_score": 0.8}]),
    ):
        out = reranker.rerank("query", hits, top_k=2)
    # A had no result row → score 0.0; B scored 0.8 → B wins.
    a = next(h for h in out if h.asset_id == "A")
    b = next(h for h in out if h.asset_id == "B")
    assert a.metadata["rerank_score"] == 0.0
    assert b.metadata["rerank_score"] == 0.8


def test_http_rerank_dashscope_sends_nested_body_and_reads_output(tmp_home, monkeypatch):
    """百炼 (dashscope) uses the DashScope-native *nested* wire shape, not the
    flat Cohere form: the request body is ``{model, input:{query, documents},
    parameters:{top_n, return_documents}}`` and the results live under
    ``output.results``. A flat-form request would 400. Verified against the
    live endpoint (qwen3-rerank, 2026-07)."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "dashscope")
    # Intentionally leave base/model at provider defaults.
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()  # pick up the env overrides above

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _fake_nested_response(
            [
                {"index": 0, "relevance_score": 0.7},
                {"index": 1, "relevance_score": 0.2},
            ]
        )

    reranker = HttpRerankApiReranker()
    hits = [_hit("A", "alpha", score=0.9), _hit("B", "beta", score=0.1)]
    with patch("requests.post", side_effect=fake_post):
        out = reranker.rerank("query", hits, top_k=2)

    # Default base is the DashScope-native endpoint (universal host).
    assert captured["url"] == (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    body = captured["json"]
    # Nested request shape, NOT flat.
    assert body["model"] == "qwen3-rerank"
    assert body["input"]["query"] == "query"
    assert body["input"]["documents"] == ["alpha", "beta"]
    assert body["parameters"]["top_n"] == 2
    assert body["parameters"]["return_documents"] is False
    # Scores read from ``output.results``, reordered by index → input order.
    assert out[0].asset_id == "A"
    assert out[0].metadata["rerank_score"] == 0.7
    assert out[1].asset_id == "B"
    assert out[1].metadata["rerank_score"] == 0.2


def test_http_rerank_skips_image_hits(tmp_home, monkeypatch):
    """Image hits are never sent to the text rerank API — CLIP is already a
    relevance signal. Only text hits appear in the ``documents`` list.

    (This was previously glued onto the end of the dashscope test above with
    no ``def`` line of its own — it ran as a silent side-effect under the
    wrong name. Split out so the image-skip contract has its own test.)"""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()

    text_hit = _hit("docA", "alpha doc", score=0.2)
    image_hit = _image_hit("imgB", score=0.35)

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["documents"] = json["documents"]
        return _fake_response([{"index": 0, "relevance_score": 0.9}])

    reranker = HttpRerankApiReranker()
    with patch("requests.post", side_effect=fake_post):
        out = reranker.rerank("query", [text_hit, image_hit], top_k=5)
    # Only the text hit was sent to the API.
    assert captured["documents"] == ["alpha doc"]
    # Image hit's raw CLIP score preserved in metadata.
    img = next(h for h in out if h.asset_id == "imgB")
    assert img.metadata["rerank_score"] == 0.35
    reset_reranker()


def test_http_rerank_degrades_on_api_error(tmp_home, monkeypatch):
    """An API failure (5xx / network) must not raise out of hybrid_search —
    it degrades to returning pre-rerank hits in hybrid order and marks the
    provider sticky-unavailable for the process. A 5xx is *transient* and is
    retried once before giving up."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    reset_reranker()
    import requests

    import mm_asset_rag.embedders.reranker as mod

    monkeypatch.setattr(mod, "_HTTP_RETRY_BACKOFF", 0.0)  # no real sleep on retry

    hits = [_hit("A", "alpha", score=0.1), _hit("B", "beta", score=0.9)]
    reranker = HttpRerankApiReranker()

    # A real requests.HTTPError wrapping a 503 response. The provider retries
    # once (transient), still fails, then degrades — not crashes.
    err_resp = MagicMock()
    err_resp.status_code = 503
    err = requests.exceptions.HTTPError("503 Server Error", response=err_resp)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise err

    with patch("requests.post", side_effect=boom):
        out = reranker.rerank("query", hits, top_k=5)
    # One retry happened (transient 5xx → 2 attempts).
    assert calls["n"] == 2
    # Degraded: hybrid-score order, not crashed.
    assert [h.asset_id for h in out] == ["B", "A"]
    assert mod._UNAVAILABLE is True
    reset_reranker()


def test_http_rerank_4xx_not_retried(tmp_home, monkeypatch):
    """A 4xx (auth / bad model) is a *config* error, not transient — it must
    not be retried; it goes straight to the degrade path."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    reset_reranker()
    import requests

    err_resp = MagicMock()
    err_resp.status_code = 401
    err = requests.exceptions.HTTPError("401 Unauthorized", response=err_resp)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise err

    reranker = HttpRerankApiReranker()
    with patch("requests.post", side_effect=boom):
        out = reranker.rerank("query", [_hit("A", "alpha", score=0.5)], top_k=1)
    # No retry on a config error.
    assert calls["n"] == 1
    assert [h.asset_id for h in out] == ["A"]
    reset_reranker()


def test_http_rerank_timeout_retried(tmp_home, monkeypatch):
    """A requests Timeout is transient → retried once before degrading."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    reset_reranker()
    import requests

    import mm_asset_rag.embedders.reranker as mod

    monkeypatch.setattr(mod, "_HTTP_RETRY_BACKOFF", 0.0)  # no real sleep on retry

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.exceptions.Timeout("read timed out")

    reranker = HttpRerankApiReranker()
    with patch("requests.post", side_effect=boom):
        out = reranker.rerank("query", [_hit("A", "alpha", score=0.5)], top_k=1)
    assert calls["n"] == 2  # one retry
    assert [h.asset_id for h in out] == ["A"]  # degraded, not crashed
    reset_reranker()


def test_http_rerank_siliconflow_default_base_used(tmp_home, monkeypatch):
    """Setting only ``RERANKER_PROVIDER=siliconflow`` + key (no explicit
    ``RERANKER_API_BASE``) must hit the provider's default base URL — the
    ``_HTTP_PROVIDER_DEFAULTS`` siliconflow entry is exercised."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _fake_response([{"index": 0, "relevance_score": 0.5}])

    reranker = HttpRerankApiReranker()
    with patch("requests.post", side_effect=fake_post):
        reranker.rerank("query", [_hit("A", "alpha", score=0.5)], top_k=1)
    assert captured["url"] == "https://api.siliconflow.cn/v1/rerank"
    reset_reranker()


def test_http_score_empty_documents_returns_empty(tmp_home, monkeypatch):
    """Contract: ``_score_text_pairs`` with no documents short-circuits to [].
    (``rerank`` upstream never passes empty text hits, but this is the
    documented boundary of the provider method.)"""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    reset_reranker()
    reranker = HttpRerankApiReranker()
    with patch("requests.post") as post:
        assert reranker._score_text_pairs("q", []) == []
    post.assert_not_called()


def test_http_rerank_falls_back_to_openai_api_key(tmp_home, monkeypatch):
    """When ``RERANKER_API_KEY`` is unset, the HTTP provider reuses
    ``OPENAI_API_KEY`` — same fallback as the embedding / LLM creds, so a
    single key configures the whole stack."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    # Intentionally do NOT set RERANKER_API_KEY.
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    reset_reranker()

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["auth"] = headers["Authorization"]
        return _fake_response([{"index": 0, "relevance_score": 0.5}])

    reranker = HttpRerankApiReranker()
    with patch("requests.post", side_effect=fake_post):
        reranker.rerank("query", [_hit("A", "alpha", score=0.5)], top_k=1)
    assert captured["auth"] == "Bearer shared-key"


def test_get_default_reranker_picks_http_provider(tmp_home, monkeypatch):
    """``get_default_reranker`` factory selects the HTTP class when the
    provider is siliconflow / dashscope, and the local class otherwise."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()
    # dashscope has a universal default base (the DashScope-native endpoint)
    # + default model, so only the key is required to be configured.
    monkeypatch.setenv("RERANKER_PROVIDER", "dashscope")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    r = get_default_reranker()
    assert isinstance(r, HttpRerankApiReranker)

    reset_reranker()
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    get_settings.cache_clear()
    with patch.object(Reranker, "_dep_available", return_value=True):
        r = get_default_reranker()
    assert type(r) is Reranker
    reset_reranker()


def test_http_reranker_unconfigured_returns_none(tmp_home, monkeypatch):
    """A misconfigured HTTP provider (no key) must report unavailable at the
    probe so ``get_default_reranker`` returns ``None`` and search skips the
    two-stage path — rather than silently constructing a reranker that 401s
    and degrades on every query."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    # No key, no OPENAI_API_KEY → unconfigured (base+model have defaults).
    reset_reranker()
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    assert get_default_reranker() is None
    # dep-missing is hard-sticky: ``_UNAVAILABLE_UNTIL`` pinned to 0.0 (never
    # auto-recovers — a missing key won't self-heal, needs a config change).
    import mm_asset_rag.embedders.reranker as mod

    assert mod._UNAVAILABLE is True
    assert mod._UNAVAILABLE_UNTIL == 0.0

    # dashscope likewise: has default base + model, so a missing key is the
    # only misconfig that matters now.
    monkeypatch.setenv("RERANKER_PROVIDER", "dashscope")
    get_settings.cache_clear()
    assert get_default_reranker() is None
    reset_reranker()


# ─── H1: soft/hard sticky + TTL recovery ─────────────────────────────────


def test_http_reranker_soft_sticky_recovers_after_ttl(tmp_home, monkeypatch):
    """H1: an HTTP provider failure soft-stickies for a TTL, then auto-recovers
    — a transient cloud outage shouldn't disable reranking until process
    restart. Within the TTL ``get_default_reranker`` returns None; after it,
    it re-probes and succeeds."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()
    import requests

    import mm_asset_rag.embedders.reranker as mod
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    fake_time = [0.0]
    monkeypatch.setattr(mod, "_now", lambda: fake_time[0])
    monkeypatch.setattr(mod, "_HTTP_RETRY_BACKOFF", 0.0)
    # Shrink the TTL so the test doesn't have to advance the clock far.
    monkeypatch.setattr(HttpRerankApiReranker, "_sticky_ttl", 10.0)

    # First call: construct succeeds, then a transient 503 → soft-sticky.
    r = get_default_reranker()
    assert r is not None
    err_resp = MagicMock()
    err_resp.status_code = 503
    err = requests.exceptions.HTTPError("503", response=err_resp)

    def boom(*a, **k):
        raise err

    with patch("requests.post", side_effect=boom):
        out = r.rerank("q", [_hit("A", "a", score=0.5)], top_k=1)
    assert [h.asset_id for h in out] == ["A"]  # degraded, not crashed
    assert mod._UNAVAILABLE is True
    assert mod._UNAVAILABLE_UNTIL > 0.0

    # Within TTL: get_default_reranker returns None (no re-probe)...
    fake_time[0] = 5.0
    assert get_default_reranker() is None
    # ...and the fresh TTL is NOT clobbered by that None-returning check — a
    # concurrent recovery attempt must not clear a TTL just set by another
    # thread's ``_mark_unavailable`` (the check+clear is atomic under ``_LOCK``).
    assert mod._UNAVAILABLE_UNTIL > 0.0
    assert mod._UNAVAILABLE is True

    # After TTL: auto-recover, re-probe constructs a fresh instance.
    fake_time[0] = 11.0
    r2 = get_default_reranker()
    assert r2 is not None
    reset_reranker()


def test_local_reranker_hard_sticky_no_recovery(tmp_home, monkeypatch):
    """H1: a local provider failure hard-stickies — advancing the clock never
    auto-recovers (a corrupted HF cache / missing dep won't self-heal in a
    process lifetime). Only ``reset_reranker`` / a restart re-enables."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()
    import mm_asset_rag.embedders.reranker as mod

    fake_time = [0.0]
    monkeypatch.setattr(mod, "_now", lambda: fake_time[0])

    r = Reranker()
    with patch.object(Reranker, "_load", side_effect=RuntimeError("corrupt cache")):
        out = r.rerank("q", [_hit("A", "a", score=0.5)], top_k=1)
    assert [h.asset_id for h in out] == ["A"]  # degraded
    assert mod._UNAVAILABLE is True
    assert mod._UNAVAILABLE_UNTIL == 0.0  # hard sticky: no TTL

    # Way past any TTL: still None — local never auto-recovers.
    fake_time[0] = 9999.0
    assert get_default_reranker() is None
    reset_reranker()


# ─── M1 / M2: exception boundary ──────────────────────────────────────────


def test_http_rerank_bad_json_degrades(tmp_home, monkeypatch):
    """M1: a 200 with a non-JSON body (gateway error page, truncated response)
    degrades as a RerankerError — not retried, soft-sticky (not hard), and
    never crashes the search. Previously ``resp.json()`` sat outside the try
    and a JSONDecodeError would leak out of the broad caller as a hard sticky."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()
    import mm_asset_rag.embedders.reranker as mod

    monkeypatch.setattr(mod, "_HTTP_RETRY_BACKOFF", 0.0)

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.side_effect = ValueError("not JSON")
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        return resp

    reranker = HttpRerankApiReranker()
    with patch("requests.post", side_effect=fake_post):
        out = reranker.rerank(
            "q", [_hit("A", "alpha", score=0.1), _hit("B", "beta", score=0.9)], top_k=2
        )
    # Bad body is not transient → no retry.
    assert calls["n"] == 1
    # Degraded in hybrid-score order.
    assert [h.asset_id for h in out] == ["B", "A"]
    # HTTP provider → soft sticky, not hard.
    assert mod._UNAVAILABLE is True
    assert mod._UNAVAILABLE_UNTIL > 0.0
    reset_reranker()


def test_local_score_pairs_raises_reranker_error_on_provider_failure(tmp_home, monkeypatch):
    """M2 boundary: a local provider failure surfaces as ``RerankerError`` from
    ``_score_text_pairs`` — the type ``rerank()`` catches to degrade."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()
    reranker = Reranker()
    with (
        patch.object(Reranker, "_load", side_effect=RuntimeError("corrupt")),
        pytest.raises(RerankerError),
    ):
        reranker._score_text_pairs("q", ["a"])
    reset_reranker()


def test_rerank_programming_bug_propagates(tmp_home, monkeypatch):
    """M2: a programming bug (TypeError) from ``_score_text_pairs`` that is NOT
    a RerankerError must propagate out of ``rerank`` rather than be swallowed
    into a silent sticky-disable. Otherwise code bugs masquerade as provider
    failures."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    reset_reranker()
    import mm_asset_rag.embedders.reranker as mod

    reranker = Reranker()
    with (
        patch.object(Reranker, "_score_text_pairs", side_effect=TypeError("bug in my code")),
        pytest.raises(TypeError),
    ):
        reranker.rerank("q", [_hit("A", "a", score=0.5)], top_k=1)
    # No sticky-disable from a programming bug.
    assert mod._UNAVAILABLE is False
    reset_reranker()


# ─── L5: insecure base_url warning ─────────────────────────────────────────


def test_http_reranker_warns_insecure_base_url(tmp_home, monkeypatch, caplog):
    """L5: a plain-HTTP non-loopback ``reranker_api_base`` warns once about
    Bearer key in cleartext — same guard the other 4 base_url call sites use."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "http://insecure.example.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()
    from mm_asset_rag.answer import _warned_insecure_base_urls

    _warned_insecure_base_urls.clear()  # warn fresh this run (set is process-global)
    import logging

    caplog.set_level(logging.WARNING, logger="mm_asset_rag.answer")
    reranker = HttpRerankApiReranker()
    with patch(
        "requests.post", return_value=_fake_response([{"index": 0, "relevance_score": 0.5}])
    ):
        reranker.rerank("q", [_hit("A", "a", score=0.5)], top_k=1)
    assert any("明文 HTTP" in r.message for r in caplog.records)
    reset_reranker()


def test_http_reranker_loopback_base_not_warned(tmp_home, monkeypatch, caplog):
    """L5: a plain-HTTP *loopback* base (local ollama-style rerank proxy) does
    NOT warn — the guard only fires for non-loopback cleartext."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "http://localhost:8080/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()
    from mm_asset_rag.answer import _warned_insecure_base_urls

    _warned_insecure_base_urls.clear()
    import logging

    caplog.set_level(logging.WARNING, logger="mm_asset_rag.answer")
    reranker = HttpRerankApiReranker()
    with patch(
        "requests.post", return_value=_fake_response([{"index": 0, "relevance_score": 0.5}])
    ):
        reranker.rerank("q", [_hit("A", "a", score=0.5)], top_k=1)
    assert not any("明文 HTTP" in r.message for r in caplog.records)
    reset_reranker()


# ─── L7: explicit form dispatch ────────────────────────────────────────────


def test_http_rerank_unknown_form_raises(tmp_home, monkeypatch):
    """L7: an unknown wire form (a bad ``_HTTP_PROVIDER_DEFAULTS`` entry) is a
    programming error — it raises ValueError and propagates out of ``rerank``
    rather than being swallowed as a RerankerError sticky-disable."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "siliconflow")
    monkeypatch.setenv("RERANKER_API_BASE", "https://example.test/v1/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "sk-test")
    reset_reranker()
    import mm_asset_rag.embedders.reranker as mod

    monkeypatch.setitem(
        mod._HTTP_PROVIDER_DEFAULTS,
        "siliconflow",
        ("https://example.test/v1/rerank", "BAAI/bge-reranker-v2-m3", "weird"),
    )
    reranker = HttpRerankApiReranker()
    with pytest.raises(ValueError, match="unknown rerank wire form"):
        reranker.rerank("q", [_hit("A", "a", score=0.5)], top_k=1)
    # Not sticky-disabled (programming bug, not provider failure).
    assert mod._UNAVAILABLE is False
    reset_reranker()
