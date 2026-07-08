"""Tests for ``mm_asset_rag.backends.qdrant_backend``.

Covers the BM25 Okapi helpers used by ``_select_top_chunks_per_pdf`` —
pure functions, no Qdrant / no embedding model required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mm_asset_rag.backends.qdrant_backend import (
    _STOP_TOKENS,
    _bm25_okapi_scores,
    _filter_by_relevance,
    _has_any_token_overlap,
    _select_top_chunks_per_pdf,
    _tokenize_for_bm25,
    _tokenize_for_prefilter,
)
from mm_asset_rag.schema import ParsedDocument


def _doc(text: str, asset_id: str, title: str | None = None) -> ParsedDocument:
    return ParsedDocument(
        text=text,
        metadata={"asset_id": asset_id, "asset_title": title or asset_id},
    )


# ─── _tokenize_for_bm25 ─────────────────────────────────────────────────


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert _tokenize_for_bm25("LayoutLM: Pre-training of Text") == [
        "layoutlm",
        "pre",
        "training",
        "of",
        "text",
    ]


def test_tokenize_empty_and_punctuation_only() -> None:
    assert _tokenize_for_bm25("") == []
    assert _tokenize_for_bm25("!!! ... ---") == []


def test_tokenize_drops_empty_tokens() -> None:
    assert _tokenize_for_bm25("  BERT  ") == ["bert"]


# ─── _bm25_okapi_scores ─────────────────────────────────────────────────


def test_bm25_okapi_empty_inputs() -> None:
    assert _bm25_okapi_scores([], []) == []
    assert _bm25_okapi_scores(["bert"], []) == []


def test_bm25_okapi_relevant_doc_scores_higher() -> None:
    """A doc that contains all query terms must outrank one that contains none."""
    docs = [
        ["this", "is", "a", "passage", "about", "bert", "and", "transformers"],
        ["completely", "unrelated", "fish", "and", "chips"],
    ]
    scores = _bm25_okapi_scores(["bert", "transformer"], docs)
    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_bm25_okapi_idf_increases_with_rarity() -> None:
    """Terms that appear in fewer docs get a higher IDF contribution."""
    # "bert" appears in 1/3 docs; "the" appears in 3/3.
    docs = [
        ["bert", "lives", "here"],
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
    ]
    rare_score = _bm25_okapi_scores(["bert"], [docs[0]])[0]
    common_score = _bm25_okapi_scores(["the"], [docs[0]])[0]
    assert rare_score > common_score


# ─── _select_top_chunks_per_pdf ──────────────────────────────────────────


def test_select_top_chunks_returns_input_when_cap_is_none() -> None:
    docs = [_doc("a", "bert"), _doc("b", "bert"), _doc("c", "bert")]
    assert _select_top_chunks_per_pdf(docs, None) == docs


def test_select_top_chunks_returns_input_when_below_cap() -> None:
    docs = [_doc("a", "bert"), _doc("b", "bert")]
    assert _select_top_chunks_per_pdf(docs, 5) == docs
    # Input is not mutated.
    assert len(docs) == 2


def test_select_top_chunks_caps_oversized_pdf() -> None:
    """A 5-chunk PDF capped at 3 returns exactly 3 chunks for that asset."""
    docs = [
        _doc(
            "bert is the bidirectional encoder representation from transformers",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "we introduce a new language representation model called bert",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "bert achieves state of the art on eleven natural language "
            "processing tasks. completely unrelated cooking recipes follow",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "appendix: hyperparameter settings and additional ablations on "
            "the bert pretraining objective",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "this passage talks about penguins and arctic wildlife",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
    ]
    selected = _select_top_chunks_per_pdf(docs, max_per_pdf=3)
    assert len(selected) == 3
    # The lowest-scoring chunk (the off-topic penguins one) should be dropped.
    kept_texts = " ".join(d.text for d in selected)
    assert "penguins" not in kept_texts


def test_select_top_chunks_handles_multiple_assets_independently() -> None:
    """The cap is per-asset, not global."""
    bert_docs = [_doc(f"bert passage {i}", "bert") for i in range(4)]
    clip_docs = [_doc(f"clip passage {i}", "clip") for i in range(6)]
    selected = _select_top_chunks_per_pdf(bert_docs + clip_docs, max_per_pdf=3)
    {d.metadata["asset_id"]: d for d in selected}
    # Order is preserved within each asset group (per-asset cap), but
    # we just check counts here.
    counts = {}
    for d in selected:
        counts[d.metadata["asset_id"]] = counts.get(d.metadata["asset_id"], 0) + 1
    assert counts == {"bert": 3, "clip": 3}


def test_select_top_chunks_falls_back_on_empty_title() -> None:
    """An asset with no title falls back to the asset_id rewritten with spaces."""
    docs = [
        _doc("text", "layout_l_m", title=""),
        _doc("layout_l_m is great", "layout_l_m", title=""),
        _doc("unrelated", "layout_l_m", title=""),
    ]
    selected = _select_top_chunks_per_pdf(docs, max_per_pdf=2)
    assert len(selected) == 2


def test_select_top_chunks_does_not_mutate_input() -> None:
    docs = [
        _doc("alpha", "a"),
        _doc("beta", "a"),
        _doc("gamma", "a"),
        _doc("delta", "a"),
    ]
    original_order = [d.text for d in docs]
    _select_top_chunks_per_pdf(docs, max_per_pdf=2)
    assert [d.text for d in docs] == original_order


# ─── _filter_by_relevance ───────────────────────────────────────────────
# Used by the image search routes to drop Qdrant points whose cosine
# similarity is below the configured floor. Off-topic natural-language
# queries (e.g. "Schrödinger equation" against a photo collection) tend
# to score below the floor even for the closest image, so filtering
# returns an empty list instead of ten random Picsum photos.


class _StubPoint:
    def __init__(self, score: float | None, pid: str = "x") -> None:
        self.score = score
        self.id = pid


def test_filter_by_relevance_zero_threshold_keeps_everything() -> None:
    pts = [_StubPoint(0.0), _StubPoint(0.18), _StubPoint(0.5)]
    assert [p.id for p in _filter_by_relevance(pts, 0.0)] == ["x", "x", "x"]


def test_filter_by_relevance_drops_below_floor() -> None:
    pts = [
        _StubPoint(0.05, "a"),
        _StubPoint(0.21, "b"),
        _StubPoint(0.22, "c"),
        _StubPoint(0.30, "d"),
    ]
    assert [p.id for p in _filter_by_relevance(pts, 0.22)] == ["c", "d"]


def test_filter_by_relevance_handles_none_score() -> None:
    """Qdrant may report ``score=None`` for points without similarity."""
    pts = [_StubPoint(None, "a"), _StubPoint(0.30, "b")]
    assert [p.id for p in _filter_by_relevance(pts, 0.22)] == ["b"]


def test_filter_by_relevance_keeps_empty_input() -> None:
    assert _filter_by_relevance([], 0.22) == []


# ─── Sparse pre-filter for image search ─────────────────────────────────
# The pre-filter is a pure-Python token-overlap check on user-controlled
# payload fields. These tests cover the helpers without touching Qdrant.


def test_tokenize_for_prefilter_drops_short_and_stopwords() -> None:
    """Tokens shorter than the min length and English stopwords are dropped."""
    tokens = _tokenize_for_prefilter("The fish in the jpg image of Linux logo")
    # "the", "in", "jpg", "image" are dropped; the semantic words remain.
    assert "the" not in tokens
    assert "in" not in tokens
    assert "jpg" not in tokens
    assert "image" not in tokens
    assert "fish" in tokens
    assert "linux" in tokens
    assert "logo" in tokens


def test_tokenize_for_prefilter_is_lowercase_and_alpha_only() -> None:
    tokens = _tokenize_for_prefilter("Hello, World! 2024 Q1")
    assert tokens == {"hello", "world", "2024"}


def test_tokenize_for_prefilter_handles_empty_and_punctuation() -> None:
    assert _tokenize_for_prefilter("") == set()
    assert _tokenize_for_prefilter("!!! ... ---") == set()


def test_stopwords_are_universal_no_project_terms() -> None:
    """The stopword set is intentionally generic — no project-specific words."""
    forbidden = {"caltech", "caltech101", "sample", "category", "license"}
    assert not (_STOP_TOKENS & forbidden), (
        f"stopword set leaked project terms: {(_STOP_TOKENS & forbidden)!r}"
    )


def test_has_any_token_overlap_no_overlap() -> None:
    """A query with no shared tokens → no overlap."""
    index = {"img1": {"fish", "logo"}, "img2": {"bird"}}
    assert _has_any_token_overlap(_tokenize_for_prefilter("schrödinger equation"), index) is False


def test_has_any_token_overlap_exact_match() -> None:
    index = {"img1": {"fish", "logo"}, "img2": {"bird"}}
    assert _has_any_token_overlap(_tokenize_for_prefilter("logo design"), index) is True


def test_has_any_token_overlap_handles_plurals_via_substring() -> None:
    """``"airplane"`` matches ``"airplanes"`` via substring containment."""
    index = {"img1": {"airplanes"}}
    assert _has_any_token_overlap(_tokenize_for_prefilter("airplane"), index) is True


def test_has_any_token_overlap_short_token_substring_does_not_match() -> None:
    """Short tokens like ``"in"`` never make it into the index because the
    index builder runs every value through ``_tokenize_for_prefilter``,
    which drops tokens below the min length. Verify that a tokenised
    index does not retain ``"in"`` and therefore cannot match.
    """
    index = {"img1": _tokenize_for_prefilter("box in scene png")}
    assert "in" not in index["img1"]
    assert _has_any_token_overlap(_tokenize_for_prefilter("vintage"), index) is False


def test_has_any_token_overlap_empty_inputs() -> None:
    index = {"img1": {"fish"}}
    assert _has_any_token_overlap(set(), index) is False
    assert _has_any_token_overlap(_tokenize_for_prefilter("fish"), {}) is False


# ─── get_qdrant_client singleton ─────────────────────────────────────────


def test_get_qdrant_client_returns_singleton(tmp_path, monkeypatch) -> None:
    """Two calls in the same process should return the same client.

    Without the singleton, two threads that both call
    ``get_qdrant_client`` would each construct a fresh ``QdrantClient``,
    and qdrant-client's local mode would refuse the second one
    (``Storage folder already accessed``).
    """
    from mm_asset_rag.backends import qdrant_backend

    # Redirect indexes_dir so the test uses a private storage location.
    monkeypatch.setattr(
        "mm_asset_rag.backends.qdrant_backend.get_indexes_dir",
        lambda: tmp_path / "indexes",
    )
    qdrant_backend.reset_qdrant_client_cache()
    try:
        c1 = qdrant_backend.get_qdrant_client()
        c2 = qdrant_backend.get_qdrant_client()
        assert c1 is c2
    finally:
        qdrant_backend.reset_qdrant_client_cache()


def test_get_qdrant_client_resets_after_reset(tmp_path, monkeypatch) -> None:
    """``reset_qdrant_client_cache()`` drops the cached instance so a
    subsequent call returns a new client (used by tests)."""
    from mm_asset_rag.backends import qdrant_backend

    monkeypatch.setattr(
        "mm_asset_rag.backends.qdrant_backend.get_indexes_dir",
        lambda: tmp_path / "indexes",
    )
    qdrant_backend.reset_qdrant_client_cache()
    c1 = qdrant_backend.get_qdrant_client()
    qdrant_backend.reset_qdrant_client_cache()
    c2 = qdrant_backend.get_qdrant_client()
    assert c1 is not c2
    qdrant_backend.reset_qdrant_client_cache()


# ─── Embedder sparse / ColBERT capability probes ────────────────────────────
# The probes are model-agnostic: the OpenAI-compatible TextEmbedder returns
# False for both (it does not implement the methods), while a stub that
# implements them and returns non-None on a probe returns True.


class _NoSparseEmbedder:
    """Stand-in for the OpenAI TextEmbedder — no sparse/colbert methods."""

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class _BgeM3StubEmbedder:
    """Stand-in for a bge-m3 embedder — implements sparse + colbert."""

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_text_sparse(self, text: str):
        return {"indices": [1, 2], "values": [0.5, 0.5]}

    def embed_text_colbert(self, text: str):
        return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class _BgeM3StubReturningNoneEmbedder:
    """A bge-m3 embedder whose probe returns None (model not actually m3)."""

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_text_sparse(self, text: str):
        return None

    def embed_text_colbert(self, text: str):
        return None


def test_embedder_sparse_capability_openai_embedder_is_false() -> None:
    """The OpenAI-compatible embedder (no ``embed_text_sparse``) → False."""
    from mm_asset_rag.backends.qdrant_backend import _embedder_sparse_capability

    assert _embedder_sparse_capability(_NoSparseEmbedder()) is False


def test_embedder_colbert_capability_openai_embedder_is_false() -> None:
    from mm_asset_rag.backends.qdrant_backend import _embedder_colbert_capability

    assert _embedder_colbert_capability(_NoSparseEmbedder()) is False


def test_embedder_sparse_capability_bge_m3_stub_is_true(monkeypatch) -> None:
    from mm_asset_rag.backends.qdrant_backend import _embedder_sparse_capability

    # auto (default) → probe returns non-None → True
    assert _embedder_sparse_capability(_BgeM3StubEmbedder()) is True


def test_embedder_colbert_capability_bge_m3_stub_is_true(monkeypatch) -> None:
    from mm_asset_rag.backends.qdrant_backend import _embedder_colbert_capability

    assert _embedder_colbert_capability(_BgeM3StubEmbedder()) is True


def test_embedder_sparse_capability_probe_returns_none_is_false(monkeypatch) -> None:
    from mm_asset_rag.backends.qdrant_backend import _embedder_sparse_capability

    # The method exists but returns None on the probe → not supported.
    assert _embedder_sparse_capability(_BgeM3StubReturningNoneEmbedder()) is False


def test_embedder_sparse_capability_force_false(monkeypatch) -> None:
    from mm_asset_rag.backends.qdrant_backend import _embedder_sparse_capability

    monkeypatch.setenv("EMBEDDING_SPARSE_ENABLED", "false")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    assert _embedder_sparse_capability(_BgeM3StubEmbedder()) is False


def test_embedder_colbert_capability_force_false(monkeypatch) -> None:
    from mm_asset_rag.backends.qdrant_backend import _embedder_colbert_capability

    monkeypatch.setenv("EMBEDDING_COLBERT_ENABLED", "false")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    assert _embedder_colbert_capability(_BgeM3StubEmbedder()) is False


def test_embedder_sparse_capability_force_true_on_unsupported_is_false(monkeypatch) -> None:
    """Force-true on an embedder without the method is still False."""
    from mm_asset_rag.backends.qdrant_backend import _embedder_sparse_capability

    monkeypatch.setenv("EMBEDDING_SPARSE_ENABLED", "true")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    assert _embedder_sparse_capability(_NoSparseEmbedder()) is False


# ─── text_to_image prefilter no longer hard-cuts ───────────────────────────
# The pre-filter index is still built, but a zero-overlap query no longer
# returns an empty list — CLIP recall is allowed to proceed. Verify the
# helper is still available (for a future boosting step) but the
# ``qdrant_text_to_image_search`` path does not short-circuit on it.


def test_has_any_token_overlap_still_available() -> None:
    """``_has_any_token_overlap`` remains for a future boosting step."""
    index = {"img1": {"fish", "logo"}}
    assert _has_any_token_overlap(_tokenize_for_prefilter("logo"), index) is True
    assert _has_any_token_overlap(_tokenize_for_prefilter("schrödinger"), index) is False


def test_text_to_image_does_not_short_circuit_on_zero_overlap(
    monkeypatch, fake_qdrant_client
) -> None:
    """A query with zero token overlap must still call Qdrant (CLIP recall).

    The previous behaviour returned [] immediately; now CLIP recall is
    allowed to proceed and the relevance-threshold floor is the sole
    precision control.
    """
    from mm_asset_rag.backends import qdrant_backend

    # Build a tag index that has no overlap with the query.
    monkeypatch.setattr(
        qdrant_backend,
        "_load_image_tag_index",
        lambda: {"img1": {"mountain"}},
    )
    monkeypatch.setattr(
        qdrant_backend,
        "_tokenize_for_prefilter",
        lambda text: {"schrödinger"},
    )

    # The fake qdrant client returns empty points by default; we just
    # verify ``query_points`` was called (not short-circuited).
    fake_qdrant_client.query_points.return_value = MagicMock(points=[])

    monkeypatch.setattr(qdrant_backend, "get_qdrant_client", lambda: fake_qdrant_client)
    monkeypatch.setattr(qdrant_backend, "image_collection", lambda dim: "multimodal_image_512d")

    class _Provider:
        def embed_text(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(qdrant_backend, "get_default_image_embedder", lambda: _Provider())

    out = qdrant_backend.qdrant_text_to_image_search("schrödinger equation", top_k=5)
    # Did not short-circuit: query_points was called.
    assert fake_qdrant_client.query_points.called
    assert out == []
