"""Tests for ``mm_asset_rag.backends.qdrant_backend``.

Covers the BM25 Okapi helpers used by ``_select_top_chunks_per_pdf`` —
pure functions, no Qdrant / no embedding model required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient, models

from mm_asset_rag.backends import qdrant_backend
from mm_asset_rag.backends.qdrant import client as qdrant_client
from mm_asset_rag.backends.qdrant import indexing as qdrant_indexing
from mm_asset_rag.backends.qdrant import search as qdrant_search
from mm_asset_rag.backends.qdrant_backend import (
    _bm25_okapi_scores,
    _filter_by_relevance,
    _select_top_chunks_per_pdf,
    _tokenize_for_bm25,
)
from mm_asset_rag.protocols import IndexBackend, SearchBackend, SearchFilter
from mm_asset_rag.registry import get_backend
from mm_asset_rag.schema import ParsedDocument


def _doc(text: str, asset_id: str, title: str | None = None) -> ParsedDocument:
    return ParsedDocument(
        text=text,
        metadata={"asset_id": asset_id, "asset_title": title or asset_id},
    )


def test_registered_qdrant_backend_implements_search_and_index_ports() -> None:
    backend = get_backend("qdrant")

    assert isinstance(backend, SearchBackend)
    assert isinstance(backend, IndexBackend)


def test_legacy_qdrant_text_search_reexports_adapter_implementation(monkeypatch) -> None:
    monkeypatch.setattr(qdrant_search, "text_search", lambda query, top_k=5: ["hit"])

    assert qdrant_backend.qdrant_text_search("needle") == ["hit"]


def test_text_index_payload_is_v2_allowlist_and_creates_native_policy_indexes(
    monkeypatch, fake_qdrant_client
) -> None:
    """Index rows must not leak parser-era asset IDs into Qdrant payloads."""
    from mm_asset_rag.knowledge_models import (
        AccessPolicy,
        Asset,
        Chunk,
        Document,
        DocumentVersion,
        Source,
    )

    source = Source(source_id="upload:report", uri="uploads/report.pdf")
    policy = AccessPolicy(
        collection="engineering",
        allowed_principals=("alice",),
        metadata={"department": "search"},
    )
    document = Document("report", "System report", source, policy)
    version = DocumentVersion.create(document, "a" * 64)
    chunk = Chunk.create(
        document_version=version,
        asset=Asset("a" * 64, "pdf", "pdfs/report.pdf"),
        ordinal=0,
        text="retrieval evidence",
        source=source,
        access_policy=policy,
        metadata={"asset_id": "legacy-leak", "section": "Summary"},
    )
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2]
    embedder.embed_batch.return_value = []
    monkeypatch.setattr(qdrant_indexing, "read_documents", lambda: [chunk])
    monkeypatch.setattr(qdrant_indexing, "get_default_text_embedder", lambda: embedder)
    monkeypatch.setattr(
        qdrant_indexing,
        "_embed_bm25",
        lambda texts: [qdrant_indexing.models.SparseVector(indices=[1], values=[1.0])],
    )
    monkeypatch.setattr(qdrant_indexing, "get_qdrant_client", lambda: fake_qdrant_client)
    monkeypatch.setattr(qdrant_indexing, "text_collection", lambda dim: "v2_text")
    monkeypatch.setattr(qdrant_indexing, "_create_collection", lambda *args, **kwargs: None)
    fake_qdrant_client.retrieve.return_value = []
    captured_points = []
    fake_qdrant_client.upsert.side_effect = lambda *, collection_name, points, wait: (
        captured_points.extend(points)
    )

    qdrant_indexing.build_qdrant_text_index(force_recreate=True)

    payload = captured_points[0].payload
    assert set(payload) == {
        "document_id",
        "title",
        "version_id",
        "version_number",
        "chunk_id",
        "ordinal",
        "text",
        "source_id",
        "source_uri",
        "source_provider",
        "source_type",
        "source_path",
        "collection",
        "allowed_principals",
        "metadata",
        "chunk_metadata",
        "cache_id",
    }
    assert "asset_id" not in repr(payload)
    assert payload["document_id"] == "report"
    assert payload["version_id"] == version.version_id
    assert payload["chunk_id"] == chunk.chunk_id
    from mm_asset_rag.paths import physical_cache_id

    assert payload["cache_id"] == physical_cache_id("pdfs/report.pdf")
    assert payload["metadata"] == {"department": "search"}
    indexed_fields = {
        call.kwargs["field_name"] for call in fake_qdrant_client.create_payload_index.call_args_list
    }
    assert {
        "document_id",
        "version_id",
        "chunk_id",
        "collection",
        "allowed_principals",
        "metadata.department",
    } <= indexed_fields


def test_qdrant_hit_keeps_physical_cache_id_for_answer_images() -> None:
    hit = qdrant_search._payload_to_hit(
        "text",
        0.9,
        {
            "document_id": "public-document",
            "cache_id": "physical-cache-key",
            "title": "Title",
            "source_type": "pdf",
            "source_path": "pdfs/file.pdf",
            "text": "body",
            "chunk_metadata": {"images": [{"path": "images/figure.png"}]},
        },
    )

    assert hit.asset_id == "public-document"
    assert hit.cache_id == "physical-cache-key"


def test_qdrant_payload_cache_id_distinguishes_same_stem_paths() -> None:
    from dataclasses import replace

    from mm_asset_rag.knowledge_models import (
        AccessPolicy,
        Asset,
        Chunk,
        Document,
        DocumentVersion,
        Source,
    )

    source = Source(source_id="upload:shared")
    policy = AccessPolicy(collection="team", allowed_principals=("alice",))
    document = Document("shared", "Shared", source, policy)
    version = DocumentVersion.create(document, "f" * 64)
    pdf = Chunk.create(
        document_version=version,
        asset=Asset("f" * 64, "pdf", "pdfs/shared.pdf"),
        ordinal=0,
        text="pdf",
        source=source,
        access_policy=policy,
    )
    office = replace(pdf, asset=Asset("f" * 64, "document", "documents/shared.pdf"))

    assert (
        qdrant_indexing._v2_payload(pdf)["cache_id"]
        != qdrant_indexing._v2_payload(office)["cache_id"]
    )


def test_native_policy_filter_is_applied_to_text_and_image_routes(
    monkeypatch, fake_qdrant_client
) -> None:
    """Every route supplies collection, metadata, and ACL predicates to Qdrant."""
    policy_filter = SearchFilter(
        collection="engineering", metadata={"department": "search"}, principal="alice"
    )
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", lambda: fake_qdrant_client)
    monkeypatch.setattr(qdrant_search, "image_collection", lambda dim: "v2_image")

    class _ImageProvider:
        def embed_text(self, query):
            return [0.1, 0.2]

        def embed_image(self, path):
            return [0.1, 0.2]

    monkeypatch.setattr(qdrant_search, "get_default_image_embedder", lambda: _ImageProvider())

    qdrant_search.qdrant_text_to_image_search("needle", search_filter=policy_filter)
    qdrant_search.qdrant_image_to_image_search(Path("needle.png"), search_filter=policy_filter)

    first_filter = fake_qdrant_client.query_points.call_args_list[0].kwargs["query_filter"]
    second_filter = fake_qdrant_client.query_points.call_args_list[1].kwargs["query_filter"]
    for native_filter in (first_filter, second_filter):
        assert native_filter is not None
        field_conditions = native_filter.must
        assert {(condition.key, condition.match.value) for condition in field_conditions[:2]} == {
            ("collection", "engineering"),
            ("metadata.department", "search"),
        }
        assert native_filter.should is not None


def test_native_public_acl_filter_rejects_a_missing_acl_payload() -> None:
    """Qdrant's explicit empty-list condition must not admit a missing ACL."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="acl_shape",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="acl_shape",
        points=[
            models.PointStruct(id=1, vector=[1.0, 0.0], payload={"collection": "team"}),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={"collection": "team", "allowed_principals": []},
            ),
            models.PointStruct(
                id=3,
                vector=[1.0, 0.0],
                payload={"collection": "team", "allowed_principals": ["alice"]},
            ),
        ],
    )

    results = client.query_points(
        collection_name="acl_shape",
        query=[1.0, 0.0],
        query_filter=qdrant_search._native_policy_filter(SearchFilter(collection="team")),
        limit=10,
    ).points

    assert [point.id for point in results] == [2]


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


# ─── qdrant_text_search source_type filter ───────────────────────────────
# The default (include_image_sources=False) filter must exclude image-source
# chunks but keep pdf AND document. An earlier implementation used
# ``must=[source_type == "pdf"]``, which silently dropped every document
# (docx/pptx/…) chunk — a freshly uploaded docx was indexed but never
# returned by search. This pins the must_not(image) form.


def test_qdrant_text_search_filter_excludes_image_keeps_pdf_and_document(monkeypatch) -> None:
    """The default text-search filter is ``must_not source_type == image``.

    We capture the ``filter`` argument passed to ``_hybrid_text_query`` and
    assert it carries a ``must_not`` clause matching ``image`` (not a
    ``must`` clause matching ``pdf``). This is the regression the fix guards:
    before the fix, a document chunk could never be returned.
    """
    from mm_asset_rag.backends import qdrant_backend

    captured: dict = {}

    def _fake_hybrid(*args, **kwargs):
        captured["filter"] = kwargs.get("text_filter")
        return []

    monkeypatch.setattr(qdrant_search, "_hybrid_text_query", _fake_hybrid)
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", MagicMock)
    # Stub the embedder + bm25 helpers so no network / model is touched.
    monkeypatch.setattr(qdrant_search, "get_default_text_embedder", lambda: _NoSparseEmbedder())
    monkeypatch.setattr(qdrant_search, "_embed_bm25", lambda texts: [{"indices": [], "values": []}])
    monkeypatch.setattr(
        qdrant_search, "_embed_bm25_zh_query", lambda q: {"indices": [], "values": []}
    )
    monkeypatch.setattr(qdrant_search, "_embedder_sparse_capability", lambda e: False)
    monkeypatch.setattr(qdrant_search, "_embedder_colbert_capability", lambda e: False)

    qdrant_backend.qdrant_text_search("query", top_k=5, include_image_sources=False)

    flt = captured["filter"]
    assert flt is not None, "default search must apply a source_type filter"
    # The filter excludes image — must_not, not must(pdf).
    assert flt.must_not, "filter should be must_not(image), not must(pdf)"
    cond = flt.must_not[0]
    assert cond.key == "source_type"
    assert cond.match.value == "image"


def test_qdrant_text_search_no_filter_when_include_image_sources(monkeypatch) -> None:
    """``include_image_sources=True`` disables the source_type filter entirely."""
    from mm_asset_rag.backends import qdrant_backend

    captured: dict = {}

    def _fake_hybrid(*args, **kwargs):
        captured["filter"] = kwargs.get("text_filter")
        return []

    monkeypatch.setattr(qdrant_search, "_hybrid_text_query", _fake_hybrid)
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", MagicMock)
    monkeypatch.setattr(qdrant_search, "get_default_text_embedder", lambda: _NoSparseEmbedder())
    monkeypatch.setattr(qdrant_search, "_embed_bm25", lambda texts: [{"indices": [], "values": []}])
    monkeypatch.setattr(
        qdrant_search, "_embed_bm25_zh_query", lambda q: {"indices": [], "values": []}
    )
    monkeypatch.setattr(qdrant_search, "_embedder_sparse_capability", lambda e: False)
    monkeypatch.setattr(qdrant_search, "_embedder_colbert_capability", lambda e: False)

    qdrant_backend.qdrant_text_search("query", top_k=5, include_image_sources=True)
    assert captured["filter"].must_not is None


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
        "mm_asset_rag.backends.qdrant.client.get_indexes_dir",
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
        "mm_asset_rag.backends.qdrant.client.get_indexes_dir",
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

    def embed(self, text: str) -> list[float]:
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


# ─── collection-missing degradation ──────────────────────────────────────
# A fresh install (or one that has only ingested one modality) has no
# matching collection yet. Each search route must degrade to an empty
# result instead of crashing hybrid_search — symmetric across text /
# text-to-image / image-to-image.


def test_qdrant_text_search_degrades_when_collection_missing(monkeypatch) -> None:
    """``_hybrid_text_query`` raising "not found" → empty list, not a raise."""
    from mm_asset_rag.backends import qdrant_backend

    def _raise_not_found(*args, **kwargs):
        raise ValueError("Collection `multimodal_text_2560d` not found")

    monkeypatch.setattr(qdrant_search, "_hybrid_text_query", _raise_not_found)
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", MagicMock)
    monkeypatch.setattr(qdrant_search, "get_default_text_embedder", lambda: _NoSparseEmbedder())
    monkeypatch.setattr(qdrant_search, "_embed_bm25", lambda texts: [{"indices": [], "values": []}])
    monkeypatch.setattr(
        qdrant_search, "_embed_bm25_zh_query", lambda q: {"indices": [], "values": []}
    )
    monkeypatch.setattr(qdrant_search, "_embedder_sparse_capability", lambda e: False)
    monkeypatch.setattr(qdrant_search, "_embedder_colbert_capability", lambda e: False)

    assert qdrant_backend.qdrant_text_search("query", top_k=5) == []


def test_qdrant_text_search_re_raises_non_missing_value_error(monkeypatch) -> None:
    """A ValueError that isn't "collection not found" must still propagate."""
    from mm_asset_rag.backends import qdrant_backend

    def _raise_other(*args, **kwargs):
        raise ValueError("totally unrelated error")

    monkeypatch.setattr(qdrant_search, "_hybrid_text_query", _raise_other)
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", MagicMock)
    monkeypatch.setattr(qdrant_search, "get_default_text_embedder", lambda: _NoSparseEmbedder())
    monkeypatch.setattr(qdrant_search, "_embed_bm25", lambda texts: [{"indices": [], "values": []}])
    monkeypatch.setattr(
        qdrant_search, "_embed_bm25_zh_query", lambda q: {"indices": [], "values": []}
    )
    monkeypatch.setattr(qdrant_search, "_embedder_sparse_capability", lambda e: False)
    monkeypatch.setattr(qdrant_search, "_embedder_colbert_capability", lambda e: False)

    with pytest.raises(ValueError, match="unrelated"):
        qdrant_backend.qdrant_text_search("query", top_k=5)


def test_qdrant_text_search_degrades_on_remote_404(monkeypatch) -> None:
    """Remote server mode (``QDRANT_URL``) raises ``UnexpectedResponse``
    (not ``ValueError``) with HTTP 404 for a missing collection. The text
    route must degrade to empty — the regression this guards: before the
    remote case was handled, the ``except ValueError`` couldn't catch
    ``UnexpectedResponse`` and ``hybrid_search`` crashed on a remote
    instance that had only ingested one modality."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    from mm_asset_rag.backends import qdrant_backend

    def _raise_remote_404(*args, **kwargs):
        raise UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"", headers={}
        )

    monkeypatch.setattr(qdrant_search, "_hybrid_text_query", _raise_remote_404)
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", MagicMock)
    monkeypatch.setattr(qdrant_search, "get_default_text_embedder", lambda: _NoSparseEmbedder())
    monkeypatch.setattr(qdrant_search, "_embed_bm25", lambda texts: [{"indices": [], "values": []}])
    monkeypatch.setattr(
        qdrant_search, "_embed_bm25_zh_query", lambda q: {"indices": [], "values": []}
    )
    monkeypatch.setattr(qdrant_search, "_embedder_sparse_capability", lambda e: False)
    monkeypatch.setattr(qdrant_search, "_embedder_colbert_capability", lambda e: False)

    assert qdrant_backend.qdrant_text_search("query", top_k=5) == []


def test_qdrant_image_to_image_search_degrades_when_collection_missing(
    monkeypatch, fake_qdrant_client
) -> None:
    """image→image route degrades to [] when the image collection is absent."""
    from mm_asset_rag.backends import qdrant_backend

    fake_qdrant_client.query_points.side_effect = ValueError(
        "Collection `multimodal_image_512d` not found"
    )
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", lambda: fake_qdrant_client)
    monkeypatch.setattr(qdrant_search, "image_collection", lambda dim: "multimodal_image_512d")

    class _Provider:
        def embed_image(self, path):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(qdrant_search, "get_default_image_embedder", lambda: _Provider())

    assert qdrant_backend.qdrant_image_to_image_search(Path("any.png"), top_k=5) == []


def test_qdrant_image_to_image_search_returns_empty_when_query_image_unencodable(
    monkeypatch, fake_qdrant_client
) -> None:
    """image→image 查询图无法编码时直接返回 [],不打 qdrant。"""
    from mm_asset_rag.backends import qdrant_backend

    monkeypatch.setattr(qdrant_search, "get_qdrant_client", lambda: fake_qdrant_client)

    class _UnencodableProvider:
        def embed_image(self, path):
            return None  # ImageEmbedderProtocol 契约:无法编码 → None

    monkeypatch.setattr(qdrant_search, "get_default_image_embedder", lambda: _UnencodableProvider())

    assert qdrant_backend.qdrant_image_to_image_search(Path("bad.png"), top_k=5) == []
    # 没去 qdrant 查
    assert fake_qdrant_client.query_points.call_count == 0


def test_is_collection_missing_predicate() -> None:
    """The shared predicate keys on Qdrant's ``not found`` signal.

    Both client modes must be recognised: local file mode raises a
    ``ValueError`` with ``not found`` in the message; remote server mode
    (``QDRANT_URL``) raises ``UnexpectedResponse`` with HTTP 404 (an
    ``ApiException`` subclass, *not* a ``ValueError``). Before the remote
    case was handled, a remote instance that had only ingested one modality
    crashed ``hybrid_search`` instead of returning an empty route.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    from mm_asset_rag.backends.qdrant_backend import _is_collection_missing

    # Local file mode: ValueError with "not found".
    assert _is_collection_missing(ValueError("Collection X not found")) is True
    assert _is_collection_missing(ValueError("something else")) is False
    assert _is_collection_missing(RuntimeError("not found")) is False
    # Remote server mode: 404 UnexpectedResponse.
    assert (
        _is_collection_missing(
            UnexpectedResponse(status_code=404, reason_phrase="", content=b"", headers={})
        )
        is True
    )
    # A 500 from the remote server is a real error, not a missing collection.
    assert (
        _is_collection_missing(
            UnexpectedResponse(status_code=500, reason_phrase="", content=b"", headers={})
        )
        is False
    )


# ─── invalidate_bm25_zh_idf_cache ─────────────────────────────────────────
# In server mode (long-lived API process), a reindex triggered from a
# separate CLI rewrites ``bm25_zh_idf.json`` on disk, but the API
# process keeps the old IDF table in ``_BM25_ZH_IDF_CACHE`` — recall
# drifts out of sync with the freshly rebuilt collection. The public
# ``invalidate_bm25_zh_idf_cache`` lets ``service`` drop the cache after
# reindex / parse so the next query re-reads the on-disk file.


def test_invalidate_bm25_zh_idf_cache_clears_cache() -> None:
    """Calling ``invalidate_bm25_zh_idf_cache`` resets the module cache to None."""
    from mm_asset_rag.backends import qdrant_backend

    # Seed the cache with a sentinel (mtime, table) pair; the function drops it.
    qdrant_indexing._BM25_ZH_IDF_CACHE = (1234567890, {"sentinel": 1.0})
    qdrant_backend.invalidate_bm25_zh_idf_cache()
    assert qdrant_indexing._BM25_ZH_IDF_CACHE is None


def test_invalidate_bm25_zh_idf_cache_idempotent_on_none() -> None:
    """Invalidating when the cache is already None is a no-op."""
    from mm_asset_rag.backends import qdrant_backend

    qdrant_indexing._BM25_ZH_IDF_CACHE = None
    qdrant_backend.invalidate_bm25_zh_idf_cache()
    assert qdrant_indexing._BM25_ZH_IDF_CACHE is None


def test_load_bm25_zh_idf_rereads_when_file_mtime_changes(tmp_path, monkeypatch) -> None:
    """The cache is keyed by the IDF file's mtime, so a rewrite (even from
    a *separate* CLI process) is picked up on the next load without an
    in-process ``invalidate`` call. This is the cross-process case an
    in-process flag alone can't reach: API server caches v1, CLI reindex
    writes v2 with a new mtime, API's next ``_load_bm25_zh_idf`` returns v2."""
    import os
    import time

    idf_path = tmp_path / "bm25_zh_idf.json"
    monkeypatch.setattr(qdrant_indexing, "get_indexes_dir", lambda: tmp_path)

    # v1 on disk.
    idf_path.write_text('{"v1": 1.0}', encoding="utf-8")
    qdrant_indexing._BM25_ZH_IDF_CACHE = None
    assert qdrant_indexing._load_bm25_zh_idf() == {"v1": 1.0}

    # Ensure the next write gets a distinct mtime_ns (same-ns rewrite on a
    # fast disk would otherwise mask the change). Bump mtime explicitly.
    idf_path.write_text('{"v2": 2.0}', encoding="utf-8")
    t = time.time() + 5
    os.utime(idf_path, (t, t))
    # No invalidate() call — the mtime change alone must force a re-read.
    assert qdrant_indexing._load_bm25_zh_idf() == {"v2": 2.0}


def test_load_bm25_zh_idf_returns_none_when_file_missing(tmp_path, monkeypatch) -> None:
    """A missing IDF file yields None and drops any stale cache rather than
    returning a stale table."""

    monkeypatch.setattr(qdrant_indexing, "get_indexes_dir", lambda: tmp_path)
    qdrant_indexing._BM25_ZH_IDF_CACHE = (999, {"stale": 1.0})
    assert qdrant_indexing._load_bm25_zh_idf() is None
    assert qdrant_indexing._BM25_ZH_IDF_CACHE is None


# ─── get_qdrant_client local→remote switch closes local client ───────────
# When the deployer flips ``QDRANT_URL`` on at runtime (e.g. moves from
# local-file mode to a Qdrant server), the previously cached local
# client must be closed so its ``.lock`` and underlying storage fd are
# released. Otherwise the lock stays held for the rest of the process
# even though we no longer use that client.


def test_get_qdrant_client_closes_local_client_when_switching_to_remote(
    monkeypatch, tmp_path
) -> None:
    """Switching to remote mode closes the cached local client."""
    from mm_asset_rag.backends import qdrant_backend

    # Build a fake local client whose close() is observable.
    closed_calls: list[int] = []

    class _FakeLocalClient:
        def __init__(self) -> None:
            self._closed = False

        def close(self) -> None:
            closed_calls.append(1)
            self._closed = True

    # Pretend we already cached a local client (simulating a prior
    # local-mode call) — populate the module globals directly.
    monkeypatch.setattr(
        "mm_asset_rag.backends.qdrant.client.get_indexes_dir",
        lambda: tmp_path / "indexes",
    )
    qdrant_backend.reset_qdrant_client_cache()
    fake_local = _FakeLocalClient()
    qdrant_client._QDRANT_CLIENT = fake_local
    qdrant_client._QDRANT_CLIENT_KEY = str(tmp_path / "indexes" / "qdrant")

    # Configure remote mode.
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()

    # Stub QdrantClient so we don't actually open a remote connection.
    constructed: list[str] = []

    class _FakeRemoteClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs.get("url", ""))

        def close(self) -> None:  # pragma: no cover — never called here
            pass

    monkeypatch.setattr(qdrant_client, "QdrantClient", _FakeRemoteClient)

    try:
        client = qdrant_backend.get_qdrant_client()
    finally:
        qdrant_backend.reset_qdrant_client_cache()
        # Restore env cache for other tests.
        monkeypatch.delenv("QDRANT_URL", raising=False)
        get_settings.cache_clear()

    # The cached local client was closed exactly once.
    assert closed_calls == [1]
    # A new remote client was constructed.
    assert constructed == ["http://example:6333"]
    # The module-level cache is cleared (local client no longer held).
    assert qdrant_client._QDRANT_CLIENT is None
    assert qdrant_client._QDRANT_CLIENT_KEY is None
    # Returned client is the freshly constructed remote one.
    assert isinstance(client, _FakeRemoteClient)


def test_get_qdrant_client_remote_mode_no_local_cache_to_close(monkeypatch) -> None:
    """Switching to remote when no local client is cached is a no-op on close."""
    from mm_asset_rag.backends import qdrant_backend

    monkeypatch.setattr(
        "mm_asset_rag.backends.qdrant.client.get_indexes_dir",
        lambda: None,  # not used in remote branch
    )
    qdrant_backend.reset_qdrant_client_cache()
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()

    constructed: list[str] = []

    class _FakeRemoteClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs.get("url", ""))

        def close(self) -> None:  # pragma: no cover
            pass

    monkeypatch.setattr(qdrant_client, "QdrantClient", _FakeRemoteClient)
    try:
        qdrant_backend.get_qdrant_client()
    finally:
        qdrant_backend.reset_qdrant_client_cache()
        monkeypatch.delenv("QDRANT_URL", raising=False)
        get_settings.cache_clear()

    assert constructed == ["http://example:6333"]
    assert qdrant_client._QDRANT_CLIENT is None
