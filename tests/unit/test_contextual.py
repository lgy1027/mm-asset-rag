"""Tests for Contextual Retrieval (``mm_asset_rag.contextual``).

Covers the three contracts the index path depends on:
1. ``generate_chunk_context`` / ``generate_doc_summary`` build the right
   prompt and degrade to ``""`` on LLM failure (never raise).
2. ``enrich_docs_with_context`` writes ``metadata["context"]`` and caches
   to ``parsed/<id>/context.jsonl`` so a second call reuses it (no second
   LLM round-trip).
3. ``build_qdrant_text_index`` prepends the context to the embedding input
   while keeping the payload ``text`` raw.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mm_asset_rag.contextual import (
    enrich_docs_with_context,
    generate_chunk_context,
    generate_doc_summary,
)
from mm_asset_rag.knowledge_models import AccessPolicy, Chunk, Document, Source
from mm_asset_rag.knowledge_models import Asset as PersistedAsset
from mm_asset_rag.llm_transport import LlmRateLimiter
from mm_asset_rag.schema import ParsedChunk


def _doc(text: str, *, chunk_index: int | None = 0, section: str = "") -> ParsedChunk:
    return ParsedChunk(
        text=text,
        metadata={"asset_id": "a1", "chunk_index": chunk_index, "section": section},
    )


def _stored_chunks(texts: list[str], *, context: str) -> list[Chunk]:
    document = Document(
        document_id="context-doc",
        title="Context document",
        source=Source(source_id="upload:context-doc"),
        access_policy=AccessPolicy(collection="tests", allowed_principals=()),
    )
    content_hash = "c" * 64
    asset = PersistedAsset(
        content_hash=content_hash,
        source_type="pdf",
        relative_path="pdfs/context-doc.pdf",
    )
    return [
        Chunk.create(
            document=document,
            asset=asset,
            ordinal=index,
            text=text,
            source=document.source,
            access_policy=document.access_policy,
            metadata={"context": context},
        )
        for index, text in enumerate(texts)
    ]


@pytest.fixture(autouse=True)
def _disable_chat_pacing(monkeypatch):
    monkeypatch.setattr(
        "mm_asset_rag.llm_transport.get_llm_rate_limiter", lambda: LlmRateLimiter(0)
    )


def test_generate_doc_summary_builds_prompt_and_strips_think(tmp_home, monkeypatch):
    """Summary call posts the full text and returns the cleaned answer."""
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "<think>hidden</think>文档摘要内容"}}]}

    def fake_post(url, **kwargs):
        captured["payload"] = kwargs["json"]
        return FakeResp()

    with (
        patch(
            "mm_asset_rag.contextual._llm_credentials",
            return_value=("https://example.com/v1", "sk-test", "test-m3"),
        ),
        patch("mm_asset_rag.contextual.requests.post", side_effect=fake_post),
    ):
        out = generate_doc_summary("正文内容" * 10, asset_title="标题")

    assert out == "文档摘要内容"
    assert captured["payload"]["model"] == "test-m3"
    assert "标题" in captured["payload"]["messages"][1]["content"]
    assert "正文内容" in captured["payload"]["messages"][1]["content"]


def test_generate_chunk_context_degrades_on_failure(tmp_home, monkeypatch):
    """Any LLM failure (network / missing creds) → empty string, never raise."""
    # No credentials → immediate "" without a request. Patch the credential
    # resolver directly because Settings loads credentials from the on-disk .env
    # (the real home .env has live MiniMax creds), which would bypass a pure
    # env-var monkeypatch.
    with patch("mm_asset_rag.contextual._llm_credentials", return_value=(None, None, None)):
        assert generate_chunk_context("chunk", "summary") == ""

    # Credentials set but request raises → still "".
    with (
        patch(
            "mm_asset_rag.contextual._llm_credentials",
            return_value=("https://example.com/v1", "sk-test", "test-m3"),
        ),
        patch("mm_asset_rag.contextual.requests.post", side_effect=Exception("boom")),
    ):
        assert generate_chunk_context("chunk", "summary") == ""


def test_enrich_docs_writes_context_and_caches(tmp_home, monkeypatch):
    """enrich attaches context to metadata and persists a reusable cache."""
    call_count = {"n": 0}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": f"ctx-{call_count['n']}"}}]}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        return FakeResp()

    docs = [_doc("片段一", chunk_index=0), _doc("片段二", chunk_index=1)]
    cache_path = tmp_home / "parsed" / "a1" / "context.jsonl"

    with (
        patch(
            "mm_asset_rag.contextual._llm_credentials",
            return_value=("https://example.com/v1", "sk-test", "test-m3"),
        ),
        patch("mm_asset_rag.contextual.requests.post", side_effect=fake_post),
    ):
        enrich_docs_with_context(docs, asset_title="标题", cache_path=cache_path)

    # 1 doc-summary call + 2 chunk calls = 3 LLM round-trips.
    assert call_count["n"] == 3
    assert all(d.metadata.get("context") for d in docs)
    assert cache_path.exists()

    # Second call reuses the cache: no new LLM calls, context preserved.
    docs2 = [_doc("片段一", chunk_index=0), _doc("片段二", chunk_index=1)]
    with (
        patch(
            "mm_asset_rag.contextual._llm_credentials",
            return_value=("https://example.com/v1", "sk-test", "test-m3"),
        ),
        patch("mm_asset_rag.contextual.requests.post", side_effect=fake_post),
    ):
        enrich_docs_with_context(docs2, asset_title="标题", cache_path=cache_path)
    assert call_count["n"] == 3  # unchanged
    assert docs2[0].metadata.get("context") == docs[0].metadata.get("context")


def test_enrich_skips_when_llm_unconfigured(tmp_home, monkeypatch):
    """No credentials → enrich is a no-op; docs keep no context key."""
    docs = [_doc("片段", chunk_index=0)]
    with patch("mm_asset_rag.contextual._llm_credentials", return_value=(None, None, None)):
        enrich_docs_with_context(docs, asset_title="t", cache_path=tmp_home / "c.jsonl")
    assert "context" not in docs[0].metadata or not docs[0].metadata["context"]


def test_build_qdrant_text_index_prepends_context(tmp_home, fake_qdrant_client, fixed_vector):
    """The embedding input gets the context prefix; the payload text stays raw."""
    from mm_asset_rag.backends.qdrant.indexing import build_text_index
    from mm_asset_rag.document_store import write_documents
    from mm_asset_rag.registry import embedders, register_embedder

    docs = _stored_chunks(["正文内容一", "正文内容二"], context="这是关于DDPM去噪扩散的前缀")
    write_documents(docs)

    seen_texts: list[str] = []

    # Register a stub text embedder on the ``("text", "default")`` slot so
    # ``build_qdrant_text_index``'s ``get_default_text_embedder()`` finds it
    # without needing real credentials (CI has none — ``build_default`` would
    # raise ``EmbeddingConfigError`` before any embed call). The stub records
    # the embedding input (context + body) so we can assert the prefix is
    # applied, and returns a fixed vector so the Qdrant upsert is well-formed.
    class _RecordingStub:
        modality = "text"

        @property
        def name(self) -> str:
            return "default"

        def dim(self) -> int:
            return 4

        def embed(self, content) -> list[float]:
            seen_texts.append(str(content))
            return [0.1, 0.2, 0.3, 0.4]

        def embed_batch(self, texts) -> list[list[float]]:
            seen_texts.extend(str(t) for t in texts)
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    register_embedder(_RecordingStub(), replace=True)
    try:
        build_text_index(force_recreate=True)
    finally:
        embedders._items.pop(("text", "default"), None)

    # Embedding input = context + body (probe or batch).
    assert any("这是关于DDPM去噪扩散的前缀" in t and "正文内容" in t for t in seen_texts)

    # Payload text = raw body only (evidence/answer stays clean). Check both points.
    upsert_call = fake_qdrant_client.upsert.call_args
    points = upsert_call.kwargs["points"]
    for p in points:
        assert "这是关于DDPM去噪扩散的前缀" not in p.payload["text"]
        assert p.payload["text"].startswith("正文内容")
        assert p.payload["context"] == "这是关于DDPM去噪扩散的前缀"


def test_build_qdrant_text_index_probe_not_reused_when_doc0_has_context(
    tmp_home, fake_qdrant_client, fixed_vector
):
    """When ``documents[0]`` carries a context preamble, the probe embedding
    (computed on the bare text) must NOT be reused into ``dense_vectors[0]``.

    Regression: the probe reuses ``first_vector = embed(documents[0].text)``
    as ``dense_vectors[0]`` when offset==0 and doc 0 is in ``to_do``. With
    context, the dense input should be ``"{ctx}\\n\\n{text}"`` but the probe
    embedded the bare ``{text}`` — reusing it gives the first chunk a
    context-less dense vector whose sparse sibling carries the context.
    The fix adds a ``not batch[0].metadata.get("context")`` guard so the
    first chunk goes through ``embed_batch`` with its context prefix."""
    from mm_asset_rag.backends.qdrant.indexing import build_text_index
    from mm_asset_rag.document_store import write_documents
    from mm_asset_rag.registry import embedders, register_embedder

    # Single doc with a context preamble so offset==0, 0 in to_do, and
    # the reuse path is the one the guard must block.
    docs = _stored_chunks(["正文内容"], context="CTX-前缀")
    write_documents(docs)

    seen_texts: list[str] = []

    class _RecordingStub:
        modality = "text"

        @property
        def name(self) -> str:
            return "default"

        def dim(self) -> int:
            return 4

        def embed(self, content) -> list[float]:
            seen_texts.append(str(content))
            return [0.1, 0.2, 0.3, 0.4]

        def embed_batch(self, texts) -> list[list[float]]:
            seen_texts.extend(str(t) for t in texts)
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    register_embedder(_RecordingStub(), replace=True)
    try:
        build_text_index(force_recreate=True)
    finally:
        embedders._items.pop(("text", "default"), None)

    # Every batch embedding input must carry the context prefix. The probe
    # (embed(documents[0].text)) is bare-text by design and is excluded: when
    # doc 0 has context it is NOT reused into dense_vectors[0], so the
    # context-prefixed input only reaches the batch call.
    assert seen_texts, "no embedding calls captured"
    batch_inputs = seen_texts[1:]  # skip the bare probe
    assert batch_inputs, "no batch embedding call captured"
    for t in batch_inputs:
        assert "CTX-前缀" in t, f"context prefix missing from input: {t!r}"


def test_enrich_noop_without_credentials_writes_no_cache(tmp_home, monkeypatch) -> None:
    """When the LLM is unconfigured, enrich is a full no-op: no LLM call, no
    exception, and no cache file written (keeps the parse dir clean).

    Pins the credentials at ``(None, None, None)`` via the same
    ``_llm_credentials`` seam the other no-creds test uses, so the result
    doesn't depend on credentials in the host environment.
    """
    docs = [_doc("片段一", chunk_index=0), _doc("片段二", chunk_index=1)]
    cache_path = tmp_home / "parsed" / "a1" / "context.jsonl"

    with patch("mm_asset_rag.contextual._llm_credentials", return_value=(None, None, None)):
        enrich_docs_with_context(docs, asset_title="t", cache_path=cache_path)

    assert not cache_path.exists()
    for d in docs:
        assert not d.metadata.get("context")


def test_enrich_degrades_silently_on_request_failure(tmp_home, monkeypatch) -> None:
    """A raised exception from requests.post degrades to empty context — never
    propagates — and writes no cache when nothing was produced."""
    docs = [_doc("片段", chunk_index=0)]
    cache_path = tmp_home / "parsed" / "a2" / "context.jsonl"

    with (
        patch(
            "mm_asset_rag.contextual._llm_credentials",
            return_value=("https://example.com/v1", "sk-test", "test-m3"),
        ),
        patch("mm_asset_rag.contextual.requests.post", side_effect=Exception("boom")),
    ):
        enrich_docs_with_context(docs, asset_title="t", cache_path=cache_path)

    assert not cache_path.exists()
    assert not docs[0].metadata.get("context")
