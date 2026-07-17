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

from mm_asset_rag.contextual import (
    enrich_docs_with_context,
    generate_chunk_context,
    generate_doc_summary,
)
from mm_asset_rag.schema import ParsedDocument


def _doc(text: str, *, chunk_index: int | None = 0, section: str = "") -> ParsedDocument:
    return ParsedDocument(
        text=text,
        metadata={"asset_id": "a1", "chunk_index": chunk_index, "section": section},
    )


def test_generate_doc_summary_builds_prompt_and_strips_think(tmp_home, monkeypatch):
    """Summary call posts the full text and returns the cleaned answer."""
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "<think>hidden</think>文档摘要内容"}}]}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
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
    # resolver directly because Settings loads OPENAI_* from the on-disk .env
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

    def fake_post(url, headers, json, timeout):
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
    from mm_asset_rag.backends.qdrant_backend import build_qdrant_text_index
    from mm_asset_rag.document_store import write_documents
    from mm_asset_rag.registry import embedders, register_embedder

    docs = [
        ParsedDocument(
            text="正文内容一",
            metadata={
                "asset_id": "a1",
                "chunk_index": 0,
                "source_type": "pdf",
                "context": "这是关于DDPM去噪扩散的前缀",
            },
        ),
        ParsedDocument(
            text="正文内容二",
            metadata={
                "asset_id": "a1",
                "chunk_index": 1,
                "source_type": "pdf",
                "context": "这是关于DDPM去噪扩散的前缀",
            },
        ),
    ]
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
        build_qdrant_text_index(force_recreate=True)
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


def test_contextual_enabled_defaults_true(tmp_home) -> None:
    """``Settings.contextual_enabled`` defaults to True so contextual runs
    without explicit opt-in (the latency/precision trade-off favors precision).
    Set ``CONTEXTUAL_ENABLED=false`` to opt out."""
    from mm_asset_rag.settings import get_settings

    assert get_settings().contextual_enabled is True

    # Env var can opt out (backward-compatible escape hatch).
    import os

    os.environ["CONTEXTUAL_ENABLED"] = "false"
    get_settings.cache_clear()
    try:
        assert get_settings().contextual_enabled is False
    finally:
        del os.environ["CONTEXTUAL_ENABLED"]
    get_settings.cache_clear()


def test_enrich_noop_without_credentials_writes_no_cache(tmp_home, monkeypatch) -> None:
    """When OPENAI_* is unconfigured, enrich is a full no-op: no LLM call, no
    exception, and no cache file written (keeps the parse dir clean)."""
    docs = [_doc("片段一", chunk_index=0), _doc("片段二", chunk_index=1)]
    cache_path = tmp_home / "parsed" / "a1" / "context.jsonl"

    # No credentials anywhere — neither env nor settings.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    # Should not raise and should not create a cache file.
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
