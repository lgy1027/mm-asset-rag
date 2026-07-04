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

    def fake_embed(self, content):
        # probe call on documents[0] — also surfaces the prefix
        seen_texts.append(str(content))
        return [0.1, 0.2, 0.3, 0.4]

    def fake_embed_batch(self, texts):
        seen_texts.extend(texts)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    import mm_asset_rag.embedders.text_embedder as te

    with (
        patch.object(te.TextEmbedder, "embed", fake_embed),
        patch.object(te.TextEmbedder, "embed_batch", fake_embed_batch),
    ):
        build_qdrant_text_index(force_recreate=True)

    # Embedding input = context + body (probe or batch).
    assert any("这是关于DDPM去噪扩散的前缀" in t and "正文内容" in t for t in seen_texts)

    # Payload text = raw body only (evidence/answer stays clean). Check both points.
    upsert_call = fake_qdrant_client.upsert.call_args
    points = upsert_call.kwargs["points"]
    for p in points:
        assert "这是关于DDPM去噪扩散的前缀" not in p.payload["text"]
        assert p.payload["text"].startswith("正文内容")
        assert p.payload["context"] == "这是关于DDPM去噪扩散的前缀"
