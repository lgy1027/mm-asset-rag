"""Focused v2 persistence tests for the parse workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.core.knowledge_models import (
    AccessPolicy,
    Document,
    Source,
)
from mm_asset_rag.core.knowledge_models import (
    Asset as PersistedAsset,
)
from mm_asset_rag.ingest.asset_index import DocumentRecord, upsert_record
from mm_asset_rag.ingest.assets import IngestAsset
from mm_asset_rag.ingest.document_store import read_documents
from mm_asset_rag.ingest.ingest_workflow import IngestWorkflow
from mm_asset_rag.service import IngestService, ParseOptions, TaskRecord


def test_parse_rejects_asset_without_persisted_document(tmp_home: Path) -> None:
    asset = IngestAsset(
        asset_id="orphan",
        title="Orphan",
        source_type="image",
        relative_path="images/orphan.png",
        asset_dir=tmp_home / "assets",
    )
    record = TaskRecord(task_id="parse-orphan", kind="parse", status="running", total=1)

    with pytest.raises(ValueError, match="no persisted document"):
        IngestWorkflow().parse(IngestService(), record, ParseOptions(assets=[asset]))

    assert record.document_statuses == {}


def test_parse_converts_transient_parser_output_to_v2_chunk(tmp_home: Path, monkeypatch) -> None:
    from mm_asset_rag.ingest import contextual
    from mm_asset_rag.parsers import image_parser

    image_path = tmp_home / "assets" / "images" / "scene.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png bytes")
    asset = IngestAsset(
        asset_id="scene.png",
        title="Scene",
        source_type="image",
        relative_path="images/scene.png",
        asset_dir=tmp_home / "assets",
    )
    source = Source(source_id="upload:scene")
    document = Document(
        document_id="scene",
        title="Scene",
        source=source,
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )
    content_hash = "d" * 64
    upsert_record(
        DocumentRecord(
            document=document,
            asset=PersistedAsset(
                content_hash=content_hash,
                source_type="image",
                relative_path=asset.relative_path,
            ),
        )
    )

    monkeypatch.setattr(image_parser, "run_ocr", lambda path: [{"text": "scene body"}])
    contextual_calls: list[str] = []
    monkeypatch.setattr(
        contextual,
        "enrich_docs_with_context",
        lambda *args, **kwargs: contextual_calls.append("called"),
    )
    service = IngestService()
    record = TaskRecord(task_id="parse-v2", kind="parse", status="running", total=1)
    IngestWorkflow().parse(service, record, ParseOptions(assets=[asset], enable_ocr=True))

    [chunk] = read_documents()
    assert "scene body" in chunk.text
    assert chunk.document_id == "scene"
    assert chunk.document_id == "scene"
    raw = json.loads((tmp_home / "documents.jsonl").read_text(encoding="utf-8"))
    assert "document" in raw
    assert "asset_id" not in raw["metadata"]
    assert "asset_id" not in raw
    assert raw["chunk_id"] == "scene:0"
    assert record.document_statuses == {"scene": "ok"}
    assert contextual_calls == []


def test_parse_video_honours_settings_enable_vlm(tmp_home: Path, monkeypatch) -> None:
    """ENABLE_VLM=true must reach the video parser for API-style parses
    (ParseOptions carries no per-task override)."""
    from mm_asset_rag.core.settings import get_settings
    from mm_asset_rag.parsers import video_parser

    video_path = tmp_home / "assets" / "video" / "talk.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"mp4 bytes")
    asset = IngestAsset(
        asset_id="talk",
        title="Talk",
        source_type="video",
        relative_path="video/talk.mp4",
        asset_dir=tmp_home / "assets",
    )
    document = Document(
        document_id="talk",
        title="Talk",
        source=Source(source_id="upload:talk"),
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )
    upsert_record(
        DocumentRecord(
            document=document,
            asset=PersistedAsset(
                content_hash="e" * 64,
                source_type="video",
                relative_path=asset.relative_path,
            ),
        )
    )

    seen: list[bool] = []

    def fake_parse_video(asset, *, enable_vlm=False, chunk_seconds=None):
        seen.append(enable_vlm)
        return [
            video_parser.ParsedChunk(text="语音内容", metadata={"document_id": "talk"})
        ]

    # The registered parser binds ``parse_video`` in the parsers package
    # namespace, so patch the name where it is looked up.
    import mm_asset_rag.parsers as parsers_pkg

    monkeypatch.setattr(parsers_pkg, "parse_video", fake_parse_video)
    get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_VLM", "true")
    try:
        service = IngestService()
        record = TaskRecord(task_id="parse-video", kind="parse", status="running", total=1)
        IngestWorkflow().parse(service, record, ParseOptions(assets=[asset]))
    finally:
        get_settings.cache_clear()

    assert seen == [True]
    assert record.document_statuses == {"talk": "ok"}
