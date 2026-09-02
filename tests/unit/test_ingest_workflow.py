"""Focused v2 persistence tests for the parse workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.asset_index import DocumentVersionRecord, upsert_record
from mm_asset_rag.assets import Asset
from mm_asset_rag.document_store import read_documents
from mm_asset_rag.ingest_workflow import IngestWorkflow
from mm_asset_rag.knowledge_models import (
    AccessPolicy,
    Document,
    DocumentVersion,
    Source,
)
from mm_asset_rag.knowledge_models import (
    Asset as PersistedAsset,
)
from mm_asset_rag.service import IngestService, ParseOptions, TaskRecord


def test_parse_rejects_asset_without_persisted_document_version(tmp_home: Path) -> None:
    asset = Asset(
        asset_id="orphan",
        title="Orphan",
        source_type="image",
        relative_path="images/orphan.png",
        asset_dir=tmp_home / "assets",
    )
    record = TaskRecord(task_id="parse-orphan", kind="parse", status="running", total=1)

    with pytest.raises(ValueError, match="no persisted document version"):
        IngestWorkflow().parse(IngestService(), record, ParseOptions(assets=[asset]))

    assert record.version_statuses == {}


def test_parse_converts_transient_parser_output_to_v2_chunk(tmp_home: Path, monkeypatch) -> None:
    from mm_asset_rag.parsers import image_parser
    from mm_asset_rag.settings import get_settings

    settings = get_settings()
    settings.contextual_enabled = False
    image_path = tmp_home / "assets" / "images" / "scene.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png bytes")
    asset = Asset(
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
        DocumentVersionRecord(
            document=document,
            version=DocumentVersion.create(document, content_hash),
            asset=PersistedAsset(
                content_hash=content_hash,
                source_type="image",
                relative_path=asset.relative_path,
            ),
        )
    )

    monkeypatch.setattr(image_parser, "run_ocr", lambda path: [{"text": "scene body"}])
    service = IngestService(settings=settings)
    record = TaskRecord(task_id="parse-v2", kind="parse", status="running", total=1)
    IngestWorkflow().parse(service, record, ParseOptions(assets=[asset], enable_ocr=True))

    [chunk] = read_documents()
    assert "scene body" in chunk.text
    assert chunk.document_id == "scene"
    version_id = f"scene@1-{'d' * 64}"
    assert chunk.document_version.version_id == version_id
    raw = json.loads((tmp_home / "documents.jsonl").read_text(encoding="utf-8"))
    assert "document_version" in raw
    assert "asset_id" not in raw["metadata"]
    assert "asset_id" not in raw
    assert raw["chunk_id"] == f"{version_id}:0"
    assert record.version_statuses == {version_id: "ok"}
