"""Tests for current-document persistence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mm_asset_rag.core.knowledge_models import AccessPolicy, Asset, Document, Source
from mm_asset_rag.ingest.asset_index import (
    DocumentRecord,
    find_document,
    load_records,
    upsert_record,
)


def _record(*, content_hash: str = "a" * 64, title: str = "Handbook") -> DocumentRecord:
    source = Source(source_id="upload:handbook")
    return DocumentRecord(
        document=Document(
            document_id="handbook",
            title=title,
            source=source,
            access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
        ),
        asset=Asset(
            content_hash=content_hash, source_type="pdf", relative_path="pdfs/handbook.pdf"
        ),
    )


def test_document_record_round_trips_without_version_identity(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    expected = _record()
    assert upsert_record(expected, path=path) == expected

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "version" not in raw
    assert raw["document"]["document_id"] == "handbook"
    assert load_records(path) == [expected]


def test_legacy_or_versioned_rows_are_not_read(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text(
        json.dumps({"document": {}, "version": {}, "asset": {}}) + "\n", encoding="utf-8"
    )
    assert load_records(path) == []


def test_upsert_replaces_the_current_document(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_record(_record(content_hash="a" * 64), path=path)
    replacement = upsert_record(_record(content_hash="b" * 64, title="New handbook"), path=path)

    assert load_records(path) == [replacement]
    assert find_document("handbook", path=path) == replacement
    assert replacement.asset.content_hash == "b" * 64


def test_concurrent_upserts_leave_one_current_document(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    records = [_record(content_hash=value * 64) for value in ("a", "b", "c")]
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda item: upsert_record(item, path=path), records))

    assert len(load_records(path)) == 1
