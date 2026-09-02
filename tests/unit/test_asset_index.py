"""Tests for breaking-v2 document-version persistence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Event, Process
from pathlib import Path

import pytest

from mm_asset_rag.asset_index import (
    DocumentVersionRecord,
    find_version,
    load_records,
    next_version_number,
    upsert_record,
    upsert_records,
)
from mm_asset_rag.document_store import _advisory_lock
from mm_asset_rag.knowledge_models import AccessPolicy, Asset, Document, DocumentVersion, Source


def _record(*, content_hash: str = "a" * 64, version_number: int = 1) -> DocumentVersionRecord:
    source = Source(source_id="upload:handbook")
    document = Document(
        document_id="handbook",
        title="Handbook",
        source=source,
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )
    return DocumentVersionRecord(
        document=document,
        version=DocumentVersion.create(document, content_hash, version_number=version_number),
        asset=Asset(
            content_hash=content_hash, source_type="pdf", relative_path="pdfs/handbook.pdf"
        ),
    )


def test_document_version_record_round_trips_without_asset_identity(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_record(_record(), path=path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "asset_id" not in raw
    assert raw["document"]["document_id"] == "handbook"
    assert raw["access_policy"] == {
        "collection": "team",
        "allowed_principals": ["alice"],
        "metadata": {},
    }
    assert load_records(path) == [_record()]


def test_legacy_asset_only_rows_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text(
        json.dumps({"asset_id": "legacy", "sha256": "a" * 64, "relative_path": "pdfs/a.pdf"})
        + "\n",
        encoding="utf-8",
    )

    assert load_records(path) == []


def test_tampered_version_identity_rows_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_record(_record(), path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"]["version_id"] = "handbook@1-tampered"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert load_records(path) == []


def test_version_number_advances_for_new_content_and_is_idempotent_for_same_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.jsonl"
    upsert_record(_record(content_hash="a" * 64, version_number=1), path=path)

    assert find_version("handbook", "a" * 64, path=path).version.version_number == 1
    assert next_version_number("handbook", path=path) == 2


def test_repeated_upsert_is_idempotent_and_allocates_versions_atomically(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"

    first = upsert_record(_record(content_hash="a" * 64), path=path)
    repeated = upsert_record(_record(content_hash="a" * 64), path=path)
    second = upsert_record(_record(content_hash="b" * 64), path=path)

    assert first.version.version_number == 1
    assert repeated == first
    assert second.version.version_number == 2
    assert [record.version.content_hash for record in load_records(path)] == ["a" * 64, "b" * 64]


def test_batch_upsert_rolls_back_all_rows_when_one_write_fails(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "index.jsonl"
    first = _record(content_hash="a" * 64)
    second = _record(content_hash="b" * 64)
    original = upsert_record
    calls = 0

    def fail_after_first(record, path=None):
        nonlocal calls
        calls += 1
        result = original(record, path=path)
        if calls == 2:
            raise OSError("disk full")
        return result

    monkeypatch.setattr("mm_asset_rag.asset_index.upsert_record", fail_after_first)
    with pytest.raises(OSError, match="disk full"):
        upsert_records([first, second], path=path)

    assert not path.exists()
    assert load_records(path) == []


def test_concurrent_upserts_allocate_unique_versions_and_dedupe_content(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    records = [_record(content_hash=hash_value * 64) for hash_value in ("a", "b", "a", "c")]

    with ThreadPoolExecutor(max_workers=4) as executor:
        persisted = list(executor.map(lambda item: upsert_record(item, path=path), records))

    loaded = load_records(path)
    assert {item.version.content_hash for item in loaded} == {"a" * 64, "b" * 64, "c" * 64}
    assert sorted(item.version.version_number for item in loaded) == [1, 2, 3]
    assert sum(item.version.content_hash == "a" * 64 for item in persisted) == 2


def _upsert_in_child(path: str, ready: Event, finished: Event) -> None:
    ready.set()
    upsert_record(_record(content_hash="d" * 64), path=Path(path))
    finished.set()


def test_upsert_waits_for_cross_process_index_lock(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    ready = Event()
    finished = Event()
    process = Process(target=_upsert_in_child, args=(str(path), ready, finished))
    with _advisory_lock(path, exclusive=True):
        process.start()
        assert ready.wait(timeout=2)
        assert not finished.wait(timeout=0.2)
    assert finished.wait(timeout=2)
    process.join(timeout=2)
    assert process.exitcode == 0
    assert len(load_records(path)) == 1
