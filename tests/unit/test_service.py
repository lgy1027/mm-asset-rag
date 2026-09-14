"""Tests for ``mm_asset_rag.service`` retry and history behaviour."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mm_asset_rag.asset_index import DocumentVersionRecord, load_records, upsert_record
from mm_asset_rag.assets import IngestAsset
from mm_asset_rag.document_store import append_documents, read_documents
from mm_asset_rag.knowledge_models import AccessPolicy, Chunk, Document, DocumentVersion, Source
from mm_asset_rag.knowledge_models import Asset as PersistedAsset
from mm_asset_rag.paths import physical_cache_id
from mm_asset_rag.service import IngestService, ParseOptions, TaskRecord


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_asset(tmp_home: Path, name: str = "fish.png") -> IngestAsset:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    images_dir = tmp_home / "assets" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    file_path = images_dir / name
    Image.new("RGB", (8, 8), color=(120, 120, 0)).save(file_path)
    return IngestAsset(
        asset_id=name,
        title=name,
        source_type="image",
        relative_path=f"images/{name}",
        source_url="",
        tags=[],
        asset_dir=tmp_home / "assets",
    )


def _persist_version(
    tmp_home: Path,
    *,
    document_id: str,
    version_hash: str,
    name: str,
    with_caches: bool = True,
) -> DocumentVersionRecord:
    asset = _make_asset(tmp_home, name)
    document = Document(
        document_id=document_id,
        title=document_id,
        source=Source(source_id=f"upload:{document_id}"),
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )
    record = upsert_record(
        DocumentVersionRecord(
            document=document,
            version=DocumentVersion.create(document, version_hash),
            asset=PersistedAsset(
                content_hash=version_hash,
                source_type=asset.source_type,
                relative_path=asset.relative_path,
            ),
        )
    )
    append_documents(
        [
            Chunk.create(
                document_version=record.version,
                asset=record.asset,
                ordinal=0,
                text=f"{document_id} {record.version.version_id}",
                source=record.document.source,
                access_policy=record.document.access_policy,
            )
        ]
    )
    if with_caches:
        cache_key = physical_cache_id(record.asset.relative_path)
        parsed_dir = tmp_home / "parsed" / cache_key
        parsed_dir.mkdir(parents=True, exist_ok=True)
        (parsed_dir / "raw.jsonl").write_text("{}", encoding="utf-8")
        captions_dir = tmp_home / "captions"
        captions_dir.mkdir(parents=True, exist_ok=True)
        (captions_dir / f"{cache_key}.jsonl").write_text("{}\n", encoding="utf-8")
    return record


def _transient_asset(record: DocumentVersionRecord, tmp_home: Path) -> IngestAsset:
    return IngestAsset(
        asset_id=physical_cache_id(record.asset.relative_path),
        title=record.document.title,
        source_type=record.asset.source_type,
        relative_path=record.asset.relative_path,
        asset_dir=tmp_home / "assets",
    )


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_load_history_reads_through_task_store(tmp_home: Path) -> None:
    store = Mock()
    store.load.return_value = []

    IngestService(task_store=store).load_history()

    store.load.assert_called_once_with()


def test_ingest_service_delegates_worker_to_workflow(tmp_home: Path) -> None:
    asset = _make_asset(tmp_home)
    workflow = Mock()

    record = IngestService(workflow=workflow).ingest_assets([asset], ParseOptions(assets=[asset]))

    _wait_until(lambda: workflow.run.called)
    assert record.task_id


def test_retry_task_resurrects_assets(tmp_home: Path) -> None:
    asset = _make_asset(tmp_home)
    service = IngestService()
    original = TaskRecord(
        task_id="origtask0001",
        kind="ingest",
        status="failed",
        total=1,
        uploaded_files=[asset.relative_path],
        parse_options={"pdf_parser": "auto"},
    )
    service._tasks[original.task_id] = original

    with patch.object(service, "_spawn") as spawn:
        new_rec = service.retry_task(original.task_id)

    spawn.assert_called_once()
    assert new_rec.source == "retry"
    assert new_rec.origin_task_id == original.task_id
    assert new_rec.kind == "ingest"
    assert new_rec.uploaded_files == [asset.relative_path]


def test_retry_task_rejects_non_terminal_status(tmp_home: Path) -> None:
    asset = _make_asset(tmp_home)
    service = IngestService()
    original = TaskRecord(
        task_id="running1",
        kind="ingest",
        status="running",
        total=1,
        uploaded_files=[asset.relative_path],
    )
    service._tasks[original.task_id] = original
    with pytest.raises(ValueError, match="cannot be retried"):
        service.retry_task(original.task_id)


def test_retry_task_rejects_unsafe_paths(tmp_home: Path) -> None:
    service = IngestService()
    original = TaskRecord(
        task_id="unsafe1",
        kind="ingest",
        status="failed",
        total=1,
        uploaded_files=["../escape.png", "/abs.png", "images/missing.png"],
    )
    service._tasks[original.task_id] = original
    with pytest.raises(FileNotFoundError, match="no assets available"):
        service.retry_task(original.task_id)


def test_retry_task_unknown_id(tmp_home: Path) -> None:
    service = IngestService()
    with pytest.raises(KeyError, match="unknown task"):
        service.retry_task("does-not-exist")


def test_parse_options_serialisation_roundtrip() -> None:
    options = ParseOptions(
        assets=[],
        pdf_parser="pymupdf",
        document_parser="docling",
        enable_ocr=True,
        enable_vlm=False,
        contextual=True,
    )
    snap = IngestService._serialise_options(options)
    assert snap == {
        "pdf_parser": "pymupdf",
        "document_parser": "docling",
        "enable_ocr": True,
        "enable_vlm": False,
        "contextual": True,
    }
    restored = IngestService._deserialise_options(snap, assets=[])
    assert restored.pdf_parser == "pymupdf"
    assert restored.document_parser == "docling"
    assert restored.enable_ocr is True
    assert restored.enable_vlm is False
    assert restored.contextual is True


def test_parse_options_serialisation_drops_invalid_values() -> None:
    snap = {"pdf_parser": "bogus", "document_parser": "weird"}
    options = IngestService._deserialise_options(snap, assets=[])
    assert options.pdf_parser == "auto"
    # Invalid document_parser falls back to the default, not the bogus value.
    assert options.document_parser == "markitdown"


# ─── document lifecycle ─────────────────────────────────────────────────────


def test_delete_document_version_preserves_sibling_versions_and_documents(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="1" * 64,
        name="guide-old.png",
    )
    current = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="2" * 64,
        name="guide-current.png",
    )
    other = _persist_version(
        tmp_home,
        document_id="handbook",
        version_hash="3" * 64,
        name="handbook.png",
    )
    qdrant = Mock()
    qdrant.get_collections.return_value = Mock(
        collections=[
            type("_Collection", (), {"name": "multimodal_text_4d"})(),
            type("_Collection", (), {"name": "multimodal_image_4d"})(),
        ]
    )
    monkeypatch.setattr("mm_asset_rag.service.get_qdrant_client", lambda: qdrant)

    report = IngestService().delete_document_version("guide", old.version.version_id)

    assert report.was_known
    assert report.deleted_version_ids == [old.version.version_id]
    assert report.versions_removed == 1
    assert report.chunks_removed == 1
    assert not report.errors
    assert {(row.document.document_id, row.version.version_id) for row in load_records()} == {
        ("guide", current.version.version_id),
        ("handbook", other.version.version_id),
    }
    assert {
        (chunk.document_id, chunk.document_version.version_id) for chunk in read_documents()
    } == {
        ("guide", current.version.version_id),
        ("handbook", other.version.version_id),
    }
    assert not (tmp_home / "assets" / old.asset.relative_path).exists()
    assert not (tmp_home / "parsed" / physical_cache_id(old.asset.relative_path)).exists()
    assert (tmp_home / "assets" / current.asset.relative_path).exists()
    assert (tmp_home / "parsed" / physical_cache_id(current.asset.relative_path)).exists()
    assert (tmp_home / "assets" / other.asset.relative_path).exists()
    assert qdrant.delete.call_count == 2
    for call in qdrant.delete.call_args_list:
        condition = call.kwargs["points_selector"].filter.must[0]
        assert condition.key == "version_id"
        assert condition.match.any == [old.version.version_id]


def test_delete_document_removes_all_versions_without_touching_other_document(
    tmp_home: Path,
) -> None:
    first = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="4" * 64,
        name="guide-v1.png",
    )
    second = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="5" * 64,
        name="guide-v2.png",
    )
    survivor = _persist_version(
        tmp_home,
        document_id="handbook",
        version_hash="6" * 64,
        name="handbook.png",
    )

    report = IngestService().delete_document("guide")

    assert report.was_known
    assert report.deleted_version_ids == [first.version.version_id, second.version.version_id]
    assert report.versions_removed == 2
    assert report.chunks_removed == 2
    assert not report.errors
    assert [row.version.version_id for row in load_records()] == [survivor.version.version_id]
    assert [chunk.document_id for chunk in read_documents()] == ["handbook"]
    assert (tmp_home / "assets" / survivor.asset.relative_path).exists()
    assert (tmp_home / "parsed" / physical_cache_id(survivor.asset.relative_path)).exists()


def test_document_lifecycle_delete_is_idempotent_for_unknown_identity(tmp_home: Path) -> None:
    service = IngestService()

    assert not service.delete_document("missing").was_known
    assert not service.delete_document_version("missing", "missing@1-aaaaaaaaaaaa").was_known


def test_delete_preserves_indexes_when_dependent_cache_cleanup_fails(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="a" * 64,
        name="guide.png",
    )
    service = IngestService()
    qdrant = Mock()
    monkeypatch.setattr("mm_asset_rag.service.get_qdrant_client", lambda: qdrant)
    from mm_asset_rag.service import _stage_delete_path

    def fail_parsed(source: Path, *args, **kwargs):
        if source.parent == tmp_home / "parsed":
            raise OSError("cache busy")
        return _stage_delete_path(source, *args, **kwargs)

    monkeypatch.setattr("mm_asset_rag.service._stage_delete_path", fail_parsed, raising=False)

    report = service.delete_document_version("guide", record.version.version_id)

    assert report.errors
    assert [row.version.version_id for row in load_records()] == [record.version.version_id]
    assert [chunk.document_version.version_id for chunk in read_documents()] == [
        record.version.version_id
    ]
    assert (tmp_home / "assets" / record.asset.relative_path).is_file()
    cache_key = physical_cache_id(record.asset.relative_path)
    assert (tmp_home / "parsed" / cache_key / "raw.jsonl").is_file()
    assert (tmp_home / "captions" / f"{cache_key}.jsonl").is_file()
    qdrant.delete.assert_not_called()


def test_delete_rolls_back_asset_and_parsed_cache_when_caption_staging_fails(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="b" * 64,
        name="guide-caption.png",
    )
    from mm_asset_rag.service import _stage_delete_path

    def fail_caption(source: Path, *args, **kwargs):
        if source.parent == tmp_home / "captions":
            raise OSError("caption busy")
        return _stage_delete_path(source, *args, **kwargs)

    monkeypatch.setattr("mm_asset_rag.service._stage_delete_path", fail_caption, raising=False)

    report = IngestService().delete_document_version("guide", record.version.version_id)

    assert any("caption" in error for error in report.errors)
    assert (tmp_home / "assets" / record.asset.relative_path).is_file()
    cache_key = physical_cache_id(record.asset.relative_path)
    assert (tmp_home / "parsed" / cache_key / "raw.jsonl").is_file()
    assert (tmp_home / "captions" / f"{cache_key}.jsonl").is_file()
    assert [row.version.version_id for row in load_records()] == [record.version.version_id]
    assert [chunk.document_version.version_id for chunk in read_documents()] == [
        record.version.version_id
    ]


def test_delete_restores_local_indexes_when_qdrant_delete_fails(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="c" * 64,
        name="guide-qdrant.png",
    )
    qdrant = Mock()
    from types import SimpleNamespace

    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="multimodal_text_768d")]
    )
    qdrant.delete.side_effect = RuntimeError("qdrant unavailable")
    monkeypatch.setattr("mm_asset_rag.service.get_qdrant_client", lambda: qdrant)

    report = IngestService().delete_document_version("guide", record.version.version_id)

    assert any("qdrant delete failed" in error for error in report.errors)
    assert [row.version.version_id for row in load_records()] == [record.version.version_id]
    assert [chunk.document_version.version_id for chunk in read_documents()] == [
        record.version.version_id
    ]
    assert (tmp_home / "assets" / record.asset.relative_path).is_file()
    cache_key = physical_cache_id(record.asset.relative_path)
    assert (tmp_home / "parsed" / cache_key / "raw.jsonl").is_file()
    assert (tmp_home / "captions" / f"{cache_key}.jsonl").is_file()


def test_delete_fails_closed_when_qdrant_collection_enumeration_fails(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="d" * 64,
        name="guide-enumeration.png",
    )
    qdrant = Mock()
    qdrant.get_collections.side_effect = RuntimeError("enumeration unavailable")
    monkeypatch.setattr("mm_asset_rag.service.get_qdrant_client", lambda: qdrant)

    report = IngestService().delete_document_version("guide", record.version.version_id)

    assert any("enumeration unavailable" in error for error in report.errors)
    qdrant.delete.assert_not_called()
    assert (tmp_home / "assets" / record.asset.relative_path).is_file()
    assert [row.version.version_id for row in load_records()] == [record.version.version_id]
    assert [chunk.document_version.version_id for chunk in read_documents()] == [
        record.version.version_id
    ]


def test_delete_preserves_unrecovered_staged_files_when_rollback_fails(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="1" * 64,
        name="guide-rollback.png",
    )
    qdrant = Mock()
    from types import SimpleNamespace

    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="multimodal_text_768d")]
    )
    qdrant.delete.side_effect = RuntimeError("qdrant unavailable")
    monkeypatch.setattr("mm_asset_rag.service.get_qdrant_client", lambda: qdrant)

    import mm_asset_rag.service as service_module

    original_replace = service_module.os.replace
    staging_parent = tmp_home / ".delete-staging"

    def fail_staged_restore(source, destination):
        if Path(source).is_relative_to(staging_parent):
            raise OSError("restore target busy")
        return original_replace(source, destination)

    monkeypatch.setattr(service_module.os, "replace", fail_staged_restore)

    report = IngestService().delete_document_version("guide", record.version.version_id)

    assert any("physical rollback failed" in error for error in report.errors)
    assert any(path.is_file() for path in staging_parent.rglob("*"))


def test_delete_scans_all_dimension_collections_even_with_active_overrides(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QDRANT_ACTIVE_TEXT_COLLECTION", "custom_text")
    monkeypatch.setenv("QDRANT_ACTIVE_IMAGE_COLLECTION", "custom_image")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    qdrant = Mock()
    from types import SimpleNamespace

    qdrant.get_collections.return_value = Mock(
        collections=[
            SimpleNamespace(name="multimodal_text_768d"),
            SimpleNamespace(name="multimodal_text_1024d"),
            SimpleNamespace(name="multimodal_image_512d"),
        ]
    )
    monkeypatch.setattr("mm_asset_rag.service.get_qdrant_client", lambda: qdrant)

    IngestService()._delete_qdrant_versions({"doc@1-hash"})

    assert {call.kwargs["collection_name"] for call in qdrant.delete.call_args_list} == {
        "custom_text",
        "custom_image",
        "multimodal_text_768d",
        "multimodal_text_1024d",
        "multimodal_image_512d",
    }


def test_delete_same_stem_path_keeps_other_physical_cache(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _persist_version(
        tmp_home,
        document_id="pdf-shared",
        version_hash="7" * 64,
        name="shared.png",
    )
    second = _persist_version(
        tmp_home,
        document_id="doc-shared",
        version_hash="8" * 64,
        name="other.png",
    )
    second_cache = physical_cache_id("documents/shared.png")
    second_asset = tmp_home / "assets" / "documents" / "shared.png"
    second_asset.parent.mkdir(parents=True)
    (tmp_home / "assets" / second.asset.relative_path).replace(second_asset)
    from mm_asset_rag.asset_index import DocumentVersionRecord

    second_record = DocumentVersionRecord(
        document=second.document,
        version=second.version,
        asset=PersistedAsset(second.asset.content_hash, "document", "documents/shared.png"),
    )
    from mm_asset_rag.service import _remove_document_version_records

    _remove_document_version_records(second.document.document_id, {second.version.version_id})
    upsert_record(second_record)
    old_cache = tmp_home / "parsed" / physical_cache_id(second.asset.relative_path)
    new_cache = tmp_home / "parsed" / second_cache
    old_cache.replace(new_cache)
    qdrant = Mock()
    from types import SimpleNamespace

    qdrant.get_collections.return_value = SimpleNamespace(collections=[])
    monkeypatch.setattr("mm_asset_rag.service.get_qdrant_client", lambda: qdrant)

    report = IngestService().delete_document_version(
        first.document.document_id, first.version.version_id
    )

    assert not report.errors
    assert not (tmp_home / "parsed" / physical_cache_id(first.asset.relative_path)).exists()
    assert (new_cache / "raw.jsonl").is_file()


def test_retry_failed_only_uses_recorded_statuses(tmp_home: Path) -> None:
    ok = _persist_version(
        tmp_home,
        document_id="ok1",
        version_hash="0" * 64,
        name="ok1.png",
    )
    bad = _persist_version(
        tmp_home,
        document_id="bad1",
        version_hash="1" * 64,
        name="bad1.png",
    )
    assets = [
        IngestAsset(
            asset_id=physical_cache_id(record.asset.relative_path),
            title=record.document.title,
            source_type=record.asset.source_type,
            relative_path=record.asset.relative_path,
            asset_dir=tmp_home / "assets",
        )
        for record in (ok, bad)
    ]
    service = IngestService()
    original = TaskRecord(
        task_id="origstatus01",
        kind="parse",
        status="partial",
        total=2,
        uploaded_files=[a.relative_path for a in assets],
        version_statuses={ok.version.version_id: "ok", bad.version.version_id: "failed"},
    )
    service._tasks[original.task_id] = original
    service._rebuild_assets_for_retry = lambda _uploaded: list(assets)  # type: ignore[method-assign]
    with patch.object(service, "_spawn") as spawn:
        service.retry_task(original.task_id, failed_only=True)
    # _spawn was called with _run_parse_task and ParseOptions whose assets are the failed-only subset.
    assert spawn.call_count == 1
    args, _kwargs = spawn.call_args
    target, _rec, options = args
    assert target.__name__ == "_run_parse_task"
    assert [a.asset_id for a in options.assets] == [physical_cache_id(bad.asset.relative_path)]


def test_retry_failed_only_all_ok_raises(tmp_home: Path) -> None:
    ok = _persist_version(
        tmp_home,
        document_id="ok1",
        version_hash="2" * 64,
        name="ok1.png",
    )
    assets = [
        IngestAsset(
            asset_id="ok1",
            title="ok1",
            source_type=ok.asset.source_type,
            relative_path=ok.asset.relative_path,
            asset_dir=tmp_home / "assets",
        )
    ]
    service = IngestService()
    original = TaskRecord(
        task_id="origstatus02",
        kind="parse",
        status="partial",
        total=1,
        uploaded_files=[a.relative_path for a in assets],
        version_statuses={ok.version.version_id: "ok"},
    )
    service._tasks[original.task_id] = original
    service._rebuild_assets_for_retry = lambda uploaded: list(assets)  # type: ignore[method-assign]
    with pytest.raises(FileNotFoundError, match="no failed or skipped assets"):
        service.retry_task(original.task_id, failed_only=True)


def test_retry_rebuild_uses_physical_cache_id_for_parser_cache(tmp_home: Path) -> None:
    record = _persist_version(
        tmp_home,
        document_id="retry-cache",
        version_hash="9" * 64,
        name="shared.png",
    )
    original = TaskRecord(
        task_id="retrycache01",
        kind="parse",
        status="failed",
        total=1,
        uploaded_files=[record.asset.relative_path],
        version_statuses={record.version.version_id: "failed"},
    )
    service = IngestService()
    service._tasks[original.task_id] = original

    with patch.object(service, "_spawn") as spawn:
        service.retry_task(original.task_id)

    _target, _retry_record, options = spawn.call_args.args
    [rebuilt] = options.assets
    expected_cache_id = physical_cache_id(record.asset.relative_path)
    assert rebuilt.asset_id == expected_cache_id
    assert (tmp_home / "parsed" / rebuilt.asset_id / "raw.jsonl").is_file()


def test_force_retry_clears_parsed_cache(tmp_home: Path, monkeypatch) -> None:
    from mm_asset_rag.paths import get_parsed_dir

    version = _persist_version(
        tmp_home,
        document_id="force",
        version_hash="d" * 64,
        name="force.png",
    )
    asset = IngestAsset(
        asset_id=physical_cache_id(version.asset.relative_path),
        title="force",
        source_type=version.asset.source_type,
        relative_path=version.asset.relative_path,
        asset_dir=tmp_home / "assets",
    )
    parsed_dir = get_parsed_dir() / asset.asset_id
    (parsed_dir / "page_0.md").write_text("# cached", encoding="utf-8")
    assert parsed_dir.exists()

    rec = TaskRecord(
        task_id="force01",
        kind="parse",
        status="running",
        total=1,
        force=True,
    )
    rec.uploaded_files = [asset.relative_path]

    service = IngestService()
    from mm_asset_rag.service import ParseOptions, _do_parse

    called: list[str] = []

    def fake_parser(asset_obj, **kwargs):
        called.append(asset_obj.asset_id)
        from mm_asset_rag.schema import ParsedChunk

        return [ParsedChunk(text="x", metadata={"asset_id": asset_obj.asset_id})]

    import mm_asset_rag.service as svc_mod

    orig_parser = svc_mod.get_parser

    def fake_get_parser(kind, name):
        class P:
            def __init__(self, asset, **kwargs):
                self.asset = asset

            def parse(self, asset, **kwargs):
                return fake_parser(asset)

        return P(asset=None)

    svc_mod.get_parser = fake_get_parser
    try:
        _do_parse(service, rec, ParseOptions(assets=[asset]))
    finally:
        svc_mod.get_parser = orig_parser

    assert not parsed_dir.exists()
    assert called == [asset.asset_id]


def test_force_retry_replaces_only_current_document_version_chunks(tmp_home: Path) -> None:
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.schema import ParsedChunk
    from mm_asset_rag.service import ParseOptions, _do_parse

    sibling = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="7" * 64,
        name="guide-v1.png",
    )
    current = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="8" * 64,
        name="guide-v2.png",
    )
    other = _persist_version(
        tmp_home,
        document_id="handbook",
        version_hash="9" * 64,
        name="handbook.png",
    )
    asset = IngestAsset(
        asset_id=physical_cache_id(current.asset.relative_path),
        title="guide",
        source_type=current.asset.source_type,
        relative_path=current.asset.relative_path,
        asset_dir=tmp_home / "assets",
    )

    rec = TaskRecord(
        task_id="force02",
        kind="parse",
        status="running",
        total=1,
        force=True,
    )
    rec.uploaded_files = [asset.relative_path]

    service = IngestService()

    def fake_get_parser(kind, name):
        class P:
            def parse(self, asset_obj, **kwargs):
                return [
                    ParsedChunk(text="fresh chunk", metadata={"asset_id": asset_obj.asset_id})
                ]

        return P()

    orig_parser = svc_mod.get_parser
    svc_mod.get_parser = fake_get_parser
    try:
        _do_parse(service, rec, ParseOptions(assets=[asset]))
    finally:
        svc_mod.get_parser = orig_parser

    chunks = read_documents()
    assert [(chunk.document_id, chunk.document_version.version_id) for chunk in chunks] == [
        ("guide", sibling.version.version_id),
        ("handbook", other.version.version_id),
        ("guide", current.version.version_id),
    ]
    [fresh] = [
        chunk for chunk in chunks if chunk.document_version.version_id == current.version.version_id
    ]
    assert fresh.text == "fresh chunk"
    assert "asset_id" not in fresh.metadata


def test_remove_document_version_rows_is_exactly_scoped(tmp_home: Path) -> None:
    old = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="a" * 64,
        name="guide-v1.png",
    )
    current = _persist_version(
        tmp_home,
        document_id="guide",
        version_hash="b" * 64,
        name="guide-v2.png",
    )
    other = _persist_version(
        tmp_home,
        document_id="handbook",
        version_hash="c" * 64,
        name="handbook.png",
    )
    from mm_asset_rag.service import _remove_document_version_rows_from_documents_jsonl

    removed = _remove_document_version_rows_from_documents_jsonl(
        "guide", {current.version.version_id}
    )

    assert removed == 1
    assert [
        (chunk.document_id, chunk.document_version.version_id) for chunk in read_documents()
    ] == [
        ("guide", old.version.version_id),
        ("handbook", other.version.version_id),
    ]


def test_ingest_task_records_indexed_status_on_success(tmp_home: Path) -> None:
    version = _persist_version(
        tmp_home,
        document_id="ing",
        version_hash="5" * 64,
        name="ing.png",
    )
    asset = IngestAsset(
        asset_id=physical_cache_id(version.asset.relative_path),
        title=version.document.title,
        source_type=version.asset.source_type,
        relative_path=version.asset.relative_path,
        asset_dir=tmp_home / "assets",
    )
    rec = TaskRecord(task_id="ingest1", kind="ingest", status="running", total=1)
    rec.uploaded_files = [asset.relative_path]
    rec.version_statuses = {version.version.version_id: "ok"}

    service = IngestService()
    from mm_asset_rag.registry import get_backend as _gb
    from mm_asset_rag.service import ParseOptions, _run_ingest_task

    class FakeBackend:
        def upsert_text(self, progress_cb=None):
            return (3, "fake_text")

        def upsert_image(self, progress_cb=None):
            return (2, "fake_image")

    original_get_backend = _gb

    def fake_get_backend(name):
        return FakeBackend()

    import mm_asset_rag.service as svc_mod

    svc_mod.get_backend = fake_get_backend
    try:
        _run_ingest_task(service, rec, ParseOptions(assets=[asset]))
    finally:
        svc_mod.get_backend = original_get_backend

    assert rec.version_statuses[version.version.version_id] == "indexed"
    assert rec.status in {"done", "partial"}
    assert rec.finished_at is not None


def test_ingest_task_records_failed_index_on_upsert_crash(tmp_home: Path) -> None:
    version = _persist_version(
        tmp_home,
        document_id="crash",
        version_hash="6" * 64,
        name="crash.png",
    )
    asset = IngestAsset(
        asset_id=physical_cache_id(version.asset.relative_path),
        title=version.document.title,
        source_type=version.asset.source_type,
        relative_path=version.asset.relative_path,
        asset_dir=tmp_home / "assets",
    )
    rec = TaskRecord(task_id="ingest2", kind="ingest", status="running", total=1)
    rec.uploaded_files = [asset.relative_path]
    rec.version_statuses = {version.version.version_id: "ok"}

    service = IngestService()
    from mm_asset_rag.service import ParseOptions, _run_ingest_task

    class FakeBackend:
        def upsert_text(self, progress_cb=None):
            raise RuntimeError("qdrant down")

        def upsert_image(self, progress_cb=None):
            return (0, "fake_image")

    import mm_asset_rag.service as svc_mod

    original_get_backend = svc_mod.get_backend
    svc_mod.get_backend = lambda name: FakeBackend()
    try:
        _run_ingest_task(service, rec, ParseOptions(assets=[asset]))
    finally:
        svc_mod.get_backend = original_get_backend

    assert rec.version_statuses[version.version.version_id] == "failed_index"
    assert rec.status == "failed"


def test_retry_failed_only_includes_failed_index(tmp_home: Path) -> None:
    ok = _persist_version(
        tmp_home,
        document_id="ok1",
        version_hash="3" * 64,
        name="ok1.png",
    )
    bad = _persist_version(
        tmp_home,
        document_id="bad1",
        version_hash="4" * 64,
        name="bad1.png",
    )
    assets = [
        IngestAsset(
            asset_id=physical_cache_id(record.asset.relative_path),
            title=record.document.title,
            source_type=record.asset.source_type,
            relative_path=record.asset.relative_path,
            asset_dir=tmp_home / "assets",
        )
        for record in (ok, bad)
    ]
    service = IngestService()
    original = TaskRecord(
        task_id="origidx01",
        kind="ingest",
        status="partial",
        total=2,
        uploaded_files=[a.relative_path for a in assets],
        version_statuses={ok.version.version_id: "indexed", bad.version.version_id: "failed_index"},
    )
    service._tasks[original.task_id] = original
    service._rebuild_assets_for_retry = lambda _uploaded: list(assets)  # type: ignore[method-assign]
    with patch.object(service, "_spawn") as spawn:
        service.retry_task(original.task_id, failed_only=True)
    assert spawn.call_count == 1
    args, _kwargs = spawn.call_args
    _target, _rec, options = args
    assert [a.asset_id for a in options.assets] == [physical_cache_id(bad.asset.relative_path)]


def test_retry_force_and_failed_only_clear_only_failed_cache(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mm_asset_rag.paths import get_parsed_dir

    healthy = _persist_version(
        tmp_home,
        document_id="healthy",
        version_hash="e" * 64,
        name="ok1.png",
    )
    failed = _persist_version(
        tmp_home,
        document_id="failed",
        version_hash="f" * 64,
        name="bad1.png",
    )
    assets = [
        IngestAsset(
            asset_id=physical_cache_id(record.asset.relative_path),
            title=record.document.title,
            source_type=record.asset.source_type,
            relative_path=record.asset.relative_path,
            asset_dir=tmp_home / "assets",
        )
        for record in (healthy, failed)
    ]

    # Construct a retried task record the same way retry_task would:
    # failed-only already narrowed to bad1, force=True to clear cache.
    rec = TaskRecord(
        task_id="combo01",
        kind="parse",
        status="running",
        total=1,
        uploaded_files=["images/bad1.png"],
        force=True,
        failed_only=True,
        version_statuses={
            failed.version.version_id: "failed",
            healthy.version.version_id: "ok",
        },
    )

    service = IngestService()
    # Stub parser; we just want the force rmtree to fire before parse.
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import ParseOptions, _do_parse

    called: list[str] = []

    class StubParser:
        def parse(self, asset, **_kwargs):
            from mm_asset_rag.schema import ParsedChunk

            called.append(asset.asset_id)
            return [ParsedChunk(text="x", metadata={"asset_id": asset.asset_id})]

    monkeypatch.setattr(svc_mod, "get_parser", lambda kind, name: StubParser())

    # Run the same parse loop retry_task would dispatch (synchronously).
    _do_parse(service, rec, ParseOptions(assets=[assets[1]]))  # only bad1

    # ok1 cache survives (its parsed/ dir is still present, untouched).
    assert (
        get_parsed_dir() / physical_cache_id(healthy.asset.relative_path) / "raw.jsonl"
    ).exists()
    # bad1 cache was force-cleared before parse, then parsed again.
    assert not (get_parsed_dir() / physical_cache_id(failed.asset.relative_path)).exists()
    assert called == [physical_cache_id(failed.asset.relative_path)]
    assert rec.version_statuses[failed.version.version_id] in {"ok", "skipped", "failed"}


# ─── stream_task multi-subscriber ───────────────────────────────────────


def test_stream_task_supports_concurrent_subscribers(tmp_home: Path) -> None:
    """Two callers streaming the same task id should both receive the
    snapshot + done events. The previous implementation used
    ``_stream_events[task_id] = (event, ...)`` and dropped the older
    subscriber's Event on the second subscribe, leaving it stuck.
    """
    import threading

    from mm_asset_rag.service import TaskStatus

    service = IngestService()
    rec = TaskRecord(task_id="multisub01", kind="ingest", total=1, status=TaskStatus.DONE)
    rec.uploaded_files = ["images/multi.png"]
    rec.finished_at = 0.0
    service._tasks[rec.task_id] = rec

    received_a: list[dict[str, object]] = []
    received_b: list[dict[str, object]] = []
    start = threading.Event()

    def consume(label: str, sink: list[dict[str, object]]) -> None:
        start.wait()
        for ev in service.stream_task(rec.task_id, heartbeat=0.1):
            sink.append(ev)

    t1 = threading.Thread(target=consume, args=("a", received_a), daemon=True)
    t2 = threading.Thread(target=consume, args=("b", received_b), daemon=True)
    t1.start()
    t2.start()
    start.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert received_a and received_b, "both subscribers should receive events"
    statuses_a = [e.get("status") for e in received_a if e.get("event") == "done"]
    statuses_b = [e.get("status") for e in received_b if e.get("event") == "done"]
    assert TaskStatus.DONE in statuses_a
    assert TaskStatus.DONE in statuses_b


# ─── dispatch_search sandbox ────────────────────────────────────────────


def test_dispatch_search_rejects_absolute_image_path(tmp_home: Path) -> None:
    """``image_path`` must be a relative path; absolute paths and ``..``
    traversal bounce at the API boundary so the CLIP encoder can't be
    pointed at ``/etc/passwd`` or similar.
    """
    from mm_asset_rag.search_service import resolve_sandboxed_image_path

    with pytest.raises(ValueError, match="must be relative"):
        resolve_sandboxed_image_path("/etc/passwd")
    with pytest.raises(ValueError, match="must be relative"):
        resolve_sandboxed_image_path("/absolute/image.png")


def test_dispatch_search_rejects_parent_traversal(tmp_home: Path) -> None:
    from mm_asset_rag.search_service import resolve_sandboxed_image_path

    with pytest.raises(ValueError, match="outside assets"):
        resolve_sandboxed_image_path("../escape.png")
    with pytest.raises(ValueError, match="outside assets"):
        resolve_sandboxed_image_path("images/../../escape.png")


def test_dispatch_search_rejects_missing_file(tmp_home: Path) -> None:
    from mm_asset_rag.search_service import resolve_sandboxed_image_path

    assets_dir = tmp_home / "assets"
    assets_dir.mkdir()
    with pytest.raises(ValueError, match="not found"):
        resolve_sandboxed_image_path("images/ghost.png")


def test_dispatch_search_accepts_file_inside_assets(tmp_home: Path) -> None:
    from mm_asset_rag.search_service import resolve_sandboxed_image_path

    assets_dir = tmp_home / "assets"
    target = assets_dir / "images" / "ok.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    resolved = resolve_sandboxed_image_path("images/ok.png")
    assert resolved is not None
    assert resolved.is_relative_to(assets_dir)
    assert resolved.name == "ok.png"


def test_dispatch_search_rejects_symlink_escape(tmp_home: Path) -> None:
    """A symlink inside ``assets/`` that points outside must be rejected.

    Without symlink-aware resolution the user could plant
    ``assets/images/leak.png -> /etc/passwd`` during a write window
    (or another local user could do it on a shared server) and the
    CLIP encoder would happily follow the link. ``resolve(strict=False)``
    followed by ``is_relative_to(assets_dir)`` catches this case.
    """
    from mm_asset_rag.search_service import resolve_sandboxed_image_path

    assets_dir = tmp_home / "assets"
    images_dir = assets_dir / "images"
    images_dir.mkdir(parents=True)
    # Sensitive target outside ``assets_dir``.
    sensitive = tmp_home / "sibling" / "sensitive.png"
    sensitive.parent.mkdir(parents=True)
    sensitive.write_bytes(b"\x89PNG\r\n\x1a\n")
    link_path = images_dir / "leak.png"
    try:
        link_path.symlink_to(sensitive)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform specific
        pytest.skip(f"symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="outside assets"):
        resolve_sandboxed_image_path("images/leak.png")


def test_list_tasks_orders_by_updated_at_desc(tmp_home: Path) -> None:
    """``list_tasks`` reads SQLite directly and returns rows ordered by
    most recent ``updated_at`` (not by insertion order, which is what
    the in-memory dict gave us). Useful because a worker that crashes
    before the next ``_patch`` lands still surfaces in the correct
    place after recovery.
    """
    import sqlite3

    from mm_asset_rag.service import IngestService, TaskRecord

    service = IngestService()
    base = time.time()
    a = TaskRecord(task_id="alpha01", kind="parse")
    b = TaskRecord(task_id="beta02", kind="parse")
    c = TaskRecord(task_id="gamma03", kind="ingest")

    db_path = service._tasks_db_path()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            "task_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS tasks_updated_at_idx ON tasks (updated_at DESC)")
        for rec, ts in ((a, base), (b, base + 1.0), (c, base + 2.0)):
            conn.execute(
                "INSERT OR REPLACE INTO tasks (task_id, payload, updated_at) VALUES (?, ?, ?)",
                (rec.task_id, json.dumps(asdict(rec)), ts),
            )

    fresh = IngestService()
    tasks = fresh.list_tasks()
    assert [t.task_id for t in tasks] == ["gamma03", "beta02", "alpha01"]


def test_load_history_round_trip(tmp_home: Path) -> None:
    """Persist a record, instantiate a fresh service, and confirm
    load_history sees it in ``self._tasks``.
    """
    from mm_asset_rag.service import IngestService, TaskRecord

    service = IngestService()
    rec = TaskRecord(
        task_id="round01",
        kind="parse",
        status="done",
        version_statuses={"guide@2-aaaaaaaaaaaa": "indexed"},
    )
    service._persist(rec)

    fresh = IngestService()
    fresh.load_history()
    loaded = fresh.get_task("round01")
    assert loaded is not None
    assert loaded.task_id == "round01"
    assert loaded.status == "done"
    assert loaded.version_statuses == {"guide@2-aaaaaaaaaaaa": "indexed"}


def test_persist_surfaces_oserror_on_rec(tmp_home: Path, monkeypatch) -> None:
    """When the SQLite store cannot be written (disk full, EROFS, …) the
    service must surface the failure on the record itself so a
    subsequent ``/tasks/{id}`` poll sees it instead of a misleading
    success.

    Patch ``sqlite3.connect`` to raise before any real DB
    work happens so we exercise the failure path under the current
    write strategy.
    """

    from mm_asset_rag.service import IngestService, TaskRecord

    service = IngestService()
    rec = TaskRecord(task_id="persist_test", kind="parse")

    def boom(*args, **kwargs):
        # Return a context manager that raises on ``__enter__``,
        # matching the ``with sqlite3.connect(...) as conn:`` shape
        # production uses — without raising from ``connect()``
        # itself (which Python would treat as a broken ctx-manager).
        class _RaiseOnEnter:
            def __enter__(self):
                raise OSError("EROFS: read-only filesystem (test stub)")

            def __exit__(self, *args):
                return False

        return _RaiseOnEnter()

    # Stub ``sqlite3.connect`` — production code imports sqlite3
    # locally, so patching the stdlib namespace once covers every
    # ``import sqlite3`` regardless of which call site resolves it.
    monkeypatch.setattr("sqlite3.connect", boom)
    service._persist(rec)

    assert rec.error is not None
    assert "persist failed" in rec.error
    assert "EROFS" in rec.error
    # In-memory rec is still usable; we did not raise.
    assert rec.task_id == "persist_test"


def test_parse_assets_empty_list_returns_done_without_thread(tmp_home: Path) -> None:
    """``parse_assets([])`` records a completed task synchronously instead of
    spawning a daemon thread that just immediately exits — keeps the
    history honest without burning a worker.
    """
    from mm_asset_rag.service import TaskStatus

    service = IngestService()
    rec = service.parse_assets([])
    assert rec.status == TaskStatus.DONE
    assert rec.total == 0
    assert rec.processed == 0
    assert rec.finished_at > 0


# ─── M2: search-cache invalidation on reindex / ingest success ────────


def test_reindex_invalidates_vocab_and_bm25_idf_caches(tmp_home: Path, monkeypatch) -> None:
    """``reindex`` force-recreates the Qdrant collections, so the
    in-process vocab + BM25-zh IDF caches (derived from the *previous*
    collection state) must be dropped before the next query. The
    invalidation is best-effort: a missing invalidator function
    (older build) must not crash the reindex path.
    """
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import IngestService

    service = IngestService()

    calls: list[str] = []

    class FakeBackend:
        def upsert_text(self, *, force_recreate=False, progress_cb=None):
            return (0, "fake_text")

        def upsert_image(self, *, force_recreate=False, progress_cb=None):
            return (0, "fake_image")

    orig_get_backend = svc_mod.get_backend
    svc_mod.get_backend = lambda name: FakeBackend()

    def fake_invalidate(name):
        def _fn():
            calls.append(name)

        return _fn

    monkeypatch.setattr(svc_mod, "invalidate_vocab_cache", fake_invalidate("vocab"))
    monkeypatch.setattr(svc_mod, "invalidate_bm25_zh_idf_cache", fake_invalidate("bm25_idf"))
    try:
        service.reindex()
    finally:
        svc_mod.get_backend = orig_get_backend

    assert calls == ["vocab", "bm25_idf"]


def test_reindex_invalidates_current_caches(tmp_home: Path, monkeypatch) -> None:
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import IngestService

    service = IngestService()

    class FakeBackend:
        def upsert_text(self, *, force_recreate=False, progress_cb=None):
            return (0, "fake_text")

        def upsert_image(self, *, force_recreate=False, progress_cb=None):
            return (0, "fake_image")

    orig_get_backend = svc_mod.get_backend
    svc_mod.get_backend = lambda name: FakeBackend()

    monkeypatch.setattr(svc_mod, "invalidate_vocab_cache", lambda: None)
    monkeypatch.setattr(svc_mod, "invalidate_bm25_zh_idf_cache", lambda: None)
    try:
        results = service.reindex()
        assert any("text" in r for r in results)
    finally:
        svc_mod.get_backend = orig_get_backend


def test_ingest_task_invalidates_caches_on_success(tmp_home: Path, monkeypatch) -> None:
    """A successful ingest (``upsert_text`` + ``upsert_image`` both
    return without raising) must drop the vocab + BM25-zh IDF caches
    after the points are written and before the task is marked
    terminal, so a query issued right after the UI flips to "done"
    rebuilds against the new collection state.
    """
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import IngestService, ParseOptions, TaskRecord, _run_ingest_task

    version = _persist_version(
        tmp_home,
        document_id="ing",
        version_hash="7" * 64,
        name="ing.png",
    )
    asset = IngestAsset(
        asset_id=physical_cache_id(version.asset.relative_path),
        title=version.document.title,
        source_type=version.asset.source_type,
        relative_path=version.asset.relative_path,
        asset_dir=tmp_home / "assets",
    )
    rec = TaskRecord(task_id="ingest-inv", kind="ingest", status="running", total=1)
    rec.uploaded_files = [asset.relative_path]
    rec.version_statuses = {version.version.version_id: "ok"}

    service = IngestService()

    class FakeBackend:
        def upsert_text(self, progress_cb=None):
            return (1, "fake_text")

        def upsert_image(self, progress_cb=None):
            return (0, "fake_image")

    orig_get_backend = svc_mod.get_backend
    svc_mod.get_backend = lambda name: FakeBackend()

    calls: list[str] = []

    def _record(name):
        def _fn():
            calls.append(name)

        return _fn

    monkeypatch.setattr(svc_mod, "invalidate_vocab_cache", _record("vocab"))
    monkeypatch.setattr(svc_mod, "invalidate_bm25_zh_idf_cache", _record("bm25_idf"))
    try:
        _run_ingest_task(service, rec, ParseOptions(assets=[asset]))
    finally:
        svc_mod.get_backend = orig_get_backend

    assert calls == ["vocab", "bm25_idf"]
    assert rec.status in {"done", "partial"}
    assert rec.finished_at is not None


# ─── M8: list_tasks overlays in-memory recs on SQLite rows ────────────


def test_list_tasks_overlays_in_memory_recs_on_sqlite_rows(tmp_home: Path) -> None:
    """For any ``task_id`` present in both SQLite and ``self._tasks``,
    the in-memory copy wins — it's the one ``_patch`` mutates, so it's
    strictly fresher than the persisted row. Without the overlay a
    just-patched task whose SQLite write is still in flight would show
    stale status/``current`` in the UI.
    """
    import sqlite3

    from mm_asset_rag.service import IngestService, TaskRecord

    service = IngestService()
    base = time.time()
    rec = TaskRecord(task_id="overlay01", kind="parse", status="pending", total=1)
    # Persist an old (stale) snapshot to SQLite.
    with sqlite3.connect(str(service._tasks_db_path())) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            "task_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, payload, updated_at) VALUES (?, ?, ?)",
            (rec.task_id, json.dumps(asdict(rec)), base),
        )
    # Mutate the in-memory copy — this is the fresher state SQLite
    # doesn't have yet.
    rec.status = "done"
    rec.current = "patched in memory"
    rec.finished_at = base + 5.0
    service._tasks[rec.task_id] = rec

    listed = service.list_tasks()
    assert listed
    hit = next(t for t in listed if t.task_id == rec.task_id)
    assert hit.status == "done"
    assert hit.current == "patched in memory"
    assert hit.finished_at == base + 5.0


def test_list_tasks_surfaces_memory_only_recs_not_yet_persisted(tmp_home: Path) -> None:
    """``_new_task`` adds to ``self._tasks`` and *then* calls
    ``_persist``; ``list_tasks`` must surface such a memory-only rec
    (no SQLite row yet) so the UI doesn't briefly lose the just-created
    task. We simulate the pre-persist window by inserting into
    ``_tasks`` only.
    """
    from mm_asset_rag.service import IngestService, TaskRecord

    service = IngestService()
    rec = TaskRecord(task_id="memonly01", kind="parse", status="running", total=1)
    service._tasks[rec.task_id] = rec
    listed = service.list_tasks()
    ids = [t.task_id for t in listed]
    assert "memonly01" in ids


def test_cancel_task_marks_running_task_cancelled(tmp_home: Path) -> None:
    """cancel_task sets the stop flag and patches status → cancelled."""
    service = IngestService()
    rec = TaskRecord(task_id="runtask0001", kind="ingest", status="running", total=3)
    service._tasks[rec.task_id] = rec

    out = service.cancel_task(rec.task_id)

    assert out.task_id == rec.task_id
    assert out.status == "cancelled"
    assert rec.status == "cancelled"
    assert rec.finished_at is not None
    # The per-task stop flag was created and set.
    assert service._is_cancelled(rec.task_id)


def test_cancel_task_unknown_raises(tmp_home: Path) -> None:
    service = IngestService()
    with pytest.raises(KeyError, match="unknown task"):
        service.cancel_task("does-not-exist")


def test_cancel_task_terminal_task_is_noop(tmp_home: Path) -> None:
    """Cancelling an already-terminal task returns it unchanged (no resurrect)."""
    service = IngestService()
    rec = TaskRecord(task_id="donetask001", kind="ingest", status="done", total=1)
    service._tasks[rec.task_id] = rec

    out = service.cancel_task(rec.task_id)

    assert out.status == "done"  # not flipped to cancelled
    # Still records a stop flag (harmless), but status untouched.
    assert service._is_cancelled(rec.task_id)


def test_worker_stops_at_checkpoint_when_cancelled(tmp_home: Path) -> None:
    """The parse worker checks _is_cancelled between assets: once the flag
    is set, it stops before the next asset and leaves the task cancelled.

    End-to-end for the cooperative-cancel checkpoint (not covered by the
    cancel_task unit tests, which only exercise the flag-set path).
    """
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.schema import ParsedChunk
    from mm_asset_rag.service import _do_parse

    records = [
        _persist_version(
            tmp_home,
            document_id=f"cancel-{index}",
            version_hash=str(index) * 64,
            name=f"a{index}.png",
            with_caches=False,
        )
        for index in (1, 2, 3)
    ]
    a1, a2, a3 = [_transient_asset(record, tmp_home) for record in records]
    service = IngestService()
    rec = TaskRecord(task_id="canceltask1", kind="parse", status="running", total=3)
    service._tasks[rec.task_id] = rec
    service._cancel_flags[rec.task_id] = __import__("threading").Event()

    parsed_ids: list[str] = []

    def fake_parser(asset, **kwargs):
        parsed_ids.append(asset.asset_id)
        # After the first asset is parsed, request cancellation. The worker
        # should check the flag before asset #2 and stop.
        if asset.asset_id == a1.asset_id:
            service._cancel_flags[rec.task_id].set()
            service._patch(rec, status="cancelled", finished_at=time.time(), current="cancelled")
        return [ParsedChunk(text="x", metadata={"asset_id": asset.asset_id})]

    def fake_get_parser(kind, name):
        class P:
            def __init__(self, *args, **kwargs):
                pass

            def parse(self, asset, **kwargs):
                return fake_parser(asset)

        return P()

    orig = svc_mod.get_parser
    svc_mod.get_parser = fake_get_parser
    try:
        _do_parse(service, rec, ParseOptions(assets=[a1, a2, a3]))
    finally:
        svc_mod.get_parser = orig

    # Only the first asset was parsed; the checkpoint before #2 saw cancel.
    assert parsed_ids == [a1.asset_id]
    assert rec.status == "cancelled"


def test_do_parse_keeps_cancelled_when_cancel_during_last_asset(tmp_home: Path) -> None:
    """If the user cancels *during* the last asset's parse (past the
    per-version checkpoint at the loop top), ``cancel_task`` has set the
    cancel flag but may not have patched status=CANCELLED yet (flag.set
    and the status patch in cancel_task are two non-atomic steps). The
    final ``_patch`` must set status=CANCELLED explicitly — otherwise the
    task stays ``running`` and ``load_history`` rewrites it to
    ``interrupted`` on next boot, losing the cancel request. Regression:
    the final patch only set finished_at/current, not status, when the
    cancel flag was set."""
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.schema import ParsedChunk
    from mm_asset_rag.service import _do_parse

    records = [
        _persist_version(
            tmp_home,
            document_id=f"cancel-last-{index}",
            version_hash=str(index + 3) * 64,
            name=f"a{index}.png",
            with_caches=False,
        )
        for index in (1, 2)
    ]
    a1, a2 = [_transient_asset(record, tmp_home) for record in records]
    service = IngestService()
    rec = TaskRecord(task_id="canceltask2", kind="parse", status="running", total=2)
    service._tasks[rec.task_id] = rec
    service._cancel_flags[rec.task_id] = __import__("threading").Event()

    def fake_parser(asset, **kwargs):
        # On the LAST asset, simulate cancel_task's flag being set but
        # the status patch NOT yet applied (the race window between
        # cancel_task's flag.set and its _patch(status=CANCELLED)). The
        # final _patch in _do_parse must set status=CANCELLED itself.
        if asset.asset_id == a2.asset_id:
            service._cancel_flags[rec.task_id].set()
            # Deliberately do NOT patch status — leave it "running".
        return [ParsedChunk(text="x", metadata={"asset_id": asset.asset_id})]

    def fake_get_parser(kind, name):
        class P:
            def __init__(self, *args, **kwargs):
                pass

            def parse(self, asset, **kwargs):
                return fake_parser(asset)

        return P()

    orig = svc_mod.get_parser
    svc_mod.get_parser = fake_get_parser
    try:
        _do_parse(service, rec, ParseOptions(assets=[a1, a2]))
    finally:
        svc_mod.get_parser = orig

    # The cancel must stick — _do_parse set CANCELLED explicitly even
    # though cancel_task hadn't patched status yet.
    assert rec.status == "cancelled", f"expected cancelled, got {rec.status!r}"


def test_index_phase_stops_when_cancelled(tmp_home: Path) -> None:
    """The index phase checks _is_cancelled at each progress tick (one per
    upsert batch) via _progress → raises _TaskCancelled, which the ingest
    worker catches cleanly (no failed/failed_index status flip).

    Covers L1's index-stage cancel path (the parse-stage checkpoint is
    covered by test_worker_stops_at_checkpoint_when_cancelled)."""
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import _run_ingest_task

    service = IngestService()
    # Pretend parse already finished successfully (parse_status=done) so
    # _run_ingest_task proceeds straight to the index phase.
    rec = TaskRecord(
        task_id="idxcancel01",
        kind="ingest",
        status="done",
        total=1,
        finished_at=None,
    )
    service._tasks[rec.task_id] = rec
    service._cancel_flags[rec.task_id] = __import__("threading").Event()
    parse_options = ParseOptions(assets=[])

    # Skip the real parse stage — we only want to drive the index path.
    def fake_parse(_svc, _rec, _opts):
        # Leave rec as parse-done so _run_ingest_task continues to index.
        _svc._patch(_rec, status="done", current="parse done")

    ticks = {"n": 0}

    class FakeBackend:
        def upsert_text(self, *, progress_cb=None, force_recreate=False):
            # First tick: normal. Second tick: cancel arrives mid-index.
            ticks["n"] += 1
            if progress_cb:
                progress_cb(1, 10, "indexing")
            # Simulate cancel arriving between batches.
            service._cancel_flags[rec.task_id].set()
            service._patch(rec, status="cancelled", finished_at=time.time(), current="cancelled")
            # Next progress tick should raise _TaskCancelled.
            if progress_cb:
                progress_cb(2, 10, "indexing")  # this raises _TaskCancelled
            return 1, "text_coll"

        def upsert_image(self, *, progress_cb=None, force_recreate=False):
            return 1, "img_coll"

    with (
        patch.object(svc_mod, "_run_parse_task", fake_parse),
        patch.object(svc_mod, "get_backend", lambda name: FakeBackend()),
    ):
        _run_ingest_task(service, rec, parse_options)

    # The _TaskCancelled was caught cleanly; status stays cancelled (not
    # flipped to failed/failed_index by the BaseException branch).
    assert rec.status == "cancelled"
    assert ticks["n"] == 1  # upsert_text ran (and was interrupted at tick 2)


def test_index_success_patch_does_not_overwrite_cancel(tmp_home: Path) -> None:
    """If cancel arrives *after* the last index progress tick but *before*
    the success _patch at the end of _run_ingest_task, the success patch
    must not overwrite CANCELLED with parse_status (done/partial). Symmetric
    with the parse-stage cancel guard. Regression: the success patch set
    status=parse_status unconditionally, flipping the UI from "cancelled"
    to "done"."""
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import _run_ingest_task

    service = IngestService()
    rec = TaskRecord(
        task_id="idxcancel02",
        kind="ingest",
        status="done",
        total=1,
        finished_at=None,
    )
    service._tasks[rec.task_id] = rec
    service._cancel_flags[rec.task_id] = __import__("threading").Event()
    parse_options = ParseOptions(assets=[])

    def fake_parse(_svc, _rec, _opts):
        _svc._patch(_rec, status="done", current="parse done")

    class FakeBackend:
        def upsert_text(self, *, progress_cb=None, force_recreate=False):
            if progress_cb:
                progress_cb(1, 10, "indexing")
            # Cancel arrives AFTER the last tick, BEFORE the success patch.
            # Only set the flag — don't patch status (the race window
            # _run_ingest_task's final patch must guard).
            service._cancel_flags[rec.task_id].set()
            return 1, "text_coll"

        def upsert_image(self, *, progress_cb=None, force_recreate=False):
            return 1, "img_coll"

    with (
        patch.object(svc_mod, "_run_parse_task", fake_parse),
        patch.object(svc_mod, "get_backend", lambda name: FakeBackend()),
    ):
        _run_ingest_task(service, rec, parse_options)

    # The success patch must not have overwritten CANCELLED with done.
    assert rec.status == "cancelled", f"expected cancelled, got {rec.status!r}"


def test_index_cancel_patch_handles_terminal_status_race(tmp_home: Path) -> None:
    """``_run_ingest_task``'s ``except _TaskCancelled`` must patch status=CANCELLED
    itself even when ``cancel_task`` saw a terminal status and skipped its own
    patch.

    The race: ``_run_parse_task`` leaves status=DONE at the parse→index seam;
    ``cancel_task`` arriving then sees a terminal status and only sets the flag
    (its ``_patch(status=CANCELLED)`` is gated on ``status not in terminal()``).
    ``_run_ingest_task`` then flips status back to ``running`` and drives the
    index loop, whose ``_progress`` tick sees the flag and raises
    ``_TaskCancelled``. Without an explicit patch in the except branch, the
    task stays ``running`` until ``load_history`` rewrites it to
    ``interrupted`` on next boot — losing the cancel intent."""
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import _run_ingest_task

    service = IngestService()
    # status=DONE as _run_parse_task leaves it at the parse→index seam.
    rec = TaskRecord(
        task_id="idxcancel03",
        kind="ingest",
        status="done",
        total=1,
        finished_at=None,
    )
    service._tasks[rec.task_id] = rec
    service._cancel_flags[rec.task_id] = __import__("threading").Event()
    parse_options = ParseOptions(assets=[])

    def fake_parse(_svc, _rec, _opts):
        # Leaves rec.status == "done" (terminal) — mirrors _run_parse_task.
        _svc._patch(_rec, status="done", current="parse done")

    class FakeBackend:
        def upsert_text(self, *, progress_cb=None, force_recreate=False):
            # cancel_task arrives while status is briefly "done" (terminal) →
            # it sets the flag but, per its terminal guard, does NOT patch
            # status. The next _progress tick raises _TaskCancelled.
            service._cancel_flags[rec.task_id].set()
            if progress_cb:
                progress_cb(1, 10, "indexing")  # raises _TaskCancelled
            return 1, "text_coll"

        def upsert_image(self, *, progress_cb=None, force_recreate=False):
            return 1, "img_coll"

    with (
        patch.object(svc_mod, "_run_parse_task", fake_parse),
        patch.object(svc_mod, "get_backend", lambda name: FakeBackend()),
    ):
        _run_ingest_task(service, rec, parse_options)

    # The except _TaskCancelled branch must have patched CANCELLED itself.
    assert rec.status == "cancelled", f"expected cancelled, got {rec.status!r}"
    assert rec.finished_at is not None


def test_ingest_cancel_at_parse_index_seam_patches_cancelled(tmp_home: Path) -> None:
    """Cancel arriving in the parse→index seam window (status briefly DONE,
    cancel_task sees a terminal status and only sets the flag) must be caught
    by the seam guard — the task is patched CANCELLED and the index loop is
    *not* entered. Regression for the parse→index seam race.

    Before the fix the seam guard only checked ``rec.status == CANCELLED``;
    a cancel that arrived while status was briefly DONE slipped past it,
    flipped the task back to ``running``, and the index loop ran (or the CLI
    polled the DONE window and exited before index ran at all)."""
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import ParseOptions, TaskRecord, _run_ingest_task

    service = IngestService()
    rec = TaskRecord(task_id="seamcancel01", kind="ingest", total=1)
    service._tasks[rec.task_id] = rec
    # status=done as _run_parse_task leaves it; cancel_task sees terminal.
    service._patch(rec, status="done", finished_at=time.time())
    service._cancel_flags[rec.task_id] = __import__("threading").Event()
    service._cancel_flags[rec.task_id].set()
    parse_options = ParseOptions(assets=[])

    upsert_called = {"text": 0, "image": 0}

    def fake_parse(_svc, _rec, _opts):
        _svc._patch(_rec, status="done", current="parse done")

    class FakeBackend:
        def upsert_text(self, *, progress_cb=None, force_recreate=False):
            upsert_called["text"] += 1
            return 0, "text_coll"

        def upsert_image(self, *, progress_cb=None, force_recreate=False):
            upsert_called["image"] += 1
            return 0, "img_coll"

    with (
        patch.object(svc_mod, "_run_parse_task", fake_parse),
        patch.object(svc_mod, "get_backend", lambda name: FakeBackend()),
    ):
        _run_ingest_task(service, rec, parse_options)

    assert rec.status == "cancelled", f"expected cancelled, got {rec.status!r}"
    assert rec.finished_at is not None
    # The index loop must not have run — cancel was caught at the seam.
    assert upsert_called == {"text": 0, "image": 0}
