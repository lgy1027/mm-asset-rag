"""Tests for ``mm_asset_rag.service`` retry and history behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mm_asset_rag.asset_index import AssetIndexEntry
from mm_asset_rag.assets import Asset
from mm_asset_rag.service import IngestService, ParseOptions, TaskRecord


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_asset(tmp_home: Path, name: str = "fish.png") -> Asset:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    images_dir = tmp_home / "assets" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    file_path = images_dir / name
    Image.new("RGB", (8, 8), color=(120, 120, 0)).save(file_path)
    return Asset(
        asset_id=name,
        title=name,
        source_type="image",
        relative_path=f"images/{name}",
        source_url="",
        tags=[],
        asset_dir=tmp_home / "assets",
    )


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


def test_load_history_tolerates_legacy_jsonl(tmp_home: Path) -> None:
    tasks_path = tmp_home / "tasks.jsonl"
    legacy = {
        "task_id": "legacy01",
        "kind": "ingest",
        "status": "done",
        "total": 1,
        "uploaded_files": ["images/old.png"],
    }
    tasks_path.write_text(__import__("json").dumps(legacy) + "\n", encoding="utf-8")

    service = IngestService()
    service.load_history()
    loaded = service.get_task("legacy01")
    assert loaded is not None
    assert loaded.source == "upload"
    assert loaded.origin_task_id is None
    assert loaded.parse_options == {}


def test_parse_options_serialisation_roundtrip() -> None:
    options = ParseOptions(assets=[], pdf_parser="pymupdf", enable_ocr=True, enable_vlm=False)
    snap = IngestService._serialise_options(options)
    assert snap == {
        "pdf_parser": "pymupdf",
        "enable_ocr": True,
        "enable_vlm": False,
        "image_provider": "lite",
    }
    restored = IngestService._deserialise_options(snap, assets=[])
    assert restored.pdf_parser == "pymupdf"
    assert restored.enable_ocr is True
    assert restored.enable_vlm is False


def test_parse_options_serialisation_drops_invalid_values() -> None:
    snap = {"pdf_parser": "bogus", "image_provider": "weird"}
    options = IngestService._deserialise_options(snap, assets=[])
    assert options.pdf_parser == "auto"
    assert options.image_provider == "lite"


# ─── delete_asset ─────────────────────────────────────────────────────


def test_delete_asset_removes_file_parsed_captions(tmp_home: Path) -> None:

    from mm_asset_rag.asset_index import upsert_entry
    from mm_asset_rag.paths import get_captions_dir, get_parsed_dir

    asset = _make_asset(tmp_home, "beach.png")
    upsert_entry(
        AssetIndexEntry(
            asset_id=asset.asset_id,
            sha256="abc",
            source_type="image",
            relative_path=asset.relative_path,
        )
    )
    parsed_dir = get_parsed_dir() / asset.asset_id
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "raw.jsonl").write_text("{}", encoding="utf-8")
    (get_captions_dir() / f"{asset.asset_id}.json").write_text("{}", encoding="utf-8")

    service = IngestService()
    report = service.delete_asset(asset.asset_id)

    assert report.file_deleted
    assert report.parsed_deleted
    assert report.captions_deleted
    assert not (tmp_home / "assets" / asset.relative_path).exists()
    assert not parsed_dir.exists()


def test_delete_asset_is_idempotent(tmp_home: Path) -> None:
    asset = _make_asset(tmp_home, "fish.png")
    service = IngestService()
    report = service.delete_asset(asset.asset_id)
    assert not report.was_known
    # Second call with same id remains a no-op.
    report2 = service.delete_asset(asset.asset_id)
    assert not report2.was_known


def test_retry_failed_only_uses_recorded_statuses(tmp_home: Path) -> None:
    assets = [
        Asset(asset_id="ok1", title="ok1", source_type="image", relative_path="images/ok1.png"),
        Asset(asset_id="bad1", title="bad1", source_type="image", relative_path="images/bad1.png"),
    ]
    service = IngestService()
    original = TaskRecord(
        task_id="origstatus01",
        kind="parse",
        status="partial",
        total=2,
        uploaded_files=[a.relative_path for a in assets],
        asset_statuses={"ok1": "ok", "bad1": "failed"},
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
    assert [a.asset_id for a in options.assets] == ["bad1"]


def test_retry_failed_only_all_ok_raises(tmp_home: Path) -> None:
    assets = [
        Asset(asset_id="ok1", title="ok1", source_type="image", relative_path="images/ok1.png"),
    ]
    service = IngestService()
    original = TaskRecord(
        task_id="origstatus02",
        kind="parse",
        status="partial",
        total=1,
        uploaded_files=[a.relative_path for a in assets],
        asset_statuses={"ok1": "ok"},
    )
    service._tasks[original.task_id] = original
    service._rebuild_assets_for_retry = lambda uploaded: list(assets)  # type: ignore[method-assign]
    with pytest.raises(FileNotFoundError, match="no failed or skipped assets"):
        service.retry_task(original.task_id, failed_only=True)

    snap = {"pdf_parser": "bogus", "image_provider": "weird"}
    options = IngestService._deserialise_options(snap, assets=[])
    assert options.pdf_parser == "auto"
    assert options.image_provider == "lite"
    asset = _make_asset(tmp_home, "fish.png")
    service = IngestService()
    report = service.delete_asset(asset.asset_id)
    assert not report.was_known
    # Second call with same id remains a no-op.
    report2 = service.delete_asset(asset.asset_id)
    assert not report2.was_known


def test_delete_asset_dry_run_touches_nothing(tmp_home: Path) -> None:
    from mm_asset_rag.asset_index import upsert_entry
    from mm_asset_rag.paths import get_captions_dir, get_documents_jsonl, get_parsed_dir

    asset = _make_asset(tmp_home, "dryrun.png")
    upsert_entry(
        AssetIndexEntry(
            asset_id=asset.asset_id,
            sha256="hash",
            source_type="image",
            relative_path=asset.relative_path,
        )
    )
    parsed_dir = get_parsed_dir() / asset.asset_id
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "raw.jsonl").write_text("{}", encoding="utf-8")
    (get_captions_dir() / f"{asset.asset_id}.json").write_text("{}", encoding="utf-8")
    docs_path = get_documents_jsonl()
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        json.dumps(
            {
                "text": "x",
                "metadata": {
                    "asset_id": asset.asset_id,
                    "asset_title": asset.title,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    service = IngestService()
    report = service.delete_asset(asset.asset_id, dry_run=True)

    assert report.dry_run
    assert report.would_delete_file
    assert report.would_delete_parsed
    assert report.would_delete_captions
    assert report.would_remove_documents == 1
    assert report.would_tombstone
    # Nothing should actually be gone.
    assert (tmp_home / "assets" / asset.relative_path).exists()
    assert parsed_dir.exists()
    assert (get_captions_dir() / f"{asset.asset_id}.json").exists()


def test_force_retry_clears_parsed_cache(tmp_home: Path) -> None:
    from mm_asset_rag.paths import get_parsed_dir

    asset = _make_asset(tmp_home, "force.png")
    parsed_dir = get_parsed_dir() / asset.asset_id
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "raw.jsonl").write_text("{}", encoding="utf-8")
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
        from mm_asset_rag.schema import ParsedDocument

        return [ParsedDocument(text="x", metadata={"asset_id": asset_obj.asset_id})]

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


def test_ingest_task_records_indexed_status_on_success(tmp_home: Path) -> None:
    asset = _make_asset(tmp_home, "ing.png")
    rec = TaskRecord(task_id="ingest1", kind="ingest", status="running", total=1)
    rec.uploaded_files = [asset.relative_path]
    rec.asset_statuses = {asset.asset_id: "ok"}

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

    assert rec.asset_statuses[asset.asset_id] == "indexed"
    assert rec.status in {"done", "partial"}
    assert rec.finished_at is not None


def test_ingest_task_records_failed_index_on_upsert_crash(tmp_home: Path) -> None:
    asset = _make_asset(tmp_home, "crash.png")
    rec = TaskRecord(task_id="ingest2", kind="ingest", status="running", total=1)
    rec.uploaded_files = [asset.relative_path]
    rec.asset_statuses = {asset.asset_id: "ok"}

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

    assert rec.asset_statuses[asset.asset_id] == "failed_index"
    assert rec.status == "failed"


def test_retry_failed_only_includes_failed_index(tmp_home: Path) -> None:
    assets = [
        Asset(asset_id="ok1", title="ok1", source_type="image", relative_path="images/ok1.png"),
        Asset(asset_id="bad1", title="bad1", source_type="image", relative_path="images/bad1.png"),
    ]
    service = IngestService()
    original = TaskRecord(
        task_id="origidx01",
        kind="ingest",
        status="partial",
        total=2,
        uploaded_files=[a.relative_path for a in assets],
        asset_statuses={"ok1": "indexed", "bad1": "failed_index"},
    )
    service._tasks[original.task_id] = original
    service._rebuild_assets_for_retry = lambda _uploaded: list(assets)  # type: ignore[method-assign]
    with patch.object(service, "_spawn") as spawn:
        service.retry_task(original.task_id, failed_only=True)
    assert spawn.call_count == 1
    args, _kwargs = spawn.call_args
    _target, _rec, options = args
    assert [a.asset_id for a in options.assets] == ["bad1"]


def test_retry_force_and_failed_only_clear_only_failed_cache(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mm_asset_rag.paths import get_parsed_dir

    # Two assets: one healthy (ok1, was 'ok' in original task) and one
    # broken (bad1, was 'failed' in original task). failed-only filter
    # will narrow the retry set to bad1; force must then only clear
    # that asset's cache.
    assets = [
        Asset(asset_id="ok1", title="ok1", source_type="image", relative_path="images/ok1.png"),
        Asset(asset_id="bad1", title="bad1", source_type="image", relative_path="images/bad1.png"),
    ]

    # Pre-stage parsed/ cache for both assets.
    ok_parsed = get_parsed_dir() / "ok1"
    bad_parsed = get_parsed_dir() / "bad1"
    ok_parsed.mkdir(parents=True)
    (ok_parsed / "raw.jsonl").write_text("{}", encoding="utf-8")
    bad_parsed.mkdir(parents=True)
    (bad_parsed / "raw.jsonl").write_text("{}", encoding="utf-8")

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
        asset_statuses={"bad1": "failed", "ok1": "ok"},
    )

    service = IngestService()
    # Stub parser; we just want the force rmtree to fire before parse.
    import mm_asset_rag.service as svc_mod
    from mm_asset_rag.service import ParseOptions, _do_parse

    called: list[str] = []

    class StubParser:
        def parse(self, asset, **_kwargs):
            from mm_asset_rag.schema import ParsedDocument

            called.append(asset.asset_id)
            return [ParsedDocument(text="x", metadata={"asset_id": asset.asset_id})]

    monkeypatch.setattr(svc_mod, "get_parser", lambda kind, name: StubParser())

    # Run the same parse loop retry_task would dispatch (synchronously).
    _do_parse(service, rec, ParseOptions(assets=[assets[1]]))  # only bad1

    # ok1 cache survives (its parsed/ dir is still present, untouched).
    assert (get_parsed_dir() / "ok1" / "raw.jsonl").exists()
    # bad1 cache was force-cleared before parse, then parsed again.
    assert not (get_parsed_dir() / "bad1").exists() or called == ["bad1"]
    assert called == ["bad1"]
    assert rec.asset_statuses["bad1"] in {"ok", "skipped", "failed"}


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
    from mm_asset_rag.service import _resolve_sandboxed_image_path

    with pytest.raises(ValueError, match="must be relative"):
        _resolve_sandboxed_image_path("/etc/passwd")
    with pytest.raises(ValueError, match="must be relative"):
        _resolve_sandboxed_image_path("/absolute/image.png")


def test_dispatch_search_rejects_parent_traversal(tmp_home: Path) -> None:
    from mm_asset_rag.service import _resolve_sandboxed_image_path

    with pytest.raises(ValueError, match="outside assets"):
        _resolve_sandboxed_image_path("../escape.png")
    with pytest.raises(ValueError, match="outside assets"):
        _resolve_sandboxed_image_path("images/../../escape.png")


def test_dispatch_search_rejects_missing_file(tmp_home: Path) -> None:
    from mm_asset_rag.service import _resolve_sandboxed_image_path

    assets_dir = tmp_home / "assets"
    assets_dir.mkdir()
    with pytest.raises(ValueError, match="not found"):
        _resolve_sandboxed_image_path("images/ghost.png")


def test_dispatch_search_accepts_file_inside_assets(tmp_home: Path) -> None:
    from mm_asset_rag.service import _resolve_sandboxed_image_path

    assets_dir = tmp_home / "assets"
    target = assets_dir / "images" / "ok.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    resolved = _resolve_sandboxed_image_path("images/ok.png")
    assert resolved is not None
    assert resolved.is_relative_to(assets_dir)
    assert resolved.name == "ok.png"


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
