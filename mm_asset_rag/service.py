"""Background-task service for ingest / reindex / task history.

This module centralizes everything the FastAPI app and the ``mmrag`` CLI
both need:

* Spawning background ``threading.Thread`` workers that run parse + index
* Persisting task state to ``$MM_ASSET_RAG_HOME/tasks.jsonl`` so it survives
  process restarts
* Surfacing :class:`TaskRecord` snapshots to ``GET /tasks`` and
  ``GET /tasks/{id}``
* Dispatching the explicit ``reindex`` command

Before this module existed, ``api.py`` and ``cli.py`` each implemented
their own version of the same loop (``_run_parse_task`` vs
``command_parse``). That duplication made it impossible to change the
parse pipeline without editing two files — exactly the kind of coupling
that the parser/embedder/backend registries already eliminated for the
hot path.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from . import asset_index
from . import parsers as _parsers  # noqa: F401  # register built-in parsers
from .asset_index import AssetIndexEntry
from .assets import Asset, from_sniffed
from .backends.qdrant_backend import delete_points_by_asset_id
from .config import load_env
from .paths import (
    get_assets_dir,
    get_captions_dir,
    get_data_dir,
    get_documents_jsonl,
    get_parsed_dir,
)
from .registry import get_backend, get_parser
from .retrieval import hybrid_search
from .settings import Settings, get_settings
from .sniff import sniff

# ─── Helpers shared by api.py and cli.py ──────────────────────────────────


def coerce_bool(form_val: str | bool | None, default: bool) -> bool:
    """Coerce a multipart boolean field to ``bool``.

    FastAPI's ``bool = Form(...)`` parsing turns the string ``"true"`` /
    ``"false"`` into ``True`` / ``False`` automatically. This helper
    handles both that case and the case where the form value comes in as
    a raw string (e.g. when declared as ``str | None = Form(default=None)``).
    Returns ``default`` when ``form_val`` is ``None`` or empty.
    """
    if form_val is None or form_val == "":
        return default
    if isinstance(form_val, bool):
        return form_val
    return str(form_val).strip().lower() in {"1", "true", "yes", "y", "on"}


IMAGE_PATH_USES = {"image-to-image", "hybrid"}


def _resolve_sandboxed_image_path(image_path: str | Path | None) -> Path | None:
    """Resolve ``image_path`` to an absolute path strictly inside ``assets_dir``.

    ``image-to-image`` / ``hybrid`` search delegates to the CLIP encoder
    which feeds the file through ``PIL.Image.open``. PIL's decoders have
    a long history of format-parsing RCEs (PSD, GIF, SGI, …) and the
    ``open()`` call also happily reads any local file the process can
    see — including ``/etc/passwd`` if the path is absolute. We refuse
    anything that resolves outside ``assets_dir`` and any symlink target
    that escapes it (``resolve(strict=False)`` then ``is_relative_to``
    catches the symlink case). Returning ``None`` when ``image_path`` is
    empty preserves the existing "no image" semantics for hybrid.
    """
    if not image_path:
        return None
    assets_dir = get_assets_dir().resolve()
    raw = Path(image_path)
    # Reject absolute paths up front — the user-visible API is
    # relative-path-only and any ``/etc/passwd``-style attempt should
    # fail *before* we hit the filesystem.
    if raw.is_absolute():
        raise ValueError("image_path must be relative to assets/")
    try:
        resolved = (assets_dir / raw).resolve()
    except OSError as exc:
        raise ValueError(f"image_path cannot be resolved: {exc}") from exc
    if not resolved.is_relative_to(assets_dir):
        raise ValueError("image_path resolves outside assets/")
    if not resolved.is_file():
        raise ValueError(f"image_path not found or not a regular file: {raw}")
    return resolved


def dispatch_search(
    *,
    query: str,
    mode: str,
    image_path: str | Path | None,
    top_k: int,
) -> list:
    """Dispatch a search request to the right backend call.

    Single source of truth for ``mode`` routing — used by ``/search``,
    ``/chat`` (one-call helper) and the ``mmrag search`` CLI so they all
    handle the four modes (``text``, ``text-to-image``, ``image-to-image``,
    ``hybrid``) the same way. When ``mode`` consumes ``image_path`` the
    path is sandboxed to ``assets_dir`` so the CLIP encoder cannot be
    steered at an arbitrary local file.
    """
    backend = get_backend("qdrant")
    sandboxed_image = _resolve_sandboxed_image_path(image_path) if mode in IMAGE_PATH_USES else None
    if mode == "text":
        return backend.search_text(query=query, top_k=top_k)
    if mode == "text-to-image":
        return backend.search_text_to_image(query=query, top_k=top_k)
    if mode == "image-to-image":
        if sandboxed_image is None:
            raise ValueError("image_path required for image-to-image")
        return backend.search_image(image_path=sandboxed_image, top_k=top_k)
    # hybrid (default)
    return hybrid_search(query, image_path=sandboxed_image, top_k=top_k)


# ─── Enums ──────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """Lifecycle states for an :class:`IngestService` background task.

    Values are kept as plain strings so :func:`json.dumps` round-trips
    them without a custom encoder and the JSONL task history stays
    human-readable. ``str`` mixin means ``TaskStatus.DONE == "done"``
    and ``"done" in {TaskStatus.DONE, TaskStatus.PARTIAL}`` both work.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    @classmethod
    def terminal(cls) -> set[TaskStatus]:
        return {cls.DONE, cls.PARTIAL, cls.FAILED, cls.INTERRUPTED}


class AssetStatus(str, Enum):
    """Per-asset progress in a task's ``asset_statuses`` map.

    Parse stage: ``OK`` / ``SKIPPED`` / ``FAILED``.
    Index stage: ``INDEXED`` / ``FAILED_INDEX``.
    """

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"
    INDEXED = "indexed"
    FAILED_INDEX = "failed_index"

    @classmethod
    def retry_eligible(cls) -> set[AssetStatus | None]:
        """The status values ``retry_task(failed_only=True)`` matches.

        ``None`` (an asset that was added to the task after the per-asset
        status map was introduced) is treated the same as ``FAILED`` to
        keep the retry path honest about uncertainty.
        """
        return {cls.FAILED, cls.SKIPPED, cls.FAILED_INDEX, None}


# ─── Data types ─────────────────────────────────────────────────────────


@dataclass
class TaskRecord:
    task_id: str
    kind: str  # "parse" or "ingest"
    status: str = "pending"  # pending | running | done | partial | failed | interrupted
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    current: str = ""
    error: str | None = None
    uploaded_files: list[str] = field(default_factory=list)
    parse_options: dict[str, object] = field(default_factory=dict)
    source: str = "upload"
    origin_task_id: str | None = None
    force: bool = False
    failed_only: bool = False
    asset_statuses: dict[str, str] = field(default_factory=dict)


@dataclass
class ParseOptions:
    """Per-task parse configuration for uploaded/auto-sniffed assets."""

    assets: list[Asset] = field(default_factory=list)
    pdf_parser: str = "auto"
    enable_ocr: bool = False
    enable_vlm: bool = False
    image_provider: str = "lite"


@dataclass
class DeleteAssetReport:
    """Per-asset cleanup outcome returned by ``IngestService.delete_asset``.

    All counts default to zero; ``errors`` collects human-readable
    descriptions of any cleanup step that failed. The report is meant to
    be JSON-serialised for the API and CLI. ``would_*`` flags are only
    meaningful when ``dry_run=True``.
    """

    asset_id: str
    file_deleted: bool = False
    parsed_deleted: bool = False
    captions_deleted: bool = False
    documents_removed: int = 0
    text_collections_scanned: int = 0
    image_collections_scanned: int = 0
    errors: list[str] = field(default_factory=list)
    was_known: bool = True
    dry_run: bool = False
    would_delete_file: bool = False
    would_delete_parsed: bool = False
    would_delete_captions: bool = False
    would_remove_documents: int = 0
    would_tombstone: bool = False
    qdrant_note: str = ""


# ─── Task bookkeeping ────────────────────────────────────────────────────


class IngestService:
    """Stateful ingest + index + task-history service.

    A single instance is constructed per process and shared between the
    FastAPI app and (in the future) the CLI. The module-level
    :func:`get_service` returns the same instance for convenience.
    """

    _TASKS_LOCK = threading.Lock()
    _PERSIST_LOCK = threading.Lock()

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._tasks: dict[str, TaskRecord] = {}
        self._stream_events: dict[str, list[tuple[threading.Event, dict[str, object]]]] = {}

    # ─── Public API used by both FastAPI and CLI ─────────────────────────

    def parse_assets(self, assets: list[Asset], options: ParseOptions | None = None) -> TaskRecord:
        """Parse explicitly provided assets.

        This is the only parse entry in the upload-first architecture: assets
        are constructed by ``UploadPipeline.confirm`` from sniff + VLM + user
        edits, not loaded from a manifest. An empty ``assets`` list still
        creates a ``done`` task record so the UI history reflects the no-op
        intent, but skips spawning a worker thread for it.
        """
        options = options or ParseOptions()
        options.assets = list(assets)
        uploaded = [a.relative_path for a in assets]
        rec = self._new_task(
            kind="parse",
            total=len(assets),
            uploaded=uploaded,
            parse_options=self._serialise_options(options),
        )
        if not assets:
            self._patch(
                rec,
                status=TaskStatus.DONE,
                processed=0,
                current="no assets to parse",
                finished_at=time.time(),
            )
            return rec
        self._spawn(_run_parse_task, rec, options)
        return rec

    def ingest_assets(self, assets: list[Asset], options: ParseOptions | None = None) -> TaskRecord:
        """Parse + index explicitly provided assets."""
        options = options or ParseOptions()
        options.assets = list(assets)
        uploaded = [a.relative_path for a in assets]
        rec = self._new_task(
            kind="ingest",
            total=len(assets),
            uploaded=uploaded,
            parse_options=self._serialise_options(options),
        )
        self._spawn(_run_ingest_task, rec, options)
        return rec

    def retry_task(
        self,
        task_id: str,
        *,
        force: bool = False,
        failed_only: bool = False,
    ) -> TaskRecord:
        """Re-run a previously failed/partial/interrupted task.

        Reconstructs ``Asset`` objects from the original task's
        ``uploaded_files`` (best-effort via re-sniff), and spawns a new
        background task that mirrors the original ``kind`` and
        ``parse_options``. The new task is recorded with
        ``source="retry"`` and ``origin_task_id`` pointing back to the
        original.

        ``force=True`` clears cached ``parsed/<id>/raw.jsonl`` before
        the retry so the parse loop re-reads from disk. ``failed_only=True``
        narrows the retry set to assets whose status is missing,
        ``failed`` or ``skipped``. The two flags compose: with both set,
        only the failed assets are re-parsed and only their caches are
        cleared — useful after upgrading the parser without touching
        already-indexed assets. Legacy tasks without ``asset_statuses``
        fall back to running every uploaded asset and emit a warning.
        """
        with self._TASKS_LOCK:
            original = self._tasks.get(task_id)
        if original is None:
            raise KeyError(f"unknown task {task_id}")
        if original.status not in {TaskStatus.FAILED, TaskStatus.PARTIAL, TaskStatus.INTERRUPTED}:
            raise ValueError(f"task {task_id} cannot be retried (status={original.status})")
        if failed_only and not original.asset_statuses:
            print(
                f"[retry] task {task_id} has no per-asset statuses; treating failed_only as force"
            )
            force = True
            failed_only = False
        assets = self._rebuild_assets_for_retry(original.uploaded_files)
        if not assets:
            raise FileNotFoundError(f"no assets available to retry for task {task_id}")
        if failed_only and original.asset_statuses:
            assets = [
                a
                for a in assets
                if original.asset_statuses.get(a.asset_id) in AssetStatus.retry_eligible()
            ]
            if not assets:
                raise FileNotFoundError(f"no failed or skipped assets to retry for task {task_id}")
        options = self._deserialise_options(original.parse_options, assets)
        uploaded = [a.relative_path for a in assets]
        preserved_statuses = dict(original.asset_statuses) if original.asset_statuses else {}
        if original.kind == "parse":
            rec = self._new_task(
                kind="parse",
                total=len(assets),
                uploaded=uploaded,
                parse_options=self._serialise_options(options),
                source="retry",
                origin_task_id=task_id,
                force=force,
                failed_only=failed_only,
            )
            rec.asset_statuses = preserved_statuses
            self._patch(rec)
            self._spawn(_run_parse_task, rec, options)
        elif original.kind == "ingest":
            rec = self._new_task(
                kind="ingest",
                total=len(assets),
                uploaded=uploaded,
                parse_options=self._serialise_options(options),
                source="retry",
                origin_task_id=task_id,
                force=force,
                failed_only=failed_only,
            )
            rec.asset_statuses = preserved_statuses
            self._patch(rec)
            self._spawn(_run_ingest_task, rec, options)
        else:
            raise ValueError(f"unknown task kind for retry: {original.kind!r}")
        return rec

    def reindex(self, text_only: bool = False, image_only: bool = False) -> tuple[str, ...]:
        """Force-recreate Qdrant collections and re-upsert from documents.jsonl."""
        backend = get_backend("qdrant")
        results = []
        if not image_only:
            _n, name = backend.upsert_text(force_recreate=True)
            results.append(f"text: {name}")
        if not text_only:
            _ni, ni_name = backend.upsert_image(force_recreate=True)
            results.append(f"image: {ni_name}")
        return tuple(results)

    def list_tasks(self) -> list[TaskRecord]:
        with self._TASKS_LOCK:
            return list(self._tasks.values())

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._TASKS_LOCK:
            return self._tasks.get(task_id)

    def load_history(self) -> None:
        """Restore tasks from ``tasks.jsonl``. Tasks still in ``running``
        state when the previous process exited are marked ``interrupted``.
        """
        path = self._tasks_log_path()
        if not path.exists():
            return
        latest: dict[str, dict[str, object]] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = obj.get("task_id")
                if isinstance(task_id, str) and task_id:
                    latest[task_id] = obj

        interrupted = 0
        with self._TASKS_LOCK:
            for task_id, obj in latest.items():
                if obj.get("finished_at") is None and obj.get("status") == "running":
                    obj["status"] = "interrupted"
                    obj["current"] = (
                        f"interrupted (previous process exited): {obj.get('current', '')}"
                    ).strip(": ")
                    obj["finished_at"] = time.time()
                    obj["error"] = obj.get("error") or "process exited before task completed"
                    interrupted += 1
                    self._persist(self._task_from_dict(obj))
                self._tasks[task_id] = self._task_from_dict(obj)
        if latest:
            print(
                f"[tasks] loaded {len(latest)} task(s) from disk; {interrupted} marked interrupted"
            )

    @staticmethod
    def _task_from_dict(obj: dict[str, object]) -> TaskRecord:
        """Build a ``TaskRecord`` from a JSONL row, tolerating legacy records."""
        kwargs: dict[str, object] = {}
        for field_name in TaskRecord.__dataclass_fields__:
            if field_name in obj:
                kwargs[field_name] = obj[field_name]
        return TaskRecord(**kwargs)  # type: ignore[arg-type]

    def list_assets(self) -> list[AssetIndexEntry]:
        """Return the non-deleted rows from the asset index, newest first."""
        return asset_index.list_active()

    def get_asset_detail(self, asset_id: str) -> dict[str, object] | None:
        """Return a read-only detail snapshot for ``asset_id``.

        Combines the asset_index row with on-disk existence checks
        (file, parsed/, captions/) so the web drawer can show whether
        each derived artefact still exists. Returns ``None`` when the
        asset is unknown or its relative_path is unsafe.
        """
        entry = asset_index.find_active_by_asset_id(asset_id)
        if entry is None:
            return None
        relative_path = Path(entry.relative_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) < 1
        ):
            return None

        assets_dir = get_assets_dir().resolve()
        try:
            file_path = (assets_dir / relative_path).resolve()
        except OSError:
            return None
        file_exists = file_path.is_file() and file_path.is_relative_to(assets_dir)

        parsed_dir = get_parsed_dir() / asset_id
        parsed_raw = parsed_dir / "raw.jsonl"
        captions_path = get_captions_dir() / f"{asset_id}.json"

        try:
            parsed_size = parsed_raw.stat().st_size if parsed_raw.exists() else 0
        except OSError:
            parsed_size = 0

        return {
            "asset_id": entry.asset_id,
            "sha256": entry.sha256,
            "source_type": entry.source_type,
            "relative_path": entry.relative_path,
            "title": entry.asset_title,
            "ingested_at": entry.ingested_at,
            "last_task_id": entry.last_task_id,
            "tags": list(entry.tags),
            "file_exists": file_exists,
            "file_size": file_path.stat().st_size if file_exists else 0,
            "parsed_exists": parsed_raw.exists(),
            "parsed_size": parsed_size,
            "parsed_dir": str(parsed_dir.relative_to(get_data_dir())),
            "captions_exists": captions_path.exists(),
            "captions_path": str(captions_path.relative_to(get_data_dir())),
        }

    def delete_asset(self, asset_id: str, *, dry_run: bool = False) -> DeleteAssetReport:
        """Best-effort cleanup of every trace of ``asset_id``.

        The function is idempotent: missing pieces are reported as
        ``False``/``0`` rather than raising. The asset_index is only
        tombstoned once per ``asset_id``; subsequent calls return a
        ``was_known=False`` report so the API can choose to 404.

        ``dry_run=True`` resolves every target but performs no writes:
        file/parsed/captions are not removed, ``documents.jsonl`` is not
        rewritten (only counted), Qdrant is not contacted, and the asset
        index is not tombstoned. The Qdrant row counts in
        ``text_collections_scanned`` / ``image_collections_scanned``
        are reported as zero in dry-run with a note, because the
        server cannot pre-flight point counts cheaply.
        """
        report = DeleteAssetReport(asset_id=asset_id, dry_run=dry_run)
        if not asset_id:
            report.was_known = False
            report.errors.append("empty asset_id")
            return report

        index_entry = asset_index.find_active_by_asset_id(asset_id)
        if index_entry is None:
            report.was_known = False
            return report

        assets_dir = get_assets_dir().resolve()
        relative_path = Path(index_entry.relative_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) < 1
        ):
            report.errors.append(f"unsafe relative_path in asset_index: {relative_path}")
            return report

        # 1. file on disk
        try:
            file_path = (assets_dir / relative_path).resolve()
            if file_path.is_relative_to(assets_dir) and file_path.is_file():
                if dry_run:
                    report.would_delete_file = True
                else:
                    file_path.unlink()
                    report.file_deleted = True
        except OSError as exc:
            report.errors.append(f"file delete failed: {exc}")

        # 2. parsed/<asset_id>/
        try:
            parsed_dir = get_parsed_dir() / asset_id
            if parsed_dir.exists():
                if dry_run:
                    report.would_delete_parsed = True
                else:
                    shutil.rmtree(parsed_dir, ignore_errors=True)
                    report.parsed_deleted = True
        except OSError as exc:
            report.errors.append(f"parsed delete failed: {exc}")

        # 3. captions/<asset_id>.json
        try:
            captions_path = get_captions_dir() / f"{asset_id}.json"
            if captions_path.exists():
                if dry_run:
                    report.would_delete_captions = True
                else:
                    captions_path.unlink()
                    report.captions_deleted = True
        except OSError as exc:
            report.errors.append(f"captions delete failed: {exc}")

        # 4. documents.jsonl rewrite (filter rows whose metadata.asset_id matches)
        try:
            docs_path = get_documents_jsonl()
            if docs_path.exists():
                removed = 0
                with docs_path.open("r", encoding="utf-8") as src:
                    for line in src:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            obj = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        meta = obj.get("metadata") if isinstance(obj, dict) else None
                        if isinstance(meta, dict) and str(meta.get("asset_id", "")) == asset_id:
                            removed += 1
                if dry_run:
                    report.would_remove_documents = removed
                else:
                    tmp_path = docs_path.with_suffix(docs_path.suffix + ".tmp")
                    kept = 0
                    with (
                        docs_path.open("r", encoding="utf-8") as src,
                        tmp_path.open("w", encoding="utf-8") as dst,
                    ):
                        for line in src:
                            stripped = line.strip()
                            if not stripped:
                                dst.write(line)
                                continue
                            try:
                                obj = json.loads(stripped)
                            except json.JSONDecodeError:
                                dst.write(line)
                                continue
                            meta = obj.get("metadata") if isinstance(obj, dict) else None
                            if isinstance(meta, dict) and str(meta.get("asset_id", "")) == asset_id:
                                continue
                            dst.write(line)
                            kept += 1
                    os.replace(tmp_path, docs_path)
                    report.documents_removed = removed
        except OSError as exc:
            report.errors.append(f"documents.jsonl rewrite failed: {exc}")

        # 5. Qdrant text + image collections
        if dry_run:
            report.qdrant_note = "would scan text+image collections (point counts unavailable)"
        else:
            try:
                counts = delete_points_by_asset_id(asset_id)
                report.text_collections_scanned = counts.get("text", 0)
                report.image_collections_scanned = counts.get("image", 0)
            except Exception as exc:
                report.errors.append(f"qdrant delete failed: {exc}")

        # 6. asset_index tombstone — only if the destructive steps above
        # all succeeded. Otherwise we would leave "Qdrant still has the
        # point but the index says it's gone", and the leftover point
        # is unreachable for any future cleanup. The caller can still
        # inspect ``report.errors`` to decide whether to retry the
        # tombstone separately.
        if dry_run:
            report.would_tombstone = True
        elif not report.errors:
            try:
                asset_index.mark_deleted(asset_id)
            except OSError as exc:
                report.errors.append(f"asset_index mark_deleted failed: {exc}")
        else:
            report.errors.append(
                "skipping tombstone: destructive steps reported errors; "
                "fix and re-run delete_asset to retry."
            )

        return report

    # ─── Internals ─────────────────────────────────────────────────────

    def _new_task(
        self,
        kind: str,
        total: int,
        uploaded: list[str] | None = None,
        parse_options: dict[str, object] | None = None,
        source: str = "upload",
        origin_task_id: str | None = None,
        force: bool = False,
        failed_only: bool = False,
    ) -> TaskRecord:
        rec = TaskRecord(
            task_id=uuid.uuid4().hex[:12],
            kind=kind,
            total=total,
            uploaded_files=uploaded or [],
            parse_options=parse_options or {},
            source=source,
            origin_task_id=origin_task_id,
            force=force,
            failed_only=failed_only,
        )
        with self._TASKS_LOCK:
            self._tasks[rec.task_id] = rec
        self._persist(rec)
        return rec

    def _spawn(self, target, rec: TaskRecord, options: ParseOptions) -> None:
        """Start a daemon thread for the parse / ingest work."""
        thread = threading.Thread(
            target=target,
            args=(self, rec, options),
            name=f"mmrag-{rec.kind}-{rec.task_id}",
            daemon=True,
        )
        thread.start()

    def _patch(self, rec: TaskRecord, **fields: Any) -> None:
        with self._TASKS_LOCK:
            for k, v in fields.items():
                setattr(rec, k, v)
            payload = self._snapshot_payload(rec)
        self._persist(rec)
        with self._TASKS_LOCK:
            entries = self._stream_events.get(rec.task_id)
            if entries:
                # Update every subscriber's payload in place and wake
                # them. List allows concurrent clients (e.g. two
                # browser tabs streaming the same task) without
                # dropping events for the older one.
                self._stream_events[rec.task_id] = [(event, payload) for event, _ in entries]
                for event, _ in entries:
                    event.set()

    def _snapshot_payload(self, rec: TaskRecord) -> dict[str, object]:
        payload = asdict(rec)
        payload["elapsed_sec"] = round((rec.finished_at or time.time()) - rec.started_at, 1)
        payload["progress"] = round(rec.processed / rec.total, 3) if rec.total else None
        return payload

    def stream_task(self, task_id: str, *, heartbeat: float = 15.0):
        """Yield NDJSON-friendly events for ``task_id`` until it terminates.

        Schema: ``{"event": "snapshot", "task": {...}}`` on every patch,
        ``{"event": "heartbeat"}`` after ``heartbeat`` seconds of silence,
        ``{"event": "done"}`` once the task reaches a terminal status.
        Unknown task ids yield a single ``{"event": "error", ...}`` and
        exit.
        """
        with self._TASKS_LOCK:
            rec = self._tasks.get(task_id)
            if rec is None:
                yield {
                    "event": "error",
                    "message": f"unknown task {task_id}",
                }
                return
            initial_payload = self._snapshot_payload(rec)
            event = threading.Event()
            entries = self._stream_events.setdefault(task_id, [])
            entries.append((event, initial_payload))
        try:
            yield {"event": "snapshot", "task": initial_payload}
            terminal = TaskStatus.terminal()
            last_payload = initial_payload
            while True:
                event_is_set = event.wait(timeout=heartbeat)
                with self._TASKS_LOCK:
                    live = self._stream_events.get(task_id)
                    if live is None:
                        return
                    # Find our entry's current payload; if the
                    # broadcaster updated the list since, pick up the
                    # freshest one. If our Event was removed (the
                    # other subscriber's ``finally`` raced us), bail.
                    payload = initial_payload
                    found_our_entry = False
                    for e, p in live:
                        if e is event:
                            payload = p
                            found_our_entry = True
                            break
                    if not found_our_entry:
                        return
                    event.clear()
                    rec_status = self._tasks.get(task_id)
                    status = rec_status.status if rec_status else None
                if event_is_set and payload is not initial_payload and payload is not last_payload:
                    last_payload = payload
                    yield {"event": "snapshot", "task": payload}
                if status in terminal:
                    yield {"event": "done", "status": status}
                    return
                if not event_is_set:
                    yield {"event": "heartbeat"}
        finally:
            with self._TASKS_LOCK:
                entries = self._stream_events.get(task_id)
                if entries is not None:
                    self._stream_events[task_id] = [(e, p) for (e, p) in entries if e is not event]
                    if not self._stream_events[task_id]:
                        self._stream_events.pop(task_id, None)

    def _persist(self, rec: TaskRecord) -> None:
        path = self._tasks_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(asdict(rec), ensure_ascii=False) + "\n"
            with self._PERSIST_LOCK, path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            # Surface the failure on the record itself so the next
            # ``/tasks/{id}`` poll or stream snapshot can show it.
            # We do not raise because the in-memory task can still
            # finish; the next process restart loses this task's
            # history, but the user at least sees a status warning
            # instead of an apparent success.
            rec.error = (rec.error or "") + f"; persist failed: {exc}"
            print(f"[tasks] warning: could not persist {rec.task_id}: {exc}")

    def _tasks_log_path(self) -> Path:
        return get_data_dir() / "tasks.jsonl"

    @staticmethod
    def _serialise_options(options: ParseOptions) -> dict[str, object]:
        """Return the JSON-friendly subset of ``ParseOptions`` we persist.

        ``assets`` is intentionally excluded — it is the runtime input the
        caller already supplies at task spawn time.
        """
        return {
            "pdf_parser": options.pdf_parser,
            "enable_ocr": options.enable_ocr,
            "enable_vlm": options.enable_vlm,
            "image_provider": options.image_provider,
        }

    @staticmethod
    def _deserialise_options(raw: dict[str, object], assets: list[Asset]) -> ParseOptions:
        """Rehydrate a ``ParseOptions`` from a persisted snapshot."""
        options = ParseOptions(assets=list(assets))
        if isinstance(raw, dict):
            pdf_parser = raw.get("pdf_parser")
            if isinstance(pdf_parser, str) and pdf_parser in {"auto", "pymupdf", "paddleocr_vl"}:
                options.pdf_parser = pdf_parser
            if isinstance(raw.get("enable_ocr"), bool):
                options.enable_ocr = raw["enable_ocr"]
            if isinstance(raw.get("enable_vlm"), bool):
                options.enable_vlm = raw["enable_vlm"]
            image_provider = raw.get("image_provider")
            if isinstance(image_provider, str) and image_provider in {
                "lite",
                "sentence_transformers",
            }:
                options.image_provider = image_provider
        return options

    @staticmethod
    def _rebuild_assets_for_retry(relative_paths: list[str]) -> list[Asset]:
        """Reconstruct ``Asset`` objects from confirmed upload paths.

        Best-effort: re-sniffs each file under ``get_assets_dir()`` and
        uses ``from_sniffed()`` so the retry task gets a coherent asset
        list. Any path that is missing or no longer a supported type is
        silently skipped (e.g. files the user manually removed). Callers
        must check that the returned list is non-empty.
        """
        assets_dir = get_assets_dir()
        rebuilt: list[Asset] = []
        for rel in relative_paths:
            if not isinstance(rel, str) or not rel:
                continue
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                continue
            file_path = (assets_dir / rel_path).resolve()
            try:
                if not file_path.is_relative_to(assets_dir.resolve()):
                    continue
            except (ValueError, OSError):
                continue
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                sniffed = sniff(file_path)
            except Exception as exc:
                print(f"[retry] sniff failed for {file_path}: {exc}")
                continue
            if sniffed.source_type not in {"pdf", "image"}:
                continue
            rebuilt.append(
                from_sniffed(
                    sniffed,
                    rel_path.as_posix(),
                    asset_dir=assets_dir,
                )
            )
        return rebuilt


# ─── Worker functions (module-level so threading can call them) ────────


def _run_parse_task(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    """Parse either the full manifest or only the uploaded files.

    Captures ``BaseException`` (not just ``Exception``) so SystemExit /
    KeyboardInterrupt / abrupt shutdowns surface as ``status="failed"``
    with the message in ``error``, instead of silently leaving the task
    at ``done`` with stale state.
    """
    service._patch(rec, status="running", current="starting")
    try:
        _do_parse(service, rec, options)
    except BaseException as exc:
        service._patch(
            rec,
            status="failed",
            current=f"parse crashed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )
        print(f"[task {rec.task_id}] parse crashed: {exc!r}")
        return

    if rec.status == TaskStatus.FAILED:
        service._patch(rec, finished_at=time.time())
        return


def _do_parse(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    assets = list(options.assets)

    if not assets:
        service._patch(
            rec,
            status="done",
            current="no assets to parse",
            finished_at=time.time(),
        )
        return

    # ``--force`` retry must clear cached parsed/<id>/ so the parse loop
    # below re-runs every asset instead of short-circuiting on the
    # existing raw.jsonl. The check is no-op for fresh uploads.
    if rec.force:
        from .paths import get_parsed_dir as _gpd

        cleared: list[str] = []
        for a in assets:
            parsed_dir_a = _gpd() / a.asset_id
            if parsed_dir_a.exists():
                shutil.rmtree(parsed_dir_a, ignore_errors=True)
                cleared.append(a.asset_id)
        if cleared:
            scope = "failed" if rec.failed_only else "all"
            service._patch(
                rec,
                current=(
                    f"force: cleared {len(cleared)} {scope} parsed/ cache dir(s) before parse"
                ),
            )

    service._patch(rec, total=len(assets), current=f"parsing {len(assets)} asset(s)")

    failed = 0
    skipped = 0
    parsed = 0
    target = get_documents_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    local_statuses: dict[str, str] = {}
    for i, asset in enumerate(assets, start=1):
        try:
            from .paths import get_parsed_dir

            raw_path = get_parsed_dir() / asset.asset_id / "raw.jsonl"
            if raw_path.exists() and raw_path.stat().st_size > 0:
                skipped += 1
                local_statuses[asset.asset_id] = "skipped"
                service._patch(rec, processed=i, current=f"skip cached: {asset.asset_id}")
                continue
            try:
                if asset.source_type == "pdf":
                    parser = get_parser("pdf", options.pdf_parser)
                    docs = parser.parse(asset)
                elif asset.source_type == "image":
                    parser = get_parser("image", "image")
                    docs = parser.parse(
                        asset,
                        enable_ocr=options.enable_ocr,
                        enable_vlm=options.enable_vlm,
                    )
                else:
                    docs = []
            except Exception as exc:
                failed += 1
                local_statuses[asset.asset_id] = "failed"
                print(f"parse task failed for {asset.asset_id}: {exc}")
                service._patch(rec, processed=i, current=f"error {asset.asset_id}: {exc}")
                continue
            with target.open("a", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
            parsed += 1
            local_statuses[asset.asset_id] = "ok"
            service._patch(
                rec,
                processed=i,
                current=f"parsed {asset.asset_id} ({len(docs)} doc)",
            )
        except Exception as exc:
            failed += 1
            local_statuses[asset.asset_id] = "failed"
            service._patch(rec, processed=i, current=f"error {asset.asset_id}: {exc}")

    status = (
        TaskStatus.DONE if failed == 0 and skipped + parsed == len(assets) else TaskStatus.PARTIAL
    )
    merged_statuses = {**rec.asset_statuses, **local_statuses}
    service._patch(
        rec,
        status=status,
        finished_at=time.time(),
        current=f"parse {status}: parsed={parsed} skipped={skipped} failed={failed}",
        asset_statuses=merged_statuses,
    )


def _run_ingest_task(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    """Parse + index in sequence. Captures BaseException for reliable failure surfacing."""
    _run_parse_task(service, rec, options)
    if rec.status == TaskStatus.FAILED:
        service._patch(rec, finished_at=time.time())
        return
    parse_status = rec.status

    # Snapshot which assets the parse step successfully produced. Only
    # these are candidates for the per-asset index status updates below.
    indexed_targets = {
        aid: status
        for aid, status in rec.asset_statuses.items()
        if status in {AssetStatus.OK, AssetStatus.SKIPPED}
    }

    def _progress(done: int, total: int, phase: str) -> None:
        service._patch(
            rec,
            processed=done,
            total=total,
            current=f"indexing: {phase}",
        )

    try:
        backend = get_backend("qdrant")
        text_n, text_name = backend.upsert_text(progress_cb=_progress)
        service._patch(rec, current=f"text indexed · {text_name}")
        image_n, _image_name = backend.upsert_image(progress_cb=_progress)
        # Index succeeded: mark parse-ok assets as fully indexed so
        # --failed-only retry can skip them next time.
        new_statuses = dict(rec.asset_statuses)
        for aid in indexed_targets:
            new_statuses[aid] = "indexed"
        service._patch(
            rec,
            current=f"index built · text={text_n} image={image_n}",
            status=parse_status,
            finished_at=time.time(),
            asset_statuses=new_statuses,
        )
    except BaseException as exc:
        new_statuses = dict(rec.asset_statuses)
        for aid in indexed_targets:
            new_statuses[aid] = "failed_index"
        service._patch(
            rec,
            current=f"index crashed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
            status="failed",
            finished_at=time.time(),
            asset_statuses=new_statuses,
        )
        print(f"[task {rec.task_id}] index crashed: {exc!r}")


# ─── Module-level service singleton ────────────────────────────────────


_service: IngestService | None = None
_service_lock = threading.Lock()


def get_service() -> IngestService:
    """Return the process-wide ``IngestService`` singleton."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                load_env()
                _service = IngestService()
    return _service


def reset_service() -> None:
    """Drop the cached singleton. Used by tests that need a fresh service."""
    global _service
    with _service_lock:
        _service = None
