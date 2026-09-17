"""Public facade for ingest workers, task lifecycle, and task history.

``IngestService`` retains thread spawning, cancellation, retry, streaming,
and public task APIs. SQLite persistence is delegated to ``TaskStore`` and
parse/enrich/document/index sequencing is delegated to ``IngestWorkflow``.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from . import parsers as _parsers  # noqa: F401  # register built-in parsers
from .core.config import load_env
from .core.paths import (
    get_asset_index_path,
    get_assets_dir,
    get_captions_dir,
    get_data_dir,
    get_documents_jsonl,
    get_parsed_dir,
    physical_cache_id,
)
from .core.protocols import KnowledgeBackend
from .core.registry import get_backend
from .core.registry import get_parser as get_parser
from .core.settings import Settings, get_settings
from .ingest import asset_index
from .ingest.assets import IngestAsset, from_sniffed
from .ingest.document_store import documents_jsonl_lock
from .ingest.ingest_workflow import IngestWorkflow
from .ingest.sniff import sniff
from .ingest.task_store import TaskRecord, TaskStore
from .query.query_preprocess import invalidate_vocab_cache

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
    CANCELLED = "cancelled"

    @classmethod
    def terminal(cls) -> set[TaskStatus]:
        # ``cancelled`` is terminal — once set, the worker stops at the next
        # checkpoint and never flips back to running.
        return {cls.DONE, cls.PARTIAL, cls.FAILED, cls.INTERRUPTED, cls.CANCELLED}


_RETRY_ELIGIBLE_DOCUMENT_STATUSES = {"failed", "skipped", "failed_index", None}


# ─── Data types ─────────────────────────────────────────────────────────


@dataclass
class ParseOptions:
    """Per-task parse configuration for uploaded assets."""

    assets: list[IngestAsset] = field(default_factory=list)
    pdf_parser: str = "auto"
    document_parser: str = "markitdown"
    audio_parser: str = "funasr"
    enable_ocr: bool = False
    enable_vlm: bool = False
    contextual: bool = False


@dataclass
class DocumentLifecycleReport:
    """Outcome of deleting a logical document and its current asset."""

    document_id: str
    documents_removed: int = 0
    chunks_removed: int = 0
    files_deleted: int = 0
    parsed_caches_deleted: int = 0
    caption_caches_deleted: int = 0
    text_collections_scanned: int = 0
    image_collections_scanned: int = 0
    errors: list[str] = field(default_factory=list)
    was_known: bool = True


# ─── Task bookkeeping ────────────────────────────────────────────────────


class IngestService:
    """Stateful facade for ingest, task lifecycle, and task history.

    A single instance is constructed per process and shared between the
    FastAPI app and (in the future) the CLI. The module-level
    :func:`get_service` returns the same instance for convenience.
    """

    _TASKS_LOCK = threading.Lock()

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        task_store: TaskStore | None = None,
        workflow: IngestWorkflow | None = None,
        backend: KnowledgeBackend | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._backend = (
            backend if backend is not None else get_backend(self._settings.vector_backend)
        )
        self._task_store = task_store if task_store is not None else TaskStore()
        self._workflow = workflow if workflow is not None else IngestWorkflow()
        self._tasks: dict[str, TaskRecord] = {}
        self._stream_events: dict[str, list[tuple[threading.Event, dict[str, object]]]] = {}
        # Per-task cancellation flags. ``_spawn`` creates one when a task
        # starts; ``cancel_task`` sets it; the worker checks it at each
        # asset checkpoint and stops cleanly. Daemon threads can't be
        # force-interrupted, so cancel is cooperative — a task mid-parse of
        # one large asset still finishes that asset before stopping.
        self._cancel_flags: dict[str, threading.Event] = {}

    @property
    def backend(self) -> KnowledgeBackend:
        """The configured index backend, exposed to the ingest workflow."""
        return self._backend

    # ─── Public API used by both FastAPI and CLI ─────────────────────────

    def parse_assets(
        self, assets: list[IngestAsset], options: ParseOptions | None = None
    ) -> TaskRecord:
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

    def ingest_assets(
        self, assets: list[IngestAsset], options: ParseOptions | None = None
    ) -> TaskRecord:
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
        self._spawn(self._workflow.run, rec, options)
        return rec

    def retry_task(
        self,
        task_id: str,
        *,
        force: bool = False,
        failed_only: bool = False,
    ) -> TaskRecord:
        """Re-run a previously failed/partial/interrupted task.

        Reconstructs ``IngestAsset`` objects from the original task's
        ``uploaded_files`` (best-effort via re-sniff), and spawns a new
        background task that mirrors the original ``kind`` and
        ``parse_options``. The new task is recorded with
        ``source="retry"`` and ``origin_task_id`` pointing back to the
        original.

        ``force=True`` clears cached ``parsed/<id>/raw.jsonl`` before
        the retry so the parse loop re-reads from disk. ``failed_only=True``
        narrows the retry set to documents whose status is missing,
        ``failed`` or ``skipped``. The two flags compose: with both set,
        only the failed documents are re-parsed and only their caches are
        cleared — useful after upgrading the parser without touching
        already-indexed documents. Tasks without ``document_statuses`` fall
        back to running every uploaded document and emit a warning.
        """
        with self._TASKS_LOCK:
            original = self._tasks.get(task_id)
        if original is None:
            raise KeyError(f"unknown task {task_id}")
        if original.status not in {TaskStatus.FAILED, TaskStatus.PARTIAL, TaskStatus.INTERRUPTED}:
            raise ValueError(f"task {task_id} cannot be retried (status={original.status})")
        if failed_only and not original.document_statuses:
            print(
                f"[retry] task {task_id} has no per-version statuses; treating failed_only as force"
            )
            force = True
            failed_only = False
        assets = self._rebuild_assets_for_retry(original.uploaded_files)
        if not assets:
            raise FileNotFoundError(f"no assets available to retry for task {task_id}")
        if failed_only and original.document_statuses:
            assets = [
                a
                for a in assets
                if original.document_statuses.get(self._document_status_key(a))
                in _RETRY_ELIGIBLE_DOCUMENT_STATUSES
            ]
            if not assets:
                raise FileNotFoundError(f"no failed or skipped assets to retry for task {task_id}")
        options = self._deserialise_options(original.parse_options, assets)
        uploaded = [a.relative_path for a in assets]
        preserved_statuses = dict(original.document_statuses)
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
            rec.document_statuses = preserved_statuses
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
            rec.document_statuses = preserved_statuses
            self._patch(rec)
            self._spawn(self._workflow.run, rec, options)
        else:
            raise ValueError(f"unknown task kind for retry: {original.kind!r}")
        return rec

    def reindex(self, text_only: bool = False, image_only: bool = False) -> tuple[str, ...]:
        """Force-recreate the active backend indexes from documents.jsonl."""
        backend = self._backend
        results = []
        if not image_only:
            _n, name = backend.upsert_text(force_recreate=True)
            results.append(f"text: {name}")
        if not text_only:
            _ni, ni_name = backend.upsert_image(force_recreate=True)
            results.append(f"image: {ni_name}")
        # Collections were just recreated with fresh vocab / IDF stats;
        # drop any in-process caches that were built against the old
        # collections so the next query rebuilds them against the new
        # state. Lazy imports + ``suppress(Exception)`` keep this
        # best-effort: a missing function (e.g. the BM25-zh invalidator
        # not yet added) or any import / runtime failure must not
        # crash the reindex path.
        self._invalidate_search_caches()
        return tuple(results)

    def _invalidate_search_caches(self) -> None:
        """Drop corpus and backend-derived caches after an index mutation."""
        invalidate_vocab_cache()
        self._backend.invalidate_caches()

    def list_tasks(self) -> list[TaskRecord]:
        """Return the task history ordered by most recent ``updated_at``.

        Reads through ``TaskStore`` from ``tasks.db`` instead of the
        in-memory dict. This means a process
        that never called ``load_history`` still sees the persisted
        history, and the SQL ``ORDER BY updated_at DESC`` gives a
        deterministic, time-ordered view rather than the dict's
        insertion order.

        In-memory recs are overlaid on top of the SQLite rows: for any
        ``task_id`` present in both, the in-memory copy wins because
        it's the one ``_patch`` mutates and ``_persist`` serialises
        from — it is strictly fresher than whatever row SQLite has.
        Recs that exist only in memory (the small window between
        ``_new_task`` adding to ``self._tasks`` and ``_persist``
        landing the first SQLite row) are prepended in newest-first
        order so a just-created task doesn't briefly vanish from the
        UI. The merged result keeps SQLite's ``updated_at DESC``
        order for the overlapping rows.
        """
        persisted = self._task_store.list()
        # Snapshot the in-memory view *under the lock* so a concurrent
        # ``_patch`` can't mutate a rec mid-overlay. We then iterate
        # SQLite rows in their existing desc order and substitute the
        # memory copy where present; any memory-only recs are prepended.
        with self._TASKS_LOCK:
            mem_snapshot = dict(self._tasks)
        out: list[TaskRecord] = []
        seen_ids: set[str] = set()
        for stored_rec in persisted:
            task_id = stored_rec.task_id
            if task_id in mem_snapshot:
                out.append(mem_snapshot[task_id])
                seen_ids.add(task_id)
            else:
                out.append(stored_rec)
        # Memory-only recs: created via ``_new_task`` but not yet
        # persisted to SQLite (the race window M8 closes). Prepend
        # newest-first so the just-spawned task shows at the top of
        # the UI history where users expect to see it.
        mem_only = [r for tid, r in mem_snapshot.items() if tid not in seen_ids]
        mem_only.sort(key=lambda r: r.started_at, reverse=True)
        return mem_only + out

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._TASKS_LOCK:
            return self._tasks.get(task_id)

    def load_history(self) -> None:
        """Restore tasks from SQLite and interrupt stale running records."""
        records = self._task_store.load()

        interrupted = 0
        with self._TASKS_LOCK:
            for rec in records:
                if rec.finished_at is None and rec.status == "running":
                    rec.status = "interrupted"
                    rec.current = (f"interrupted (previous process exited): {rec.current}").strip(
                        ": "
                    )
                    rec.finished_at = time.time()
                    rec.error = rec.error or "process exited before task completed"
                    interrupted += 1
                    self._persist(rec)
                self._tasks[rec.task_id] = rec
        if records:
            print(
                f"[tasks] loaded {len(records)} task(s) from disk; {interrupted} marked interrupted"
            )

    def delete_document(self, document_id: str) -> DocumentLifecycleReport:
        """Delete one logical document and its current asset."""
        records = [
            record
            for record in asset_index.load_records()
            if record.document.document_id == document_id
        ]
        return self._delete_document_records(document_id, records)

    def _delete_document_records(
        self,
        document_id: str,
        records: list[asset_index.DocumentRecord],
    ) -> DocumentLifecycleReport:
        report = DocumentLifecycleReport(document_id=document_id)
        if not document_id or not records:
            report.was_known = False
            return report

        all_records = asset_index.load_records()
        survivors = [record for record in all_records if record.document.document_id != document_id]
        docs_path = get_documents_jsonl()
        index_path = get_asset_index_path()
        docs_snapshot = docs_path.read_bytes() if docs_path.exists() else None
        index_snapshot = index_path.read_bytes() if index_path.exists() else None

        survivor_paths = {record.asset.relative_path for record in survivors}
        survivor_cache_keys = {physical_cache_id(path) for path in survivor_paths}
        assets_dir = get_assets_dir().resolve()
        staging_root = get_data_dir() / ".delete-staging" / uuid.uuid4().hex
        staged: list[tuple[Path, Path]] = []
        for record in records:
            relative_path = Path(record.asset.relative_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                report.errors.append(f"unsafe relative_path in document index: {relative_path}")
                continue
            if record.asset.relative_path not in survivor_paths:
                try:
                    file_path = (assets_dir / relative_path).resolve()
                    if not file_path.is_relative_to(assets_dir):
                        raise OSError(f"asset path escapes asset root: {relative_path}")
                    if _stage_delete_path(file_path, staging_root, staged):
                        report.files_deleted += 1
                except OSError as exc:
                    report.errors.append(f"file delete failed for {document_id}: {exc}")

            cache_key = physical_cache_id(record.asset.relative_path)
            if cache_key in survivor_cache_keys:
                continue
            parsed_dir = get_parsed_dir() / cache_key
            try:
                if _stage_delete_path(parsed_dir, staging_root, staged):
                    report.parsed_caches_deleted += 1
            except OSError as exc:
                report.errors.append(f"parsed cache delete failed for {document_id}: {exc}")
            for suffix in (".jsonl", ".json"):
                caption_path = get_captions_dir() / f"{cache_key}{suffix}"
                try:
                    if _stage_delete_path(caption_path, staging_root, staged):
                        report.caption_caches_deleted += 1
                except OSError as exc:
                    report.errors.append(f"caption cache delete failed for {document_id}: {exc}")

        if report.errors:
            report.errors.extend(_rollback_staged_paths(staged))
            _remove_empty_staging_root(staging_root)
            report.files_deleted = 0
            report.parsed_caches_deleted = 0
            report.caption_caches_deleted = 0
            report.errors.append(
                "document index retained because cleanup reported errors; fix and retry"
            )
            return report

        try:
            report.chunks_removed = _remove_document_rows_from_documents_jsonl(document_id)
        except OSError as exc:
            report.errors.append(f"documents.jsonl rewrite failed: {exc}")
            report.errors.extend(_rollback_staged_paths(staged))
            _remove_empty_staging_root(staging_root)
            report.files_deleted = 0
            report.parsed_caches_deleted = 0
            report.caption_caches_deleted = 0
            return report

        try:
            report.documents_removed = _remove_document_records(document_id)
        except OSError as exc:
            report.errors.append(f"document index rewrite failed: {exc}")
            _restore_file_snapshot(docs_path, docs_snapshot)
            report.errors.extend(_rollback_staged_paths(staged))
            _remove_empty_staging_root(staging_root)
            report.chunks_removed = 0
            report.files_deleted = 0
            report.parsed_caches_deleted = 0
            report.caption_caches_deleted = 0
            return report

        try:
            counts = self._backend.delete_documents({document_id})
            report.text_collections_scanned = counts["text"]
            report.image_collections_scanned = counts["image"]
        except Exception as exc:
            report.errors.append(f"backend delete failed: {exc}")
            _restore_file_snapshot(docs_path, docs_snapshot)
            _restore_file_snapshot(index_path, index_snapshot)
            report.errors.extend(_rollback_staged_paths(staged))
            _remove_empty_staging_root(staging_root)
            report.chunks_removed = 0
            report.documents_removed = 0
            report.files_deleted = 0
            report.parsed_caches_deleted = 0
            report.caption_caches_deleted = 0
            return report
        try:
            if staging_root.exists():
                shutil.rmtree(staging_root)
        except OSError as exc:
            report.errors.append(f"staged physical cleanup failed: {exc}")
        self._invalidate_search_caches()
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
        self._cancel_flags[rec.task_id] = threading.Event()
        thread = threading.Thread(
            target=target,
            args=(self, rec, options),
            name=f"mmrag-{rec.kind}-{rec.task_id}",
            daemon=True,
        )
        thread.start()

    def _is_cancelled(self, task_id: str) -> bool:
        """True iff ``cancel_task`` was called for ``task_id``."""
        flag = self._cancel_flags.get(task_id)
        return flag is not None and flag.is_set()

    def cancel_task(self, task_id: str) -> TaskRecord:
        """Request cooperative cancellation of a running task.

        Sets the per-task stop flag; the worker checks it at each asset
        checkpoint (between parse/index of individual assets) and stops
        cleanly, marking the task ``cancelled``. A task that's already
        terminal stays untouched (calling cancel on a done/failed task is
        a no-op that just returns its current record). Unknown task_id
        raises ``KeyError``.

        Daemon threads can't be force-killed, so a task currently deep in
        parsing/indexing one large asset finishes that asset before the
        flag takes effect — this is a cooperative, not preemptive, cancel.
        """
        with self._TASKS_LOCK:
            rec = self._tasks.get(task_id)
        if rec is None:
            raise KeyError(f"unknown task {task_id}")
        flag = self._cancel_flags.setdefault(task_id, threading.Event())
        flag.set()
        # If the task is already terminal, don't resurrect it as cancelled.
        if rec.status not in TaskStatus.terminal():
            self._patch(
                rec,
                status=TaskStatus.CANCELLED,
                finished_at=time.time(),
                current="cancelled by request",
            )
        return rec

    def _patch(self, rec: TaskRecord, **fields: Any) -> None:
        # Hold ``_TASKS_LOCK`` across both the setattr *and* the
        # ``_persist`` call so the snapshot ``_persist`` serialises
        # with ``asdict(rec)`` is the same one we publish to stream
        # subscribers. Previously ``_persist`` ran outside the lock,
        # which meant a concurrent ``_patch`` could setattr the rec
        # mid-serialise and produce a torn JSON line on disk (some
        # fields from the old state, some from the new). ``_persist``
        # delegates to ``TaskStore``, which only takes its class-level
        # persistence lock (not ``_TASKS_LOCK``), so this can't deadlock;
        # the critical section is short because the
        # JSONL tail-append happens after the SQLite write releases
        # the connection.
        with self._TASKS_LOCK:
            for k, v in fields.items():
                setattr(rec, k, v)
            payload = self._snapshot_payload(rec)
            self._persist(rec)
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

    def stream_task(self, task_id: str, *, heartbeat: float = 2.0):
        """Yield NDJSON-friendly events for ``task_id`` until it terminates.

        Schema: ``{"event": "snapshot", "task": {...}}`` on every patch,
        ``{"event": "heartbeat"}`` after ``heartbeat`` seconds of silence,
        ``{"event": "done"}`` once the task reaches a terminal status.
        Unknown task ids yield a single ``{"event": "error", ...}`` and
        exit.

        The default ``heartbeat`` is 2s (not 15s) so that a client that
        disconnects without closing the SSE stream frees the worker
        thread quickly — ``event.wait(timeout=heartbeat)`` is the only
        signal the generator loop has that the consumer is gone, and a
        15s timeout kept the thread pinned for the whole window on a
        dropped connection. 2s bounds the wasted threadpool occupancy
        without meaningfully increasing event-loop overhead.
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
        self._task_store.save(rec)

    def _tasks_db_path(self) -> Path:
        return self._task_store.db_path()

    @staticmethod
    def _serialise_options(options: ParseOptions) -> dict[str, object]:
        """Return the JSON-friendly subset of ``ParseOptions`` we persist.

        ``assets`` is intentionally excluded — it is the runtime input the
        caller already supplies at task spawn time.
        """
        return {
            "pdf_parser": options.pdf_parser,
            "document_parser": options.document_parser,
            "audio_parser": options.audio_parser,
            "enable_ocr": options.enable_ocr,
            "enable_vlm": options.enable_vlm,
            # Persisted so a retry of a ``--contextual`` task keeps the
            # per-task override; without this the retry silently falls back
            # to the global ``CONTEXTUAL_ENABLED`` and produces chunks
            # without the context preamble, inconsistent with the original.
            "contextual": options.contextual,
        }

    @staticmethod
    def _deserialise_options(raw: dict[str, object], assets: list[IngestAsset]) -> ParseOptions:
        """Rehydrate a ``ParseOptions`` from a persisted snapshot."""
        options = ParseOptions(assets=list(assets))
        if isinstance(raw, dict):
            pdf_parser = raw.get("pdf_parser")
            if isinstance(pdf_parser, str) and pdf_parser in {
                "auto",
                "pymupdf",
                "paddleocr_vl",
                "docling",
            }:
                options.pdf_parser = pdf_parser
            document_parser = raw.get("document_parser")
            if isinstance(document_parser, str) and document_parser in {
                "markitdown",
                "docling",
            }:
                options.document_parser = document_parser
            audio_parser = raw.get("audio_parser")
            if isinstance(audio_parser, str) and audio_parser:
                options.audio_parser = audio_parser
            if isinstance(raw.get("enable_ocr"), bool):
                options.enable_ocr = raw["enable_ocr"]
            if isinstance(raw.get("enable_vlm"), bool):
                options.enable_vlm = raw["enable_vlm"]
            if isinstance(raw.get("contextual"), bool):
                options.contextual = raw["contextual"]
        return options

    @staticmethod
    def _rebuild_assets_for_retry(relative_paths: list[str]) -> list[IngestAsset]:
        """Reconstruct ``IngestAsset`` objects from confirmed upload paths.

        Best-effort: re-sniffs each file under ``get_assets_dir()`` and
        uses ``from_sniffed()`` so the retry task gets a coherent asset
        list. Any path that is missing or no longer a supported type is
        silently skipped (e.g. files the user manually removed). Callers
        must check that the returned list is non-empty.
        """
        assets_dir = get_assets_dir()
        rebuilt: list[IngestAsset] = []
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
            if sniffed.source_type not in {"pdf", "image", "document"}:
                continue
            rebuilt.append(
                from_sniffed(
                    sniffed,
                    rel_path.as_posix(),
                    asset_dir=assets_dir,
                    asset_id_override=physical_cache_id(rel_path.as_posix()),
                )
            )
        return rebuilt

    @staticmethod
    def _document_status_key(asset: IngestAsset) -> str:
        record = asset_index.find_by_relative_path(asset.relative_path)
        if record is None:
            raise ValueError(f"no persisted document for asset path {asset.relative_path!r}")
        return record.document.document_id


# ─── Worker functions (module-level so threading can call them) ────────


def _run_parse_task(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    """Compatibility worker entry; sequencing lives in ``IngestWorkflow``."""
    service._workflow.run_parse(service, rec, options)


def _remove_document_records(document_id: str) -> int:
    """Atomically remove one current document record."""
    target = get_asset_index_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with asset_index._index_guard(target, exclusive=True):
        current = asset_index._load_records_unlocked(target)
        records = [record for record in current if record.document.document_id != document_id]
        removed = len(current) - len(records)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_record(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return removed


def _restore_file_snapshot(target: Path, snapshot: bytes | None) -> None:
    """Atomically restore a local index after a later delete step fails."""
    if snapshot is None:
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.rollback")
    try:
        temporary.write_bytes(snapshot)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_delete_path(
    source: Path,
    staging_root: Path,
    staged: list[tuple[Path, Path]],
) -> bool:
    """Atomically move one physical deletion target into reversible staging."""
    if not source.exists():
        return False
    staging_root.mkdir(parents=True, exist_ok=True)
    destination = staging_root / f"{len(staged):04d}"
    os.replace(source, destination)
    staged.append((source, destination))
    return True


def _rollback_staged_paths(staged: list[tuple[Path, Path]]) -> list[str]:
    """Restore staged files/directories in reverse order and report failures."""
    errors: list[str] = []
    for source, staged_path in reversed(staged):
        try:
            if not staged_path.exists():
                continue
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, source)
        except OSError as exc:
            errors.append(f"physical rollback failed for {source}: {exc}")
    return errors


def _remove_empty_staging_root(staging_root: Path) -> None:
    """Remove only empty staging directories, preserving unrecovered originals."""
    with suppress(FileNotFoundError, OSError):
        staging_root.rmdir()
    with suppress(FileNotFoundError, OSError):
        staging_root.parent.rmdir()


def _remove_document_rows_from_documents_jsonl(document_id: str) -> int:
    """Remove all current chunks for one document."""
    if not document_id:
        return 0
    docs_path = get_documents_jsonl()
    if not docs_path.exists():
        return 0
    removed = 0
    tmp_path = docs_path.with_suffix(docs_path.suffix + ".tmp")
    # Hold the documents.jsonl cross-process lock across read → tmp-write
    # → os.replace. The replace *must* be inside the lock: if it ran
    # outside, a concurrent appender could grab the lock right after we
    # release it, open("a") the old inode, and we'd then os.replace-swap
    # docs_path out from under it — the appender's writes would land in
    # the now-unlinked inode and vanish. Nest the two file contexts so a
    # dst.open failure doesn't leak the src fd.
    with documents_jsonl_lock(docs_path):
        src = docs_path.open("r", encoding="utf-8")
        try:
            dst = tmp_path.open("w", encoding="utf-8")
            try:
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
                    document = obj.get("document") if isinstance(obj, dict) else None
                    if isinstance(document, dict) and document.get("document_id") == document_id:
                        removed += 1
                        continue
                    dst.write(line)
            finally:
                dst.close()
        finally:
            src.close()
        os.replace(tmp_path, docs_path)
    return removed


def _do_parse(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    """Compatibility parse seam; implementation lives in ``IngestWorkflow``."""
    service._workflow.parse(service, rec, options)


def _run_ingest_task(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    """Compatibility worker entry; sequencing lives in ``IngestWorkflow``."""
    service._workflow.run(service, rec, options)


# ─── Module-level service singleton


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
