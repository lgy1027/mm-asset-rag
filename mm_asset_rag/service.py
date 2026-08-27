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

from . import asset_index
from . import parsers as _parsers  # noqa: F401  # register built-in parsers
from .asset_index import AssetIndexEntry
from .assets import Asset, from_sniffed
from .backends.qdrant_backend import (
    IMAGE_COLLECTION_BASE,
    TEXT_COLLECTION_BASE,
    _existing_collections_for,
    delete_points_by_asset_id,
    get_qdrant_client,
)
from .config import load_env
from .document_store import documents_jsonl_lock
from .ingest_workflow import IngestWorkflow
from .paths import (
    get_assets_dir,
    get_captions_dir,
    get_data_dir,
    get_documents_jsonl,
    get_parsed_dir,
)
from .registry import get_backend
from .registry import get_parser as get_parser
from .search_service import (
    SearchCommand,
    coerce_search_mode,
    get_search_service,
    resolve_sandboxed_image_path,
)
from .settings import Settings, get_settings
from .sniff import sniff
from .task_store import TaskRecord, TaskStore, task_from_dict

# ─── Helpers shared by api.py and cli.py ──────────────────────────────────


def _resolve_sandboxed_image_path(image_path: str | Path | None) -> Path | None:
    """Compatibility alias for the image-path resolver now owned by SearchService."""
    return resolve_sandboxed_image_path(image_path)


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


def dispatch_search(
    *,
    query: str,
    mode: str,
    image_path: str | Path | None,
    top_k: int,
) -> list:
    """Adapt the legacy primitive arguments into one typed search command."""
    return get_search_service().execute(
        SearchCommand(
            query=query,
            mode=coerce_search_mode(mode),
            image_path=image_path,
            top_k=top_k,
        )
    )


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
class ParseOptions:
    """Per-task parse configuration for uploaded/auto-sniffed assets.

    Note on ``image_provider``: the embedder dispatch in
    ``mm_asset_rag.embedders.build_default_image_embedder`` only reads
    ``Settings.image_provider`` — the per-task override here is currently
    round-tripped through ``_serialise_options`` / ``_deserialise_options``
    (so old task records keep their value) but is *not* consulted at
    dispatch time. To change the image backend, set ``IMAGE_PROVIDER`` in
    the environment / ``.env`` before launching ``mmrag-api``; mid-run
    switches require ``register_embedder(..., replace=True)``.
    """

    assets: list[Asset] = field(default_factory=list)
    pdf_parser: str = "auto"
    document_parser: str = "markitdown"
    enable_ocr: bool = False
    enable_vlm: bool = False
    image_provider: str = "lite"
    contextual: bool = False


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
    ) -> None:
        self._settings = settings or get_settings()
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
            self._spawn(self._workflow.run, rec, options)
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
        """Drop in-process vocab + BM25-zh IDF caches after a reindex or
        successful ingest.

        ``query_preprocess`` caches the corpus vocab used for query
        expansion / hyphenisation, and ``qdrant_backend`` caches the
        IDF vector used by the BM25-zh sparse index. Both caches are
        derived from the *current* Qdrant collection state, so after a
        ``force_recreate=True`` reindex (or any successful ``upsert_text``
        / ``upsert_image`` on the ingest path) the cached values go
        stale and silently degrade retrieval quality. We invalidate
        them here so the next query rebuilds them against the new data.

        Lazy imports keep ``service`` from hard-depending on either
        module at import time (e.g. ``invalidate_bm25_zh_idf_cache``
        may not exist yet in older deployed builds), and
        ``suppress(Exception)`` ensures cache invalidation can never
        crash the task-completion path — at worst we print nothing
        and the next query pays the cache-miss cost.
        """
        with suppress(Exception):
            from .query_preprocess import invalidate_vocab_cache

            invalidate_vocab_cache()
        with suppress(Exception):
            from .backends.qdrant_backend import invalidate_bm25_zh_idf_cache

            invalidate_bm25_zh_idf_cache()

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
        """Restore tasks from SQLite (auto-migrate from legacy ``tasks.jsonl``).

        Tasks still in ``running`` state when the previous process
        exited are marked ``interrupted``. If a legacy ``tasks.jsonl``
        is found alongside (no ``tasks.db`` yet), we migrate every
        line into SQLite once and rename the source file to
        ``tasks.jsonl.migrated`` so we don't redo the migration on the
        next boot.
        """
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

    def _maybe_legacy_migrate(self, db_path: Path) -> None:
        """One-shot migration: if ``tasks.db`` is missing but a legacy
        ``tasks.jsonl`` exists, import every line into SQLite and
        rename the source file to ``tasks.jsonl.migrated`` so we never
        redo it.

        A pre-existing ``tasks.db`` short-circuits the migration so we
        do not clobber the new store on a partial-boot upgrade.
        """
        self._task_store._maybe_legacy_migrate(db_path)

    def _tasks_log_path_legacy(self) -> Path:
        """Return the *legacy* ``tasks.jsonl`` location.

        Retained so the migration in :meth:`_maybe_legacy_migrate`
        can find the pre-SQLite history file. New writes go to
        :meth:`_tasks_db_path` / :meth:`_tasks_jsonl_path`.
        """
        return self._task_store.legacy_path()

    @staticmethod
    def _task_from_dict(obj: dict[str, object]) -> TaskRecord:
        """Build a ``TaskRecord`` from a JSONL row, tolerating legacy records."""
        return task_from_dict(obj)

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
        # Captions are cached per asset. Image assets use a single-object
        # ``.json`` (one VLM caption); documents with embedded figures use
        # ``.jsonl`` (one JSON-per-figure, written by image_caption). Check
        # both so the detail view reports the path that actually exists.
        captions_candidates = [
            get_captions_dir() / f"{asset_id}.jsonl",
            get_captions_dir() / f"{asset_id}.json",
        ]
        captions_path = next(
            (p for p in captions_candidates if p.exists()),
            captions_candidates[0],
        )

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

        # 3. captions/<asset_id>.{jsonl,json} — document embedded figures
        #    use .jsonl (one JSON per figure), image assets use .json. Clean
        #    both so no caption cache is left behind on delete.
        for cap_name in (f"{asset_id}.jsonl", f"{asset_id}.json"):
            try:
                captions_path = get_captions_dir() / cap_name
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
                    removed = _remove_asset_rows_from_documents_jsonl({asset_id})
                    report.documents_removed = removed
        except OSError as exc:
            report.errors.append(f"documents.jsonl rewrite failed: {exc}")

        # 5. Qdrant text + image collections
        if dry_run:
            # Preview which collections would be scanned. Resolved from the
            # live server (not the active-cache) so the count is accurate even
            # in a process that never ingested — previously dry_run always
            # reported 0, hiding the points that would be left behind.
            try:
                client = get_qdrant_client()
                tcols = (
                    [self._settings.qdrant_active_text_collection]
                    if (self._settings.qdrant_active_text_collection)
                    else _existing_collections_for(client, TEXT_COLLECTION_BASE)
                )
                icols = (
                    [self._settings.qdrant_active_image_collection]
                    if (self._settings.qdrant_active_image_collection)
                    else _existing_collections_for(client, IMAGE_COLLECTION_BASE)
                )
                report.text_collections_scanned = len(tcols)
                report.image_collections_scanned = len(icols)
                report.qdrant_note = (
                    f"would scan {len(tcols)} text + {len(icols)} image collection(s)"
                )
            except Exception as exc:
                report.qdrant_note = f"would scan text+image collections (listing failed: {exc})"
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
        """Compatibility delegate to the injected task store."""
        self._task_store.save(rec)

    def _tasks_db_path(self) -> Path:
        return self._task_store.db_path()

    def _tasks_jsonl_path(self) -> Path:
        return self._task_store.jsonl_path()

    @staticmethod
    def _serialise_options(options: ParseOptions) -> dict[str, object]:
        """Return the JSON-friendly subset of ``ParseOptions`` we persist.

        ``assets`` is intentionally excluded — it is the runtime input the
        caller already supplies at task spawn time.
        """
        return {
            "pdf_parser": options.pdf_parser,
            "document_parser": options.document_parser,
            "enable_ocr": options.enable_ocr,
            "enable_vlm": options.enable_vlm,
            "image_provider": options.image_provider,
            # Persisted so a retry of a ``--contextual`` task keeps the
            # per-task override; without this the retry silently falls back
            # to the global ``CONTEXTUAL_ENABLED`` and produces chunks
            # without the context preamble, inconsistent with the original.
            "contextual": options.contextual,
        }

    @staticmethod
    def _deserialise_options(raw: dict[str, object], assets: list[Asset]) -> ParseOptions:
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
            if isinstance(raw.get("enable_ocr"), bool):
                options.enable_ocr = raw["enable_ocr"]
            if isinstance(raw.get("enable_vlm"), bool):
                options.enable_vlm = raw["enable_vlm"]
            image_provider = raw.get("image_provider")
            if isinstance(image_provider, str) and image_provider in {
                "lite",
                "sentence_transformers",
                "cn_clip",
            }:
                options.image_provider = image_provider
            if isinstance(raw.get("contextual"), bool):
                options.contextual = raw["contextual"]
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
            if sniffed.source_type not in {"pdf", "image", "document"}:
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
    """Compatibility worker entry; sequencing lives in ``IngestWorkflow``."""
    service._workflow.run_parse(service, rec, options)


def _remove_asset_rows_from_documents_jsonl(asset_ids: set[str]) -> int:
    """Drop every ``documents.jsonl`` row whose ``metadata.asset_id`` is in ``asset_ids``.

    Atomic (tmp file + ``os.replace``), returns the number of rows removed.
    Robust to a missing file (returns 0) and to per-line JSON decode errors
    (those lines are kept as-is rather than aborting the rewrite). Shared by
    ``delete_asset`` (one asset) and the force-retry parse path (the assets
    whose ``parsed/<id>/`` cache was just cleared, so the re-parse does not
    append duplicate chunk rows next to the originals).
    """
    if not asset_ids:
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
                    meta = obj.get("metadata") if isinstance(obj, dict) else None
                    if isinstance(meta, dict) and str(meta.get("asset_id", "")) in asset_ids:
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
