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

import argparse
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .backends.qdrant_backend import (
    build_qdrant_image_index,
    build_qdrant_text_index,
)
from .config import load_env
from .document_store import write_documents
from .paths import get_data_dir, get_documents_jsonl
from .parsers.image_parser import parse_image
from .parsers.pdf_parser import parse_pdf
from .settings import Settings, get_settings


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


@dataclass
class ParseOptions:
    """Per-task parse configuration. Comes from the ``/upload`` form or CLI."""

    pdf_parser: str = "auto"
    enable_ocr: bool = False
    enable_vlm: bool = False
    image_provider: str = "lite"
    only_uploaded: bool = False
    uploaded_files: list[str] = field(default_factory=list)


# ─── Task bookkeeping ────────────────────────────────────────────────────


class IngestService:
    """Stateful ingest + index + task-history service.

    A single instance is constructed per process and shared between the
    FastAPI app and (in the future) the CLI. The module-level
    :func:`get_service` returns the same instance for convenience.
    """

    _TASKS_LOCK = threading.Lock()

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._tasks: dict[str, TaskRecord] = {}

    # ─── Public API used by both FastAPI and CLI ─────────────────────────

    def parse_manifest(self, limit: int = 0, options: ParseOptions | None = None) -> TaskRecord:
        """Parse every asset in the bundled manifest (CLI ``mmrag parse``)."""
        options = options or ParseOptions()
        # The CLI path always parses the manifest; explicit override is fine.
        options.only_uploaded = False
        rec = self._new_task(kind="parse", total=0)
        self._spawn(_run_parse_task, rec, options)
        return rec

    def parse_uploaded(self, paths: list[str], options: ParseOptions) -> TaskRecord:
        """Parse just-uploaded files (FastAPI ``/upload``)."""
        options.only_uploaded = True
        options.uploaded_files = list(paths)
        rec = self._new_task(kind="parse", total=len(paths))
        self._spawn(_run_parse_task, rec, options)
        return rec

    def ingest_uploaded(self, paths: list[str], options: ParseOptions) -> TaskRecord:
        """Parse + index just-uploaded files (FastAPI ``/upload`` with auto_index)."""
        options.only_uploaded = True
        options.uploaded_files = list(paths)
        rec = self._new_task(kind="ingest", total=len(paths))
        self._spawn(_run_ingest_task, rec, options)
        return rec

    def reindex(self, text_only: bool = False, image_only: bool = False) -> tuple[str, ...]:
        """Force-recreate Qdrant collections and re-upsert from documents.jsonl."""
        results = []
        if not image_only:
            n, name = build_qdrant_text_index(force_recreate=True)
            results.append(f"text: {name}")
        if not text_only:
            ni, ni_name = build_qdrant_image_index(force_recreate=True)
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
                    self._persist(TaskRecord(**obj))  # type: ignore[arg-type]
                self._tasks[task_id] = TaskRecord(**obj)  # type: ignore[arg-type]
        if latest:
            print(
                f"[tasks] loaded {len(latest)} task(s) from disk; "
                f"{interrupted} marked interrupted"
            )

    # ─── Internals ─────────────────────────────────────────────────────

    def _new_task(self, kind: str, total: int, uploaded: list[str] | None = None) -> TaskRecord:
        rec = TaskRecord(
            task_id=uuid.uuid4().hex[:12],
            kind=kind,
            total=total,
            uploaded_files=uploaded or [],
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
        self._persist(rec)

    def _persist(self, rec: TaskRecord) -> None:
        path = self._tasks_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[tasks] warning: could not persist {rec.task_id}: {exc}")

    def _tasks_log_path(self) -> Path:
        return get_data_dir() / "tasks.jsonl"


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

    if rec.status == "failed":
        service._patch(rec, finished_at=time.time())
        return


def _do_parse(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    from .assets import Asset, load_assets

    if options.only_uploaded:
        assets_dir = service._settings.data_dir / "assets"
        assets: list[Asset] = []
        for rel in options.uploaded_files:
            full = assets_dir / rel
            if not full.exists():
                continue
            source_type = "pdf" if full.suffix.lower() == ".pdf" else "image"
            assets.append(
                Asset(
                    asset_id=full.stem,
                    title=full.name,
                    source_type=source_type,
                    relative_path=rel,
                    source_url="",
                    tags=[],
                    asset_dir=assets_dir,
                )
            )
    else:
        service._patch(rec, current="loading assets")
        assets = load_assets(limit=0)

    if not assets:
        service._patch(
            rec,
            status="done",
            current="no assets to parse",
            finished_at=time.time(),
        )
        return

    service._patch(rec, total=len(assets), current=f"parsing {len(assets)} asset(s)")

    failed = 0
    skipped = 0
    parsed = 0
    target = get_documents_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    for i, asset in enumerate(assets, start=1):
        try:
            from .paths import get_parsed_dir

            raw_path = get_parsed_dir() / asset.asset_id / "raw.jsonl"
            if raw_path.exists() and raw_path.stat().st_size > 0:
                skipped += 1
                service._patch(rec, processed=i, current=f"skip cached: {asset.asset_id}")
                continue
            try:
                if asset.source_type == "pdf":
                    docs = parse_pdf(asset, parser=options.pdf_parser)
                elif asset.source_type == "image":
                    docs = parse_image(
                        asset,
                        enable_ocr=options.enable_ocr,
                        enable_vlm=options.enable_vlm,
                    )
                else:
                    docs = []
            except Exception as exc:
                failed += 1
                print(f"parse task failed for {asset.asset_id}: {exc}")
                service._patch(
                    rec, processed=i, current=f"error {asset.asset_id}: {exc}"
                )
                continue
            with target.open("a", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
            parsed += 1
            service._patch(
                rec,
                processed=i,
                current=f"parsed {asset.asset_id} ({len(docs)} doc)",
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            service._patch(rec, processed=i, current=f"error {asset.asset_id}: {exc}")

    status = "done" if failed == 0 and skipped + parsed == len(assets) else "partial"
    service._patch(
        rec,
        status=status,
        finished_at=time.time(),
        current=f"parse {status}: parsed={parsed} skipped={skipped} failed={failed}",
    )


def _run_ingest_task(service: IngestService, rec: TaskRecord, options: ParseOptions) -> None:
    """Parse + index in sequence. Captures BaseException for reliable failure surfacing."""
    _run_parse_task(service, rec, options)
    if rec.status == "failed":
        service._patch(rec, finished_at=time.time())
        return
    parse_status = rec.status

    def _progress(done: int, total: int, phase: str) -> None:
        service._patch(
            rec,
            processed=done,
            total=total,
            current=f"indexing: {phase}",
        )

    try:
        text_n, text_name = build_qdrant_text_index(progress_cb=_progress)
        service._patch(rec, current=f"text indexed · {text_name}")
        image_n, image_name = build_qdrant_image_index(progress_cb=_progress)
        service._patch(
            rec,
            current=f"index built · text={text_n} image={image_n}",
            status=parse_status,
            finished_at=time.time(),
        )
    except BaseException as exc:
        service._patch(
            rec,
            current=f"index crashed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
            status="failed",
            finished_at=time.time(),
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