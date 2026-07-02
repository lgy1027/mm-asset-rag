"""HTTP API for mm-asset-rag.

Endpoints
---------
GET  /health                       liveness + index state
POST /upload/preview (multipart)     sniff files + return editable metadata cards
POST /upload/confirm (json)          apply edits, parse + index in background
GET  /tasks/{id}                    poll background task status
POST /search   (json)               retrieval only
POST /answer   (json)               grounded LLM answer
POST /eval     (json)               run mmrag eval
POST /chat     (json)               one-call hybrid_search + answer
GET  /                              serves the bundled single-file web UI (index.html)
GET  /static/{path}                other static assets (none today, but reserved)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .__init__ import __version__
from .answer import answer_question, stream_answer_chunks
from .backends.qdrant_backend import (
    get_qdrant_client,
)
from .config import load_env
from .evaluation import run_eval
from .paths import (
    get_assets_dir,
    get_documents_jsonl,
    get_indexes_dir,
    get_preview_cache_dir,
    get_text_index_dir,
)
from .service import (
    ParseOptions,
    dispatch_search,
    get_service,
)
from .settings import get_settings
from .upload_pipeline import UploadCommitError, UploadManifestError, UserEdits, get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: restore task history from disk. Tasks still marked 'running'
    # when the previous process exited get reclassified as 'interrupted'.
    get_service().load_history()
    with suppress(Exception):
        get_pipeline().cleanup_expired_caches()
    yield
    # Graceful shutdown: close the qdrant client so it removes its .lock
    # file. If the process is killed before this runs, the next startup will
    # also tolerate a stale .lock (see qdrant_store._clean_stale_lock).
    with suppress(Exception):
        get_qdrant_client().close()


app = FastAPI(
    title="mm-asset-rag",
    version="0.1.0",
    description="Multimodal asset RAG: PDF + image parsing, hybrid retrieval, grounded answers.",
    lifespan=lifespan,
)


# ─── Service layer (task scheduling + history) ─────────────────────────
#
# All background work, persistence, and task queries live in
# ``mm_asset_rag.service``. The FastAPI app stays a thin route layer.

from .service import TaskRecord  # noqa: E402, F401

# ─── Static web UI ────────────────────────────────────────────────────────


WEB_DIR = Path(__file__).resolve().parent / "web"


# ─── Request / response models ───────────────────────────────────────────


def _validate_image_path(value: str | None) -> str | None:
    """Reject absolute paths / ``..`` traversal on user-supplied ``image_path``.

    ``dispatch_search`` re-resolves the path against ``assets_dir``, but
    bouncing clearly bad input at the API boundary gives the client a
    proper 422 instead of a 500 once the filesystem call fails.
    """
    if value is None:
        return value
    if not value.strip():
        return value
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("image_path must be a relative path inside assets/")
    return value


class _RouteRequest(BaseModel):
    """Shared fields for ``SearchRequest`` and ``ChatRequest``.

    Both endpoints expose the same routing surface: ``mode``, an
    optional ``image_path``, and a ``top_k``. Pulling them onto a base
    keeps the validator in one place so a change (e.g. adding a new
    mode) lands in both endpoints instead of drift-creeping.
    """

    mode: str = Field(default="hybrid", pattern="^(text|text-to-image|image-to-image|hybrid)$")
    image_path: str | None = Field(default=None, max_length=1024)
    top_k: int = Field(default=5, ge=1, le=200)

    @field_validator("image_path")
    @classmethod
    def _check_image_path(cls, v: str | None) -> str | None:
        return _validate_image_path(v)


class SearchRequest(_RouteRequest):
    query: str = Field(..., min_length=1, max_length=2000)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=200)


class EvalRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=200)


class ChatRequest(_RouteRequest):
    question: str = Field(..., min_length=1, max_length=2000)


class UploadEdit(BaseModel):
    preview_id: str
    title: str | None = None
    tags: list[str] | str | None = None
    description: str | None = None
    rejected: bool = False


class UploadConfirmRequest(BaseModel):
    cache_id: str
    edits: list[UploadEdit] = Field(default_factory=list)


# ─── Endpoints ──────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, object]:
    load_env()
    assets_dir = get_assets_dir()
    asset_files = [p for p in assets_dir.rglob("*") if p.is_file()] if assets_dir.exists() else []
    return {
        "status": "ok",
        "version": __version__,
        "assets": len(asset_files),
        "documents_jsonl_exists": get_documents_jsonl().exists(),
        "text_index_exists": get_text_index_dir().exists(),
        "image_index_exists": (get_indexes_dir() / "qdrant").exists(),
        "vector_backend": "qdrant",
        "model": os.environ.get("OPENAI_MODEL", ""),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    try:
        hits = dispatch_search(
            query=request.query,
            mode=request.mode,
            image_path=request.image_path,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"query": request.query, "mode": request.mode, "hits": [h.__dict__ for h in hits]}


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, object]:
    return answer_question(request.question, top_k=request.top_k)


@app.post("/eval")
def eval_endpoint(request: EvalRequest) -> dict[str, object]:
    return {"results": [r.__dict__ for r in run_eval(top_k=request.top_k)]}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    """One-call: retrieve + grounded LLM answer in a single response."""
    try:
        hits = dispatch_search(
            query=request.question,
            mode=request.mode,
            image_path=request.image_path,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    answer = answer_question(request.question, top_k=request.top_k, hits=hits)
    return {
        "question": request.question,
        "answer": answer,
        "sources": [h.__dict__ for h in hits],
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """NDJSON stream of the chat answer.

    Each line is a JSON object:

    - ``{"event": "sources", "sources": [...]}``  once, up front
    - ``{"event": "token", "text": "..."}``        one per LLM token
    - ``{"event": "done"}``                        exactly once at the end

    Implemented as an ``async`` generator so FastAPI can deliver
    tokens on the event loop directly — the LLM call (sync HTTP via
    ``openai``) runs through a thread via ``asyncio.to_thread`` so the
    worker pool doesn't get pinned by a long-running chat session.
    """

    async def gen():
        try:
            hits = await asyncio.to_thread(
                dispatch_search,
                query=request.question,
                mode=request.mode,
                image_path=request.image_path,
                top_k=request.top_k,
            )
            yield (
                json.dumps(
                    {"event": "sources", "sources": [h.__dict__ for h in hits]},
                    ensure_ascii=False,
                )
                + "\n"
            )

            # ``stream_answer_chunks`` is a sync generator (the OpenAI
            # SDK yields its stream chunks synchronously). Run the
            # blocking producer in a worker thread and ``await`` each
            # chunk so the event loop stays responsive to client
            # disconnects (``asyncio.CancelledError`` propagates
            # through ``to_thread`` and stops further yield).
            def produce():
                return list(stream_answer_chunks(request.question, hits))

            chunks = await asyncio.to_thread(produce)
            for chunk in chunks:
                yield json.dumps({"event": "token", "text": chunk}, ensure_ascii=False) + "\n"
            yield json.dumps({"event": "done"}, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ─── Upload + background parse ───────────────────────────────────────────


@app.post("/upload/preview")
async def upload_preview(files: list[UploadFile] = File(...)) -> dict[str, object]:
    """Stage uploaded files and return editable metadata previews.

    This endpoint does not parse, embed or index. It streams multipart bytes
    into a short-lived incoming directory, lets ``UploadPipeline`` sniff and
    optionally VLM-tag them, then returns preview cards for the web UI.
    """
    load_env()
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    settings = get_settings()
    with suppress(Exception):
        get_pipeline().cleanup_expired_caches()
    incoming_dir = get_preview_cache_dir() / f"incoming_{uuid.uuid4().hex[:12]}"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[str, Path]] = []
    rejected: list[dict[str, str]] = []
    batch_bytes = 0
    try:
        for f in files:
            name = Path(f.filename or "").name
            if not name:
                rejected.append({"filename": "", "reason": "empty filename"})
                continue
            # ``Path("..").name`` returns ``".."``; reject it and the
            # other special cases before the bytes hit disk. Also
            # normalise to NFC so a Unicode-look-alike directory name
            # cannot pass a server-side ``safe`` check by re-encoding
            # as NFD.
            import unicodedata

            name = unicodedata.normalize("NFC", name)
            if name in {"", ".", ".."} or "/" in name or "\\" in name:
                rejected.append({"filename": f.filename or "", "reason": f"unsafe name: {name!r}"})
                continue
            target = incoming_dir / name
            if target.exists():
                target = incoming_dir / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
            file_bytes = 0
            with target.open("wb") as out:
                while chunk := f.file.read(1024 * 1024):
                    file_bytes += len(chunk)
                    batch_bytes += len(chunk)
                    if file_bytes > settings.upload_max_file_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{name} exceeds upload_max_file_bytes",
                        )
                    if batch_bytes > settings.upload_max_batch_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="batch exceeds upload_max_batch_bytes",
                        )
                    out.write(chunk)
            staged.append((name, target))

        if not staged:
            raise HTTPException(status_code=400, detail={"rejected": rejected})

        previews = get_pipeline().preview(staged)
    except HTTPException:
        raise
    finally:
        import shutil

        shutil.rmtree(incoming_dir, ignore_errors=True)

    cache_id = previews[0].cache_id if previews else ""
    return {
        "cache_id": cache_id,
        "previews": [
            json.loads(json.dumps(asdict(p), ensure_ascii=False, default=str)) for p in previews
        ],
        "rejected": rejected,
    }


@app.post("/upload/confirm")
def upload_confirm(request: UploadConfirmRequest) -> dict[str, object]:
    """Apply user edits and kick off parse + index for confirmed previews."""
    edits = [
        UserEdits(
            preview_id=e.preview_id,
            title=e.title,
            tags=e.tags if isinstance(e.tags, list) else None if e.tags is None else [e.tags],
            description=e.description,
            rejected=e.rejected,
        )
        for e in request.edits
    ]
    try:
        assets = get_pipeline().confirm(request.cache_id, edits)
    except (KeyError, UploadManifestError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadCommitError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not assets:
        raise HTTPException(status_code=400, detail="no confirmed assets")

    options = ParseOptions(assets=assets)
    rec = get_service().ingest_assets(assets, options)
    return {
        "task_id": rec.task_id,
        "kind": rec.kind,
        "uploaded": [a.relative_path for a in assets],
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, object]:
    from dataclasses import asdict

    rec = get_service().get_task(task_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    payload = asdict(rec)
    payload["elapsed_sec"] = round((rec.finished_at or time.time()) - rec.started_at, 1)
    if rec.total:
        payload["progress"] = round(rec.processed / rec.total, 3)
    else:
        payload["progress"] = None
    return payload


@app.get("/tasks")
def list_tasks() -> dict[str, object]:
    from dataclasses import asdict

    return {"tasks": [asdict(t) for t in get_service().list_tasks()]}


@app.get("/tasks/{task_id}/stream")
def task_stream(task_id: str) -> StreamingResponse:
    """Stream task snapshots as NDJSON.

    Events: ``snapshot`` (per task patch), ``heartbeat`` (every
    ~15 s of silence), ``done`` (terminal status reached), ``error``
    (unknown task id). See ``IngestService.stream_task`` for the
    generator semantics.
    """

    def gen():
        try:
            for event in get_service().stream_task(task_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tasks/{task_id}/retry")
def retry_task(
    task_id: str,
    force: bool = Query(False, description="Clear parsed/<id>/ cache before re-running"),
    failed_only: bool = Query(
        False,
        description="Only re-run assets whose previous status was failed or skipped",
    ),
) -> dict[str, object]:
    """Re-run a previously failed/partial/interrupted task.

    The new task mirrors the original ``kind`` and ``parse_options`` and
    is recorded with ``source="retry"`` and ``origin_task_id`` pointing
    back to the original task.
    """
    try:
        rec = get_service().retry_task(task_id, force=force, failed_only=failed_only)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "task_id": rec.task_id,
        "kind": rec.kind,
        "origin_task_id": rec.origin_task_id,
        "source": rec.source,
        "force": rec.force,
        "failed_only": rec.failed_only,
        "uploaded": rec.uploaded_files,
    }


@app.get("/assets")
def list_assets() -> dict[str, object]:
    """Return every non-deleted asset recorded in the asset index."""
    entries = get_service().list_assets()
    return {
        "assets": [
            {
                "asset_id": entry.asset_id,
                "relative_path": entry.relative_path,
                "source_type": entry.source_type,
                "asset_title": entry.asset_title,
                "ingested_at": entry.ingested_at,
            }
            for entry in entries
        ]
    }


@app.delete("/assets/{asset_id}")
def delete_asset(asset_id: str) -> dict[str, object]:
    """Best-effort cleanup of every trace of ``asset_id``.

    404 if the asset is unknown to the index. 200 with a ``was_known``
    payload otherwise. ``errors`` lists any cleanup step that failed.
    """
    report = get_service().delete_asset(asset_id)
    if not report.was_known:
        raise HTTPException(status_code=404, detail=f"unknown asset id: {asset_id}")
    return asdict(report)


@app.get("/assets/{asset_id}")
def get_asset(asset_id: str) -> dict[str, object]:
    """Read-only detail for ``asset_id`` (asset_index + on-disk checks).

    404 when the asset is unknown or its relative_path fails the
    safety check. The response includes the index row plus
    ``file_exists`` / ``parsed_exists`` / ``captions_exists`` flags so
    the web drawer can show whether the derived artefacts are still on
    disk.
    """
    detail = get_service().get_asset_detail(asset_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"unknown asset id: {asset_id}")
    return detail


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the bundled single-page web UI from ``mm_asset_rag/web/``.

    The UI is a self-contained ``index.html`` (no external assets), so
    we serve it as a ``FileResponse`` with permissive caching headers:
    the page is tiny, version-bumps use the bundled query-string cache
    buster (``?v=<sha>``) and a hard-refresh repulls it.
    """
    return FileResponse(
        WEB_DIR / "index.html",
        media_type="text/html; charset=utf-8",
        headers={
            # Loosen CSP compared to the API responses: the UI ships
            # inline ``<script>`` and inline ``style``. ``script-src
            # 'unsafe-inline'`` is required because the bundle is one
            # file — splitting it would change the architecture. If
            # you fork the UI, convert to nonce/csp-hash and drop
            # ``'unsafe-inline'`` here.
            "Content-Security-Policy": (
                "default-src 'self'; "
                "img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-cache",
        },
    )


# ─── Static UI ───────────────────────────────────────────────────────────


def run() -> None:
    """Console-script entry point declared in ``pyproject.toml``.

    Runs the FastAPI app on ``127.0.0.1:8011`` via uvicorn. We bind to
    loopback only — the API has no auth and would be trivially
    exploitable on a routable interface. Deployments that need a public
    endpoint must add auth (e.g. a reverse proxy with a token) before
    changing the host.
    """
    import uvicorn

    uvicorn.run("mm_asset_rag.api:app", host="127.0.0.1", port=8011, log_level="info")
