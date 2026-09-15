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
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

from . import asset_index
from .__init__ import __version__
from .answer import answer_question, stream_answer_chunks
from .api_models import (
    AnswerRequest,
    ChatRequest,
    EvalRequest,
    SearchRequest,
    UploadConfirmRequest,
    _RouteRequest,
)
from .api_models import UploadEdit as UploadEdit
from .api_security import _resolve_trusted_hosts, require_token
from .api_streaming import (
    _STREAM_DONE,
    _iter_sync_in_thread,
    _safe_stream_error,
)
from .api_streaming import _STREAM_ERR_MAX_CHARS as _STREAM_ERR_MAX_CHARS
from .backends.qdrant.client import get_qdrant_client
from .evaluation_service import EvaluationCommand, get_evaluation_service
from .observability import runtime_metrics
from .paths import (
    get_assets_dir,
    get_documents_jsonl,
    get_preview_cache_dir,
    physical_cache_id,
    safe_parsed_image_path,
)
from .search_service import dispatch_search, get_search_service
from .service import ParseOptions, get_service
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
    # file. If the process is killed before this runs, the next startup
    # tolerates a stale local Qdrant lock.
    with suppress(Exception):
        get_qdrant_client().close()


app = FastAPI(
    title="mm-asset-rag",
    version=__version__,
    description="Multimodal asset RAG: PDF + image parsing, hybrid retrieval, grounded answers.",
    lifespan=lifespan,
)


# ─── Auth + host guard ───────────────────────────────────────────────────
# TrustedHostMiddleware is locked to loopback by default so a malicious web
# page can't read GET responses via DNS rebinding (multipart /upload is a
# simple request, no SOP preflight). Relax via MMRAG_TRUSTED_HOSTS.
# When MMRAG_API_TOKEN is set, write + destructive + LLM-calling endpoints
# (/answer, /chat … — they spend provider quota) require a bearer / X-API-Key
# token; unset = zero-config loopback. Read endpoints stay open.

app.add_middleware(TrustedHostMiddleware, allowed_hosts=_resolve_trusted_hosts())


# ─── Request body size limit ─────────────────────────────────────────────
#
# Starlette streams multipart bodies into a ``SpooledTemporaryFile`` *before*
# the route handler runs, so the in-handler ``upload_max_*`` byte checks
# only gate the copy into ``incoming_dir`` — a 50 GB POST would still fill
# ``/tmp`` before our 413 fires. This middleware wraps ``receive`` so the
# body is rejected as soon as the cumulative byte count crosses the
# configured cap, before Starlette spools it to disk.
#
# The cap is the per-batch upload limit (``upload_max_batch_bytes``, default
# 200 MiB) — large enough that ordinary JSON requests (search/answer/chat,
# tens of KB) sail through, small enough to bound a malicious upload. The
# limit applies to every request body; NDJSON/JSON payloads are tiny so this
# never rejects legitimate traffic.


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than ``upload_max_batch_bytes`` (HTTP 413).

    Wraps the ASGI ``receive`` callable and accumulates ``http.request``
    body chunks; on overflow it stops pulling and returns a 413 response.
    The cap is read from :class:`Settings` on every request (not fixed at
    middleware construction) so a test that lowers ``UPLOAD_MAX_BATCH_BYTES``
    sees the new limit immediately. Streaming responses are unaffected
    (the limit is on the *request* body).
    """

    async def dispatch(self, request: StarletteRequest, call_next):
        max_bytes = int(get_settings().upload_max_batch_bytes)
        # Only bound requests that actually carry a body. GET / HEAD / OPTIONS
        # with no body pass through untouched.
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        sent = 0
        overflow = False
        receive = request.receive

        async def bounded_receive():
            nonlocal sent, overflow
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                sent += len(body)
                if sent > max_bytes:
                    overflow = True
                    # Pretend the body is now complete so Starlette stops
                    # reading and our 413 can fire.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        request._receive = bounded_receive  # type: ignore[attr-defined]
        response = await call_next(request)
        if overflow:
            return Response(
                content=json.dumps({"detail": f"request body exceeds {max_bytes} bytes"}),
                status_code=413,
                media_type="application/json",
            )
        return response


app.add_middleware(_BodySizeLimitMiddleware)


# ─── Service layer (task scheduling + history) ─────────────────────────
#
# All background work, persistence, and task queries live in
# ``mm_asset_rag.service``. The FastAPI app stays a thin route layer.

from .service import TaskRecord  # noqa: E402, F401

# ─── Static web UI ────────────────────────────────────────────────────────


WEB_DIR = Path(__file__).resolve().parent / "web"


def _resolve_cases_path(value: str | None) -> str | Path | None:
    """Resolve a validated ``cases_path`` to an on-disk path inside
    ``eval_cases/`` or the repo ``examples/`` dir. ``None`` passes through
    (bundled default / ``EVAL_CASES_PATH`` are resolved server-side and
    trusted). Translates the shared resolver's ``ValueError`` /
    ``FileNotFoundError`` into HTTP 422 so a bad path surfaces as a proper
    client error instead of a 500 from ``load_cases``.
    """
    from .paths import resolve_cases_path

    try:
        resolved = resolve_cases_path(value)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return None if resolved is None else str(resolved)


def _main_text(request: _RouteRequest) -> str:
    """Return the main text field of ``SearchRequest`` / ``ChatRequest``.

    ``SearchRequest.query`` and ``ChatRequest.question`` are the only
    field-name differences between the two; everything else (mode,
    image_path, top_k) lives on ``_RouteRequest``. Centralising the
    dispatch + HTTP-error translation here means a future ``QueryRequest``
    just inherits ``_RouteRequest`` and adds its own text field.
    """
    # Pydantic's ``__getattribute__`` makes ``getattr(..., default)``
    # itself raise (it bypasses ``__pydantic_extra__`` lookup), so we
    # branch on declared fields instead.
    if "query" in type(request).model_fields:
        return request.query
    return request.question


def _run_search(request: _RouteRequest) -> list[object]:
    """Run ``dispatch_search`` for either endpoint with a single
    ``ValueError → HTTPException(400)`` translation.
    """
    try:
        return dispatch_search(
            query=_main_text(request),
            mode=request.mode,
            image_path=request.image_path,
            top_k=request.top_k,
            collection=request.collection,
            metadata_filter=request.metadata_filter,
            principal=request.principal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _without_asset_id(value: object) -> object:
    """Keep the transitional SearchHit cache key out of public transport."""
    if isinstance(value, dict):
        return {key: _without_asset_id(item) for key, item in value.items() if key != "asset_id"}
    if isinstance(value, list):
        return [_without_asset_id(item) for item in value]
    return value


def _serialize_hit(hit: object) -> dict[str, object]:
    """Render the stable document/chunk retrieval contract."""
    metadata = getattr(hit, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "document_id": metadata.get("document_id"),
        "chunk_id": metadata.get("chunk_id"),
        "title": getattr(hit, "title", ""),
        "source_type": getattr(hit, "source_type", ""),
        "source_path": getattr(hit, "source_path", ""),
        "evidence": getattr(hit, "evidence", ""),
        "score": getattr(hit, "score", 0.0),
        "routes": metadata.get("routes", [getattr(hit, "route", "")]),
        "page": metadata.get("page"),
        "parser": metadata.get("parser") or metadata.get("provider"),
        "images": _without_asset_id(getattr(hit, "images", []) or metadata.get("images") or []),
    }


def _serialize_document(record: asset_index.DocumentRecord) -> dict[str, object]:
    """Render public document identity without its server-side policy."""
    return {
        "document_id": record.document.document_id,
        "title": record.document.title,
        "source": record.document.source.to_record(),
        "asset": record.asset.to_record(),
    }


@dataclass(frozen=True)
class _DocumentAccessContext:
    """Validated policy context required by every document read route."""

    collection: str
    principal: str
    metadata_filter: dict[str, object]


def _document_access_context(
    collection: str = Query(..., min_length=1, max_length=200),
    principal: str = Query(..., min_length=1, max_length=200),
    metadata_filter: str | None = Query(None, max_length=4096),
) -> _DocumentAccessContext:
    """Parse the query context used to enforce persisted access policies."""
    if metadata_filter is None or not metadata_filter.strip():
        parsed: dict[str, object] = {}
    else:
        try:
            value = json.loads(metadata_filter)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422, detail="metadata_filter must be a JSON object"
            ) from exc
        if not isinstance(value, dict):
            raise HTTPException(status_code=422, detail="metadata_filter must be a JSON object")
        parsed = value
    return _DocumentAccessContext(
        collection=collection,
        principal=principal,
        metadata_filter=parsed,
    )


def _record_is_visible(record: asset_index.DocumentRecord, context: _DocumentAccessContext) -> bool:
    """Return whether a persisted document is visible to this request."""
    policy = record.access_policy
    return (
        policy.collection == context.collection
        and policy.allows(context.principal)
        and all(policy.metadata.get(key) == value for key, value in context.metadata_filter.items())
    )


# ─── Endpoints ──────────────────────────────────────────────────────────


@app.get("/health")
def health(
    deep: bool = Query(False, description="probe LLM/embedder config completeness"),
) -> dict[str, object]:
    """Liveness + index state.

    ``text_index_exists`` / ``image_index_exists`` probe the actual Qdrant
    collections (previously they checked the stale ``indexes/text``
    LlamaIndex-era directory, which reported false even when the index was
    healthy). Both degrade to ``False`` if the Qdrant client can't answer
    (e.g. local-mode lock held by another process) — health must never 500.

    ``?deep=true`` adds config-completeness probes: ``llm_configured``
    (MODEL_* or LLM_* triple complete) and ``embedder_configured``
    (embedding creds resolvable). These don't make an outbound LLM call
    (no quota spend); they only check the configured triples so a
    docker/编排 healthcheck can tell "will /answer work" from /health.
    """
    assets_dir = get_assets_dir()
    asset_files = [p for p in assets_dir.rglob("*") if p.is_file()] if assets_dir.exists() else []
    payload: dict[str, object] = {
        "status": "ok",
        "version": __version__,
        "files": len(asset_files),
        "documents_jsonl_exists": get_documents_jsonl().exists(),
        "text_index_exists": _qdrant_collection_alive("text"),
        "image_index_exists": _qdrant_collection_alive("image"),
        "vector_backend": "qdrant",
        "model": get_settings().llm_model or "",
    }
    if deep:
        s = get_settings()
        # llm_creds → (base_url, api_key, model); text_embedding_creds →
        # (api_key, base_url, model) — different orders, name carefully.
        lb, lk, lm = s.llm_creds
        ek, eb, _em = s.text_embedding_creds
        payload["llm_configured"] = bool(lb and lk and lm)
        payload["embedder_configured"] = bool(ek and eb and _em)
    return payload


def _qdrant_collection_alive(kind: str) -> bool:
    """True iff any Qdrant collection for ``kind`` ('text'/'image') exists.

    Resolves collections from the live server (via
    ``_existing_collections_for``), not the module's active-cache. A cold-start
    API process that never ingested would otherwise see ``text_collection()``
    fall back to the bare base name ``multimodal_text`` (the real collection is
    ``multimodal_text_<dim>d``) and ``collection_exists`` would wrongly return
    False — reporting the index as missing on ``/health`` right after boot.

    Swallows every error: /health must stay 200 even when Qdrant local-mode
    is locked by another process, the server is unreachable, or no
    collection has been created yet.
    """
    try:
        from .backends.qdrant.client import get_qdrant_client
        from .backends.qdrant.collections import (
            IMAGE_COLLECTION_BASE,
            TEXT_COLLECTION_BASE,
            _existing_collections_for,
        )

        client = get_qdrant_client()
        base = TEXT_COLLECTION_BASE if kind == "text" else IMAGE_COLLECTION_BASE
        return bool(_existing_collections_for(client, base))
    except Exception:
        return False


@app.get("/metrics")
def metrics() -> dict[str, object]:
    """Return bounded process-local retrieval and refusal counters."""
    return runtime_metrics.snapshot()


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, object]:
    hits = await asyncio.to_thread(_run_search, request)
    return {"query": request.query, "mode": request.mode, "hits": [_serialize_hit(h) for h in hits]}


@app.post("/answer")
async def answer(
    request: AnswerRequest,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    return await asyncio.to_thread(
        answer_question,
        request.question,
        top_k=request.top_k,
        search_service=get_search_service(),
        collection=request.collection,
        metadata_filter=request.metadata_filter,
        principal=request.principal,
        min_confidence=request.min_confidence,
    )


@app.post("/eval")
async def eval_endpoint(
    request: EvalRequest,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    cases_path = _resolve_cases_path(request.cases_path)
    return await asyncio.to_thread(
        get_evaluation_service().execute,
        EvaluationCommand(
            top_k=request.top_k,
            cases_path=cases_path,
            collection=request.collection,
            metadata_filter=request.metadata_filter,
            principal=request.principal,
            v2=request.v2,
            answer_quality=request.answer_quality,
        ),
    )


@app.post("/chat")
async def chat(
    request: ChatRequest,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    """One-call: retrieve + grounded LLM answer in a single response."""
    hits = await asyncio.to_thread(_run_search, request)
    answer = await asyncio.to_thread(
        answer_question,
        request.question,
        top_k=request.top_k,
        hits=hits,
        collection=request.collection,
        metadata_filter=request.metadata_filter,
        principal=request.principal,
        min_confidence=request.min_confidence,
    )
    return {
        "question": request.question,
        "answer": answer,
        "sources": answer.get("sources", []),
    }


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    _auth: None = Depends(require_token),
) -> StreamingResponse:
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
        # Bridge object for the producer thread; created lazily so the
        # ``finally`` below can tell whether the producer ever started.
        bridge: asyncio.Queue | None = None
        try:
            hits = await asyncio.to_thread(
                dispatch_search,
                query=request.question,
                mode=request.mode,
                image_path=request.image_path,
                top_k=request.top_k,
                collection=request.collection,
                metadata_filter=request.metadata_filter,
                principal=request.principal,
            )
            yield (
                json.dumps(
                    {
                        "event": "sources",
                        "sources": [
                            _serialize_hit(h) for h in hits if h.score >= request.min_confidence
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            # ``stream_answer_chunks`` is a sync generator (the OpenAI SDK
            # yields its stream chunks synchronously). Bridge it onto the
            # event loop with a thread + queue so each token is yielded as
            # soon as the SDK emits it (true streaming, not buffer-all) and
            # a client disconnect cancels this coroutine — we then signal
            # the worker to stop instead of letting it run the full LLM
            # response to completion.
            bridge = await _iter_sync_in_thread(
                stream_answer_chunks,
                request.question,
                hits,
                min_confidence=request.min_confidence,
            )
            while True:
                item = await asyncio.to_thread(bridge.get)
                if item is _STREAM_DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield json.dumps({"event": "token", "text": item}, ensure_ascii=False) + "\n"
            yield json.dumps({"event": "done"}, ensure_ascii=False) + "\n"
        except asyncio.CancelledError:
            # Client disconnected mid-stream. Tell the producer thread to
            # stop (it checks ``stop`` between yields); let the loop's
            # default cancellation handling proceed by re-raising so
            # Starlette tears down the response cleanly.
            if bridge is not None:
                bridge.stop.set()  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            yield (
                json.dumps(
                    {"event": "error", "message": _safe_stream_error(exc)}, ensure_ascii=False
                )
                + "\n"
            )

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ─── Upload + background parse ───────────────────────────────────────────


@app.post("/upload/preview")
async def upload_preview(
    files: list[UploadFile] = File(...),
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    """Stage uploaded files and return editable metadata previews.

    This endpoint does not parse, embed or index. It streams multipart bytes
    into a short-lived incoming directory, lets ``UploadPipeline`` sniff and
    optionally VLM-tag them, then returns preview cards for the web UI.
    """
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    settings = get_settings()
    # Cap the file count up front — each previewed file can trigger a VLM
    # auto-meta call, so an unbounded batch is a quota-burn vector. The
    # byte caps alone don't bound the number of (small) files.
    if len(files) > settings.upload_max_files:
        raise HTTPException(
            status_code=413,
            detail=(
                f"too many files: {len(files)} > upload_max_files ({settings.upload_max_files})"
            ),
        )
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

        # ``UploadPipeline.preview`` does sync I/O (shutil copy, sha256,
        # sniff via PyMuPDF/Pillow, optional VLM HTTP calls). Running it
        # inline in this ``async def`` would freeze the event loop for the
        # whole batch — /health, other /search, and SSE heartbeats all stall.
        # Hand it to the default thread pool so the loop stays responsive.
        previews = await asyncio.to_thread(get_pipeline().preview, staged)
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
def upload_confirm(
    request: UploadConfirmRequest,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    """Apply user edits and kick off parse + index for confirmed previews."""
    edits = [
        UserEdits(
            preview_id=e.preview_id,
            title=e.title,
            tags=e.tags if isinstance(e.tags, list) else None if e.tags is None else [e.tags],
            description=e.description,
            document_id=e.document_id,
            collection=e.collection,
            allowed_principals=e.allowed_principals,
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
            yield (
                json.dumps(
                    {"event": "error", "message": _safe_stream_error(exc)}, ensure_ascii=False
                )
                + "\n"
            )

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
    force: bool = Query(False, description="Clear targeted document caches before re-running"),
    failed_only: bool = Query(
        False,
        description="Only re-run documents whose previous status was failed or skipped",
    ),
    _auth: None = Depends(require_token),
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


@app.post("/tasks/{task_id}/cancel")
def cancel_task(
    task_id: str,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    """Request cooperative cancellation of a running task.

    Sets a per-task stop flag the worker checks between assets; the task
    is marked ``cancelled`` (terminal). A task that is already terminal
    (done/failed/interrupted/cancelled) is returned unchanged. Cancellation
    is cooperative — a task deep in parsing one large asset finishes that
    asset first. 404 for an unknown task_id.
    """
    try:
        rec = get_service().cancel_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "task_id": rec.task_id,
        "status": rec.status,
        "finished_at": rec.finished_at,
    }


@app.get("/documents")
def list_documents(
    context: _DocumentAccessContext = Depends(_document_access_context),
) -> dict[str, object]:
    """Return current documents visible to the supplied context."""
    latest: dict[str, asset_index.DocumentRecord] = {}
    for record in asset_index.load_records():
        if not _record_is_visible(record, context):
            continue
        latest[record.document.document_id] = record
    return {"documents": [_serialize_document(latest[key]) for key in sorted(latest)]}


@app.get("/documents/{document_id}")
def get_document(
    document_id: str,
    context: _DocumentAccessContext = Depends(_document_access_context),
) -> dict[str, object]:
    """Return one visible current document without exposing its policy."""
    records = [
        record
        for record in asset_index.load_records()
        if record.document.document_id == document_id and _record_is_visible(record, context)
    ]
    if not records:
        raise HTTPException(status_code=404, detail=f"unknown document {document_id}")
    latest = records[-1]
    return {
        "document_id": latest.document.document_id,
        "title": latest.document.title,
        "source": latest.document.source.to_record(),
        "asset": latest.asset.to_record(),
    }


@app.get("/parsed-image/{document_id}/{filename}")
def get_parsed_image(
    document_id: str,
    filename: str,
    context: _DocumentAccessContext = Depends(_document_access_context),
) -> FileResponse:
    """Serve an image extracted from one current document.

    Used by the web UI ``<img src>`` to render figure thumbnails attached
    to text hits (tier-1 multimodal: the figure path rides in the hit
    payload). Validation is delegated to :func:`paths.safe_parsed_image_path`
    so the endpoint and the tier-3 answer image loader apply identical
    traversal guards.
    """
    record = next(
        (row for row in asset_index.load_records() if row.document.document_id == document_id),
        None,
    )
    if record is None or not _record_is_visible(record, context):
        raise HTTPException(status_code=404, detail="not found")
    # Parsed-image storage still uses a transient physical cache key. It is
    # resolved server-side from the asset's physical path and never appears
    # in the public URL or response payload.
    candidate = safe_parsed_image_path(physical_cache_id(record.asset.relative_path), filename)
    if candidate is None:
        raise HTTPException(status_code=404, detail="not found")
    suffix = candidate.suffix.lower().lstrip(".")
    return FileResponse(candidate, media_type=f"image/{suffix}")


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

    Runs the FastAPI app at the address configured by ``MMRAG_API_HOST`` /
    ``MMRAG_API_PORT``. The package default is loopback; when binding to the
    LAN, set ``MMRAG_TRUSTED_HOSTS`` and an API token deliberately.
    """
    import sys

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "mmrag-api — start the mm-asset-rag HTTP API + web UI.\n\n"
            "  mmrag-api            # serve on MMRAG_API_HOST:MMRAG_API_PORT\n\n"
            "No CLI flags; configure via env vars (see .env.example / "
            "docs/configuration.md). The default bind is 127.0.0.1:8011; set "
            "MMRAG_API_HOST=0.0.0.0, MMRAG_TRUSTED_HOSTS and MMRAG_API_TOKEN "
            "before exposing it beyond the local machine."
        )
        return
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "mm_asset_rag.api:app",
        host=settings.mmrag_api_host,
        port=settings.mmrag_api_port,
        log_level="info",
    )
