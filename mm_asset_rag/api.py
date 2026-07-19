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
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

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
    safe_parsed_image_path,
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


# ─── Auth + host guard ───────────────────────────────────────────────────
#
# Two independent layers, both opt-in but with safe defaults:
#
# * ``TrustedHostMiddleware`` — locked to loopback (127.0.0.1, localhost) by
#   default so a malicious web page cannot reach the API via DNS rebinding
#   (the browser SOP preflight blocks JSON POST, but multipart ``/upload``
#   is a simple request and the rebinding trick can read GET responses).
#   Set ``MMRAG_TRUSTED_HOSTS`` to your public hostname(s) or ``*`` to relax.
#
# * bearer-token dependency — when ``MMRAG_API_TOKEN`` is set, the
#   destructive + write endpoints *and* the LLM-calling endpoints
#   (/answer, /chat, /chat/stream — they spend provider quota) require
#   ``Authorization: Bearer <token>`` or ``X-API-Key: <token>``. Unset =
#   zero-config loopback (no auth), so the bundled web UI works out of
#   the box on a developer's machine.
#
# Read endpoints (/search /assets* /tasks* /health /) stay open so the
# web UI's same-origin fetches keep working without a token; they don't
# mutate data or spend quota.

_DEFAULT_TRUSTED_HOSTS = ["127.0.0.1", "localhost", "[::1]"]


def _resolve_trusted_hosts() -> list[str]:
    raw = get_settings().mmrag_trusted_hosts
    if raw is None or raw.strip() == "":
        return _DEFAULT_TRUSTED_HOSTS
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    return hosts or _DEFAULT_TRUSTED_HOSTS


app.add_middleware(TrustedHostMiddleware, allowed_hosts=_resolve_trusted_hosts())


# Both header schemes resolve to the same token; either is accepted so the
# client can use whichever its HTTP library makes ergonomic. ``auto_error``
# is False so the dependency (not Starlette) controls the 401 — that lets an
# unset token keep the endpoint open (zero-config) rather than always 403.
_TOKEN_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_BEARER_HEADER = APIKeyHeader(name="Authorization", auto_error=False, scheme_name="bearer")


def _extract_bearer(value: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer <t>`` header."""
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def require_token(
    x_api_key: str | None = Depends(_TOKEN_HEADER),
    authorization: str | None = Depends(_BEARER_HEADER),
) -> None:
    """Dependency: reject the request unless it carries the configured token.

    When ``MMRAG_API_TOKEN`` is unset the dependency is a no-op — the
    loopback default stays zero-config. When set, a request missing the
    token (or carrying the wrong one) gets 401. Constant-time comparison
    avoids a timing oracle on the token.
    """
    expected = get_settings().mmrag_api_token
    # An empty / whitespace token is treated as "unset" (no auth) — a
    # misconfigured ``MMRAG_API_TOKEN=`` must not silently read as "enabled
    # with empty token" (which would let any request through via
    # compare_digest("", "")).
    if not expected or not expected.strip():
        return  # auth disabled — zero-config loopback default
    provided = x_api_key or _extract_bearer(authorization)
    if not provided or not _const_time_eq(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API token (set Authorization: Bearer <token> or X-API-Key)",
            headers={"WWW-Authenticate": 'Bearer realm="mmrag"'},
        )


def _const_time_eq(a: str, b: str) -> bool:
    """Constant-time string compare to avoid a token timing oracle."""
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())


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
    v2: bool = Field(
        default=False,
        description=(
            "Run the v2 regression set (83 cases, Chinese-primary, "
            "multi-dimensional) instead of v1. Default is v1."
        ),
    )


class ChatRequest(_RouteRequest):
    question: str = Field(..., min_length=1, max_length=2000)


class UploadEdit(BaseModel):
    preview_id: str
    title: str | None = None
    tags: list[str] | str | None = None
    description: str | None = None
    rejected: bool = False


# How much of an exception's string we surface in a streamed ``error``
# event. Streaming endpoints (``/chat/stream``, ``/tasks/{id}/stream``)
# send ``{"event": "error", "message": str(exc)}`` to the client; a raw
# ``str(exc)`` from ``requests`` / the OpenAI SDK commonly embeds the full
# request URL (and thus the ``OPENAI_BASE_URL`` / ``VLM_BASE_URL`` host,
# occasionally an inlined userinfo ``https://user:pass@host/...``) and a
# slice of the upstream response body. We strip URLs and cap the length so
# the error stays actionable without leaking provider topology or creds.
_STREAM_ERR_MAX_CHARS = 240
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
# ``requests.exceptions.ConnectionError`` / ``ReadTimeout`` render as e.g.
# ``HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out.``
# — the host appears quoted with no ``http(s)://`` prefix, so ``_URL_RE``
# alone misses it. Strip both the ``host='...'`` form and bare ``host:port``.
_HOST_QUOTED_RE = re.compile(r"\bhost='([^']+)'")
_HOSTPORT_RE = re.compile(r"\b([a-z0-9][a-z0-9.-]*):(\d{2,5})\b")


def _provider_hosts() -> set[str]:
    """Hosts parsed from the configured LLM/VLM base URLs.

    Used to scrub bare host mentions (``Connection to api.openai.com timed
    out``) that the URL/host:port regexes miss. Reading settings here is
    safe: ``_safe_stream_error`` is only called from streaming endpoints,
    not at import time.
    """
    from urllib.parse import urlparse

    hosts: set[str] = set()
    try:
        s = get_settings()
        for base in (s.openai_base_url, s.vlm_base_url, s.embedding_base_url):
            if base:
                h = urlparse(base).hostname
                if h:
                    hosts.add(h)
    except Exception:
        pass
    return hosts


def _safe_stream_error(exc: BaseException) -> str:
    """Render ``exc`` for a streamed error event without leaking URLs/hosts.

    Strips any ``http(s)://...`` substring (provider base URLs, inlined
    userinfo) and the ``host='...'`` / ``host:port`` forms that
    ``requests``' connection errors use, takes the first line, and caps the
    length. Non-ASCII is preserved (Chinese error text), but control
    characters are dropped.

    Additionally substitutes the configured LLM/VLM provider hosts (parsed
    from ``OPENAI_BASE_URL`` / ``VLM_BASE_URL``) wherever they appear —
    ``requests``' ``ConnectionError`` renders the host bare (``Connection
    to api.openai.com timed out``) with no URL prefix, so the regexes above
    miss it. Matching the exact configured host avoids the false-positive
    risk of a generic bare-host regex.
    """
    msg = str(exc)
    msg = _URL_RE.sub("<url>", msg)
    msg = _HOST_QUOTED_RE.sub("host=<host>", msg)
    msg = _HOSTPORT_RE.sub("<host>:<port>", msg)
    for host in _provider_hosts():
        if host:
            msg = msg.replace(host, "<host>")
    first_line = msg.splitlines()[0] if msg else ""
    # Strip control chars except tab/newline (already single-line).
    first_line = "".join(c for c in first_line if c >= " " or c == "\t")
    if len(first_line) > _STREAM_ERR_MAX_CHARS:
        first_line = first_line[:_STREAM_ERR_MAX_CHARS] + "…"
    return first_line or f"{type(exc).__name__}: <no message>"


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
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        "model": get_settings().openai_model or "",
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    hits = _run_search(request)
    return {"query": request.query, "mode": request.mode, "hits": [h.__dict__ for h in hits]}


@app.post("/answer")
def answer(
    request: AnswerRequest,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    return answer_question(request.question, top_k=request.top_k)


@app.post("/eval")
def eval_endpoint(
    request: EvalRequest,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    if request.v2:
        from .evaluation_v2 import run_eval_v2

        results = run_eval_v2(top_k=request.top_k)
        # ``V2Result`` mirrors v1's ``EvalResult`` (same fields:
        # query / expected_asset_ids / actual_asset_ids / hit / rank /
        # group), so ``asdict`` produces the same row shape the v1
        # branch returns; the only addition is a ``version`` tag so
        # clients can tell which set ran.
        return {"results": [asdict(r) for r in results], "version": "v2"}
    return {"results": [r.__dict__ for r in run_eval(top_k=request.top_k)]}


@app.post("/chat")
def chat(
    request: ChatRequest,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
    """One-call: retrieve + grounded LLM answer in a single response."""
    hits = _run_search(request)
    answer = answer_question(request.question, top_k=request.top_k, hits=hits)
    return {
        "question": request.question,
        "answer": answer,
        "sources": [h.__dict__ for h in hits],
    }


# Sentinel pushed onto the bridge queue to signal "no more items" (either the
# sync generator finished or the worker thread exited, normally or via error).
_STREAM_DONE: object = object()


async def _iter_sync_in_thread(factory, *args, **kwargs) -> asyncio.Queue:
    """Bridge a *sync* generator into the event loop as an ``async`` queue.

    ``factory(*args, **kwargs)`` returns a fresh sync generator (called inside
    the worker thread so any blocking setup — e.g. the OpenAI SDK's first
    HTTP read — runs off the loop). Each yielded item is pushed onto a
    ``queue.Queue``; the consumer awaits ``queue.get`` via ``to_thread`` so it
    stays responsive to client disconnects (``asyncio.CancelledError``).

    This is the fix for the previous "buffer-all" ``chat_stream``: it ran
    ``list(stream_answer_chunks(...))`` in one ``to_thread`` call, so the
    client saw its first token only after the *entire* LLM response finished
    — defeating the NDJSON streaming contract and burning LLM tokens after a
    mid-stream client disconnect (the sync ``list(...)`` couldn't be cancelled).

    Returns the queue; the caller iterates it until it sees ``_STREAM_DONE``.
    On cancellation the caller drops the queue and the orphaned worker exits
    on its next ``put``. The queue is **bounded** (maxsize 64): a slow client
    (or one that stopped reading) can't let a fast LLM stream buffer an
    unbounded response in memory. When the queue is full the worker waits
    up to 0.5s per put, re-checking ``stop`` so a client disconnect still
    unblocks it promptly rather than blocking forever. The worker is a
    daemon thread, so a runaway producer that ignores ``stop`` is reaped
    at process exit rather than pinning the loop forever.
    """
    out: queue.Queue = queue.Queue(maxsize=64)
    stop = threading.Event()

    def _put_terminal(item) -> None:
        """Put the sentinel / error without ever blocking.

        On a client disconnect the consumer stops draining, so the queue
        may be full. A blocking ``put`` here would hang the daemon thread
        forever (and pin the 64-item buffer until process exit). Drop
        silently instead — nobody is listening anymore.
        """
        with suppress(queue.Full):
            out.put_nowait(item)

    def _worker():
        try:
            for item in factory(*args, **kwargs):
                if stop.is_set():
                    return
                # Bounded put: if the consumer fell behind, wait but keep
                # checking stop so a client disconnect unblocks us.
                while not stop.is_set():
                    try:
                        out.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    return  # stop was set during the wait
        except BaseException as exc:  # surface producer errors to the consumer
            _put_terminal(exc)
        finally:
            _put_terminal(_STREAM_DONE)

    t = threading.Thread(target=_worker, daemon=True, name="chat-stream-producer")
    t.start()

    # Give the caller a handle to signal cancellation. Stashed on the queue
    # object so the gen() closure below can reach it without a closure var.
    out.stop = stop  # type: ignore[attr-defined]
    out.thread = t  # type: ignore[attr-defined]
    return out


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
            )
            yield (
                json.dumps(
                    {"event": "sources", "sources": [h.__dict__ for h in hits]},
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
            bridge = await _iter_sync_in_thread(stream_answer_chunks, request.question, hits)
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
    load_env()
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
    force: bool = Query(False, description="Clear parsed/<id>/ cache before re-running"),
    failed_only: bool = Query(
        False,
        description="Only re-run assets whose previous status was failed or skipped",
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
def delete_asset(
    asset_id: str,
    _auth: None = Depends(require_token),
) -> dict[str, object]:
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


@app.get("/parsed-image/{asset_id}/{filename}")
def get_parsed_image(asset_id: str, filename: str) -> FileResponse:
    """Serve an image extracted from a parsed PDF (``parsed/<id>/images/``).

    Used by the web UI ``<img src>`` to render figure thumbnails attached
    to text hits (tier-1 multimodal: the figure path rides in the hit
    payload). Validation is delegated to :func:`paths.safe_parsed_image_path`
    so the endpoint and the tier-3 answer image loader apply identical
    traversal guards.
    """
    candidate = safe_parsed_image_path(asset_id, filename)
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

    Runs the FastAPI app on ``127.0.0.1:8011`` via uvicorn. We bind to
    loopback only — the API is unauthenticated by default (set
    ``MMRAG_API_TOKEN`` to guard destructive + LLM-quota endpoints, and
    ``MMRAG_TRUSTED_HOSTS`` to widen the Host allow-list beyond loopback).
    """
    import sys

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "mmrag-api — start the mm-asset-rag HTTP API + web UI.\n\n"
            "  mmrag-api            # serve on http://127.0.0.1:8011\n\n"
            "No CLI flags; configure via env vars (see .env.example / "
            "docs/configuration.md). Bind is loopback-only by default; set "
            "MMRAG_TRUSTED_HOSTS + MMRAG_API_TOKEN before exposing publicly."
        )
        return
    import uvicorn

    uvicorn.run("mm_asset_rag.api:app", host="127.0.0.1", port=8011, log_level="info")
