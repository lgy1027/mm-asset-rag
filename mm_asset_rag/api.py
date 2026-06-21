"""HTTP API for mm-asset-rag.

Endpoints
---------
GET  /health                       liveness + index state
POST /upload    (multipart)         upload files, kick off background parse+index
GET  /tasks/{id}                    poll background task status
POST /search   (json)               retrieval only
POST /answer   (json)               grounded LLM answer
POST /eval     (json)               run mmrag eval
POST /chat     (json)               one-call hybrid_search + answer
GET  /                              serves the bundled single-file web UI (index.html)
GET  /static/{path}                other static assets (none today, but reserved)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import answer_question, stream_answer_chunks
from .assets import load_assets
from .cli import command_index, command_parse
from .config import load_env
from .evaluation import run_eval
from .paths import get_assets_dir, get_documents_jsonl, get_indexes_dir, get_text_index_dir
from .qdrant_store import (
    get_qdrant_client,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .retrieval import hybrid_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Graceful shutdown: close the qdrant client so it removes its .lock
    # file. If the process is killed before this runs, the next startup will
    # also tolerate a stale .lock (see qdrant_store._clean_stale_lock).
    try:
        get_qdrant_client().close()
    except Exception:
        pass


app = FastAPI(
    title="mm-asset-rag",
    version="0.1.0",
    description="Multimodal asset RAG: PDF + image parsing, hybrid retrieval, grounded answers.",
    lifespan=lifespan,
)


# ─── Background task bookkeeping ──────────────────────────────────────────


@dataclass
class TaskRecord:
    task_id: str
    kind: str  # "parse" or "ingest" (parse + index)
    status: str = "pending"  # pending | running | done | failed
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    current: str = ""
    error: str | None = None
    uploaded_files: list[str] = field(default_factory=list)


_TASKS: dict[str, TaskRecord] = {}
_TASKS_LOCK = threading.Lock()


def _new_task(kind: str, total: int, uploaded: list[str] | None = None) -> TaskRecord:
    rec = TaskRecord(task_id=uuid.uuid4().hex[:12], kind=kind, total=total,
                     uploaded_files=uploaded or [])
    with _TASKS_LOCK:
        _TASKS[rec.task_id] = rec
    return rec


def _patch(rec: TaskRecord, **fields: Any) -> None:
    with _TASKS_LOCK:
        for k, v in fields.items():
            setattr(rec, k, v)


@dataclass
class ParseOptions:
    """Per-task parse configuration. Comes from the ``/upload`` form."""

    pdf_parser: str = "auto"
    enable_ocr: bool = False
    enable_vlm: bool = False
    image_provider: str = "lite"
    only_uploaded: bool = False
    uploaded_files: list[str] = field(default_factory=list)  # relative to assets_dir


def _run_parse_task(rec: TaskRecord, options: ParseOptions) -> None:
    """Run mmrag parse in a worker thread, updating rec as it progresses.

    When ``options.only_uploaded`` is True, only the files in
    ``options.uploaded_files`` are parsed (manifest is bypassed). Otherwise the
    full ``asset_manifest.json`` is loaded.
    """
    load_env()
    from .image_parser import parse_image
    from .pdf_parser import parse_pdf
    from .paths import get_parsed_dir

    if options.only_uploaded:
        # Construct ephemeral Asset objects for the just-uploaded files; no
        # manifest lookup needed and the rest of the catalog stays untouched.
        from .assets import Asset

        assets_dir = get_assets_dir()
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
        _patch(rec, current="loading assets")
        assets = load_assets()

    if not assets:
        _patch(
            rec,
            status="done",
            current="no assets to parse",
            finished_at=time.time(),
        )
        return

    _patch(rec, total=len(assets), current=f"parsing {len(assets)} asset(s)")

    failed = 0
    skipped = 0
    parsed = 0
    target = get_documents_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    for i, asset in enumerate(assets, start=1):
        try:
            # Cache hit: skip if raw.jsonl already exists.
            raw_path = get_parsed_dir() / asset.asset_id / "raw.jsonl"
            if raw_path.exists() and raw_path.stat().st_size > 0:
                skipped += 1
                _patch(rec, processed=i, current=f"skip cached: {asset.asset_id}")
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
                _patch(rec, processed=i, current=f"error {asset.asset_id}: {exc}")
                continue
            # Append so progress is visible and survives restarts.
            with target.open("a", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
            parsed += 1
            _patch(
                rec,
                processed=i,
                current=f"parsed {asset.asset_id} ({len(docs)} doc)",
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            _patch(rec, processed=i, current=f"error {asset.asset_id}: {exc}")

    status = "done" if failed == 0 and skipped + parsed == len(assets) else "partial"
    _patch(
        rec,
        status=status,
        finished_at=time.time(),
        current=f"parse {status}: parsed={parsed} skipped={skipped} failed={failed}",
    )


def _run_ingest_task(rec: TaskRecord, options: ParseOptions) -> None:
    """Parse + index in sequence on a background thread."""
    _run_parse_task(rec, options)
    if rec.status == "failed":
        _patch(rec, finished_at=time.time())
        return
    parse_status = rec.status
    _patch(rec, current="building vector index")
    try:
        command_index(
            argparse.Namespace(backend="qdrant", image_provider=options.image_provider)
        )
        _patch(rec, current="index built", status=parse_status, finished_at=time.time())
    except SystemExit:
        # command_index calls parser.exit on missing args; treat as success.
        _patch(rec, current="index built", status=parse_status, finished_at=time.time())
    except Exception as exc:  # noqa: BLE001
        _patch(
            rec,
            current=f"index failed: {exc}",
            error=str(exc),
            status="failed",
            finished_at=time.time(),
        )


# ─── Static web UI ────────────────────────────────────────────────────────


WEB_DIR = Path(__file__).resolve().parent / "web"


# ─── Request / response models ───────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(text|text-to-image|image-to-image|hybrid)$")
    image_path: str | None = None
    top_k: int = 5


class AnswerRequest(BaseModel):
    question: str
    top_k: int = 5


class EvalRequest(BaseModel):
    top_k: int = 5


class ChatRequest(BaseModel):
    question: str
    mode: str = Field(default="hybrid", pattern="^(text|text-to-image|image-to-image|hybrid)$")
    image_path: str | None = None
    top_k: int = 5


# ─── Endpoints ──────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, object]:
    load_env()
    return {
        "status": "ok",
        "assets": len(load_assets()),
        "documents_jsonl_exists": get_documents_jsonl().exists(),
        "text_index_exists": get_text_index_dir().exists(),
        "image_index_exists": (get_indexes_dir() / "qdrant").exists(),
        "vector_backend": "qdrant",
        "model": os.environ.get("OPENAI_MODEL", ""),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    if request.mode == "text":
        hits = qdrant_text_search(request.query, top_k=request.top_k)
    elif request.mode == "text-to-image":
        hits = qdrant_text_to_image_search(request.query, top_k=request.top_k)
    elif request.mode == "image-to-image":
        if not request.image_path:
            raise HTTPException(status_code=400, detail="image_path required for image-to-image")
        hits = qdrant_image_to_image_search(Path(request.image_path), top_k=request.top_k)
    else:
        hits = hybrid_search(
            request.query,
            image_path=Path(request.image_path) if request.image_path else None,
            top_k=request.top_k,
        )
    return {"query": request.query, "mode": request.mode, "hits": [h.__dict__ for h in hits]}


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, object]:
    return answer_question(request.question, top_k=request.top_k)


@app.post("/eval")
def eval_endpoint(request: EvalRequest) -> dict[str, object]:
    return {"results": [r.__dict__ for r in run_eval(top_k=request.top_k)]}


def _retrieve_for_chat(request: ChatRequest) -> list:
    """Shared retrieval for /chat and /chat/stream. Raises 400 if image mode lacks a path."""
    if request.mode == "text":
        return qdrant_text_search(request.question, top_k=request.top_k)
    if request.mode == "text-to-image":
        return qdrant_text_to_image_search(request.question, top_k=request.top_k)
    if request.mode == "image-to-image":
        if not request.image_path:
            raise HTTPException(status_code=400, detail="image_path required for image-to-image")
        return qdrant_image_to_image_search(Path(request.image_path), top_k=request.top_k)
    return hybrid_search(
        request.question,
        image_path=Path(request.image_path) if request.image_path else None,
        top_k=request.top_k,
    )


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    """One-call: retrieve + grounded LLM answer in a single response."""
    hits = _retrieve_for_chat(request)
    answer = answer_question(request.question, top_k=request.top_k, hits=hits)
    return {
        "question": request.question,
        "answer": answer,
        "sources": [h.__dict__ for h in hits],
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """NDJSON stream of the chat answer.

    Each line is a JSON object:

    - ``{"event": "sources", "sources": [...]}``  once, up front
    - ``{"event": "token", "text": "..."}``        one per LLM token
    - ``{"event": "done"}``                        exactly once at the end
    """

    def gen():
        try:
            hits = _retrieve_for_chat(request)
            yield json.dumps(
                {"event": "sources", "sources": [h.__dict__ for h in hits]},
                ensure_ascii=False,
            ) + "\n"
            for chunk in stream_answer_chunks(request.question, hits):
                yield json.dumps({"event": "token", "text": chunk}, ensure_ascii=False) + "\n"
            yield json.dumps({"event": "done"}, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001
            yield json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ─── Upload + background parse ───────────────────────────────────────────


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    auto_index: str | None = Form(default=None),
    pdf_parser: str | None = Form(default=None),
    enable_ocr: str | None = Form(default=None),
    enable_vlm: str | None = Form(default=None),
    image_provider: str | None = Form(default=None),
) -> dict[str, object]:
    """Accept a batch of files, copy them into ``$MM_ASSET_RAG_HOME/assets/``,
    then run a background parse (+ index if ``auto_index``) restricted to the
    uploaded files only.

    Form fields are accepted as strings and parsed here so they can also be
    unset (None) and fall back to .env. Resolution order:

    1. The form value, if provided by the client.
    2. The matching .env variable (``AUTO_INDEX`` / ``PDF_PARSER`` / etc).
    3. A hardcoded default.
    """
    load_env()
    assets_dir = get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)

    def _str_or_env(form_val: str | None, env_name: str, default: str) -> str:
        if form_val:
            return form_val
        return os.environ.get(env_name, default)

    def _bool_or_env(form_val: str | None, env_name: str, default: bool) -> bool:
        raw = form_val if form_val else os.environ.get(env_name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    auto_index = _bool_or_env(auto_index, "AUTO_INDEX", True)
    pdf_parser = _str_or_env(pdf_parser, "PDF_PARSER", "auto")
    enable_ocr = _bool_or_env(enable_ocr, "ENABLE_OCR", False)
    enable_vlm = _bool_or_env(enable_vlm, "ENABLE_VLM", False)
    image_provider = _str_or_env(image_provider, "IMAGE_PROVIDER", "lite")

    saved: list[str] = []
    rejected: list[dict[str, str]] = []
    for f in files:
        name = Path(f.filename or "").name
        if not name:
            rejected.append({"filename": "", "reason": "empty filename"})
            continue
        # Route by extension into the right subfolder.
        lower = name.lower()
        if lower.endswith(".pdf"):
            target_dir = assets_dir / "pdfs"
        elif lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            target_dir = assets_dir / "images"
        else:
            rejected.append({"filename": name, "reason": "unsupported extension"})
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / name
        # Avoid silently overwriting an existing file (e.g. a sample asset in
        # chapter11_assets). If the target exists, append a short hash to the
        # stem so the new copy lands next to it.
        if target_path.exists():
            import hashlib
            digest = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:6]
            target_path = target_dir / f"{target_path.stem}_{digest}{target_path.suffix}"
        # Stream to disk so large files don't buffer in RAM.
        with target_path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        # Path relative to assets_dir — matches Asset.relative_path so we can
        # build ephemeral Asset objects without consulting the manifest.
        saved.append(str(target_path.relative_to(assets_dir)))

    if not saved:
        raise HTTPException(status_code=400, detail={"rejected": rejected})

    options = ParseOptions(
        pdf_parser=pdf_parser,
        enable_ocr=enable_ocr,
        enable_vlm=enable_vlm,
        image_provider=image_provider,
        only_uploaded=True,
        uploaded_files=saved,
    )

    kind = "ingest" if auto_index else "parse"
    rec = _new_task(kind=kind, total=len(saved), uploaded=saved)
    target_fn = _run_ingest_task if auto_index else _run_parse_task
    threading.Thread(
        target=target_fn,
        args=(rec, options),
        name=f"mmrag-{kind}-{rec.task_id}",
        daemon=True,
    ).start()
    return {
        "task_id": rec.task_id,
        "kind": kind,
        "uploaded": saved,
        "options": {
            "auto_index": auto_index,
            "pdf_parser": pdf_parser,
            "enable_ocr": enable_ocr,
            "enable_vlm": enable_vlm,
            "image_provider": image_provider,
        },
        "rejected": rejected,
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, object]:
    with _TASKS_LOCK:
        rec = _TASKS.get(task_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
    payload = asdict(rec)
    payload["elapsed_sec"] = round(
        (rec.finished_at or time.time()) - rec.started_at, 1
    )
    if rec.total:
        payload["progress"] = round(rec.processed / rec.total, 3)
    else:
        payload["progress"] = None
    return payload


@app.get("/tasks")
def list_tasks() -> dict[str, object]:
    with _TASKS_LOCK:
        return {"tasks": [asdict(t) for t in _TASKS.values()]}


# ─── Static UI ───────────────────────────────────────────────────────────


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web UI not built; see mm_asset_rag/web/")
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ─── Server entrypoint ───────────────────────────────────────────────────


def run() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8011)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


# Suppress an unused-import warning when running as a module.
_ = (qdrant_image_to_image_search, qdrant_text_to_image_search)
