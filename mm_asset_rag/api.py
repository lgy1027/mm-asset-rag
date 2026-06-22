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

import json
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import answer_question, stream_answer_chunks
from .assets import load_assets
from .config import load_env
from .evaluation import run_eval
from .paths import get_assets_dir, get_documents_jsonl, get_indexes_dir, get_text_index_dir
from .backends.qdrant_backend import (
    get_qdrant_client,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .retrieval import hybrid_search
from .service import IngestService, ParseOptions, get_service
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: restore task history from disk. Tasks still marked 'running'
    # when the previous process exited get reclassified as 'interrupted'.
    get_service().load_history()
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


# ─── Service layer (task scheduling + history) ─────────────────────────
#
# All background work, persistence, and task queries live in
# ``mm_asset_rag.service``. The FastAPI app stays a thin route layer.

from .service import IngestService, ParseOptions, TaskRecord, get_service  # noqa: E402, F401


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


def _coerce_bool(form_val: str | bool | None, default: bool) -> bool:
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

    settings = get_settings()
    pdf_parser = pdf_parser or settings.pdf_parser
    image_provider = image_provider or settings.image_provider
    # The form value for booleans is a string ("true"/"false"); coerce via
    # ``_coerce_bool`` so the precedence form > env > default applies uniformly.
    auto_index = _coerce_bool(auto_index, settings.auto_index)
    enable_ocr = _coerce_bool(enable_ocr, settings.enable_ocr)
    enable_vlm = _coerce_bool(enable_vlm, settings.enable_vlm)

    options = ParseOptions(
        pdf_parser=pdf_parser,
        enable_ocr=enable_ocr,
        enable_vlm=enable_vlm,
        image_provider=image_provider,
    )

    service = get_service()
    if auto_index:
        rec = service.ingest_uploaded(saved, options)
    else:
        rec = service.parse_uploaded(saved, options)
    kind = rec.kind
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
    from dataclasses import asdict

    rec = get_service().get_task(task_id)
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
    from dataclasses import asdict

    return {"tasks": [asdict(t) for t in get_service().list_tasks()]}


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
