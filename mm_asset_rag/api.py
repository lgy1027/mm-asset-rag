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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import answer_question, stream_answer_chunks
from .assets import load_assets
from .cli import command_index, command_parse
from .config import load_env
from .evaluation import run_eval
from .paths import get_assets_dir, get_documents_jsonl, get_indexes_dir, get_text_index_dir
from .qdrant_store import qdrant_image_to_image_search, qdrant_text_search, qdrant_text_to_image_search
from .retrieval import hybrid_search


app = FastAPI(
    title="mm-asset-rag",
    version="0.1.0",
    description="Multimodal asset RAG: PDF + image parsing, hybrid retrieval, grounded answers.",
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


def _run_parse_task(rec: TaskRecord, limit: int) -> None:
    """Run mmrag parse in a worker thread, updating rec as it progresses."""
    from .config import load_env as _load_env
    from .document_store import read_documents as _read

    _load_env()
    from .paths import get_parsed_dir as _parsed

    _patch(rec, status="running", current="loading assets")
    assets = load_assets(limit=limit)
    if not assets:
        _patch(rec, status="done", current="no assets to parse", finished_at=time.time())
        return

    _patch(rec, total=len(assets), current=f"parsing {len(assets)} asset(s)")

    failed = 0
    skipped = 0
    for i, asset in enumerate(assets, start=1):
        try:
            # If raw.jsonl already exists on disk, skip the parse.
            raw_path = _parsed() / asset.asset_id / "raw.jsonl"
            if raw_path.exists() and raw_path.stat().st_size > 0:
                skipped += 1
                _patch(rec, processed=i, current=f"skip cached: {asset.asset_id}")
                continue
            # Mimic the per-asset parse from cli.command_parse
            from .pdf_parser import parse_pdf as _pp
            from .image_parser import parse_image as _pi

            try:
                if asset.source_type == "pdf":
                    docs = _pp(asset, parser=os.environ.get("PDF_PARSER", "pymupdf"))
                elif asset.source_type == "image":
                    docs = _pi(asset, enable_ocr=False, enable_vlm=False)
                else:
                    docs = []
            except Exception as exc:
                failed += 1
                print(f"parse task failed for {asset.asset_id}: {exc}")
                continue
            # Append to documents.jsonl incrementally so progress is visible
            from .document_store import write_documents as _wd
            # _wd overwrites; we append instead
            target = get_documents_jsonl()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
            _patch(rec, processed=i, current=f"parsed {asset.asset_id} ({len(docs)} doc)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            _patch(rec, processed=i, current=f"error {asset.asset_id}: {exc}")

    _patch(
        rec,
        status="done" if failed == 0 else "partial",
        finished_at=time.time(),
        current=f"done: processed={rec.processed} skipped={skipped} failed={failed}",
    )


def _run_ingest_task(rec: TaskRecord, limit: int) -> None:
    """Parse + index in sequence on a background thread."""
    _run_parse_task(rec, limit=limit)
    if rec.status == "failed":
        return
    _patch(rec, current="building vector index")
    try:
        command_index(argparse.Namespace(backend="qdrant"))
        _patch(rec, current="index built", status="done", finished_at=time.time())
    except SystemExit:
        _patch(rec, current="index built", status="done", finished_at=time.time())
    except Exception as exc:  # noqa: BLE001
        _patch(rec, status="failed", error=str(exc), finished_at=time.time())


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
async def upload(files: list[UploadFile] = File(...), auto_index: bool = True) -> dict[str, object]:
    """Accept a batch of files, copy them into ``$MM_ASSET_RAG_HOME/assets/``,
    then run a background parse (+ index if ``auto_index``).

    Returns immediately with a ``task_id``; poll ``/tasks/{id}`` for progress.
    """
    load_env()
    assets_dir = get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)

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
        # Stream to disk so large files don't buffer in RAM.
        with target_path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(target_path.relative_to(assets_dir.parent)))

    if not saved:
        raise HTTPException(status_code=400, detail={"rejected": rejected})

    rec = _new_task(kind="ingest" if auto_index else "parse", total=len(saved), uploaded=saved)
    kind = rec.kind
    target = _run_ingest_task if kind == "ingest" else _run_parse_task
    threading.Thread(
        target=target, args=(rec, 0), name=f"mmrag-{kind}-{rec.task_id}", daemon=True
    ).start()
    return {"task_id": rec.task_id, "kind": kind, "uploaded": saved, "rejected": rejected}


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
