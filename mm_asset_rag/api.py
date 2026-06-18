"""HTTP API for mm-asset-rag (FastAPI)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import answer_question
from .assets import load_assets
from .cli import command_index, command_parse
from .config import load_env
from .evaluation import run_eval
from .paths import get_documents_jsonl, get_text_index_dir
from .qdrant_store import (
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .retrieval import hybrid_search

app = FastAPI(
    title="mm-asset-rag",
    version="0.1.0",
    description="Multimodal asset RAG: PDF + image parsing, Qdrant indexing, hybrid retrieval, LLM answering.",
)


class IngestRequest(BaseModel):
    limit: int = 0
    pdf_parser: str = Field(default="auto", pattern="^(auto|pymupdf|paddleocr_vl)$")
    ocr: bool = False
    vlm: bool = False


class SearchRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(text|text-to-image|image-to-image|hybrid)$")
    image_path: str | None = None
    top_k: int = 5


class AnswerRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health() -> dict[str, object]:
    load_env()
    return {
        "status": "ok",
        "assets": len(load_assets()),
        "documents_jsonl_exists": get_documents_jsonl().exists(),
        "text_index_exists": get_text_index_dir().exists(),
        "vector_backend": "qdrant",
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    import argparse

    command_parse(
        argparse.Namespace(
            limit=request.limit,
            pdf_parser=request.pdf_parser,
            ocr=request.ocr,
            vlm=request.vlm,
        )
    )
    command_index(argparse.Namespace())
    return {
        "status": "ok",
        "documents_jsonl": str(get_documents_jsonl()),
        "backend": "qdrant",
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    if request.mode == "text":
        hits = qdrant_text_search(request.query, top_k=request.top_k)
    elif request.mode == "text-to-image":
        hits = qdrant_text_to_image_search(request.query, top_k=request.top_k)
    elif request.mode == "image-to-image":
        if not request.image_path:
            raise HTTPException(
                status_code=400,
                detail="image_path is required for image-to-image search",
            )
        hits = qdrant_image_to_image_search(Path(request.image_path), top_k=request.top_k)
    else:
        hits = hybrid_search(
            request.query,
            image_path=Path(request.image_path) if request.image_path else None,
            top_k=request.top_k,
        )
    return {"query": request.query, "hits": [hit.__dict__ for hit in hits]}


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, object]:
    return answer_question(request.question, top_k=request.top_k)


@app.post("/eval")
def eval_endpoint() -> dict[str, object]:
    results = run_eval()
    return {"results": [result.__dict__ for result in results]}


def run() -> None:
    """Entry point for ``mmrag-api`` console script."""
    import uvicorn

    host = "127.0.0.1"
    port = 8011
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
