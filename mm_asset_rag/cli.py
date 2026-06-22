"""Command-line interface for mm-asset-rag.

The CLI is intentionally a thin wrapper around the same
:class:`~mm_asset_rag.service.IngestService` the FastAPI app uses, so the
parse / index pipeline only lives in one place.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .answer import answer_json
from .config import load_env
from .evaluation import run_eval, write_eval_report
from .paths import get_data_dir, get_documents_jsonl, get_text_index_dir
from .backends.qdrant_backend import (
    build_qdrant_image_index,
    build_qdrant_text_index,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .retrieval import hybrid_search
from .service import ParseOptions, get_service


def _wait_for_task(task_id: str, poll_interval: float = 1.0) -> None:
    """Block until ``task_id`` finishes; print progress to stdout."""
    service = get_service()
    last_line = ""
    while True:
        rec = service.get_task(task_id)
        if rec is None:
            print(f"task {task_id} not found")
            return
        cur = rec.current or "(starting)"
        if cur != last_line:
            print(f"[task {rec.task_id}] {rec.status} · {cur}", flush=True)
            last_line = cur
        if rec.status in ("done", "partial", "failed", "interrupted"):
            return
        time.sleep(poll_interval)


def command_parse(args: argparse.Namespace) -> None:
    """Parse every asset in the manifest, synchronously.

    Delegates to :class:`IngestService.parse_manifest` (which uses the same
    worker-thread scaffold the API uses) and waits for the background
    thread to finish so the CLI exit code reflects parse failures.
    """
    load_env()
    options = ParseOptions(
        pdf_parser=args.pdf_parser,
        enable_ocr=args.ocr,
        enable_vlm=args.vlm,
    )
    rec = get_service().parse_manifest(limit=args.limit, options=options)
    print(f"started task {rec.task_id} (parse only)")
    _wait_for_task(rec.task_id)
    print(f"documents_jsonl={get_documents_jsonl()}")


def command_index(args: argparse.Namespace) -> None:
    """Incrementally upsert the Qdrant text + image indexes."""
    load_env()
    service = get_service()
    # Trigger an empty-manifest ingest so the service does the index work
    # through its normal pipeline (consistency with /upload).
    rec = service.parse_manifest(limit=0, options=ParseOptions())
    _wait_for_task(rec.task_id)
    print(f"text_index={get_text_index_dir()}")


def command_reindex(args: argparse.Namespace) -> None:
    """Drop and rebuild the qdrant collections from documents.jsonl.

    The default ``index`` command is incremental (skips already-indexed docs);
    use ``reindex`` when you want a clean slate — e.g. after changing the
    embedding model or fixing a corrupted collection.
    """
    load_env()
    only = "text" if args.text_only else "image" if args.image_only else "both"
    if only in ("text", "both"):
        n, name = build_qdrant_text_index(force_recreate=True)
        print(f"[reindex] text: {name}")
    if only in ("image", "both"):
        ni, ni_name = build_qdrant_image_index(force_recreate=True)
        print(f"[reindex] image: {ni_name}")


def print_hits(hits) -> None:
    rows = [asdict(hit) for hit in hits]
    safe_print(json.dumps(rows, ensure_ascii=False, indent=2))


def safe_print(text: str) -> None:
    print(
        text.encode("utf-8", errors="replace")
        .decode("utf-8")
        .encode("gbk", errors="replace")
        .decode("gbk")
    )


def command_search(args: argparse.Namespace) -> None:
    load_env()
    if args.mode == "text":
        hits = qdrant_text_search(args.query, top_k=args.top_k)
    elif args.mode == "text-to-image":
        hits = qdrant_text_to_image_search(args.query, top_k=args.top_k)
    elif args.mode == "image-to-image":
        if not args.image:
            raise RuntimeError("--image is required for image-to-image search")
        hits = qdrant_image_to_image_search(Path(args.image), top_k=args.top_k)
    elif args.mode == "hybrid":
        hits = hybrid_search(
            args.query,
            image_path=Path(args.image) if args.image else None,
            top_k=args.top_k,
        )
    else:
        raise RuntimeError(f"Unsupported search mode: {args.mode}")
    print_hits(hits)


def command_eval(args: argparse.Namespace) -> None:
    load_env()
    results = run_eval(top_k=args.top_k)
    write_eval_report(results)
    safe_print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


def command_answer(args: argparse.Namespace) -> None:
    load_env()
    safe_print(answer_json(args.question, top_k=args.top_k))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrag",
        description="mm-asset-rag: multimodal asset RAG (Qdrant backend)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse PDF and image assets")
    parse_cmd.add_argument("--limit", type=int, default=0)
    parse_cmd.add_argument(
        "--pdf-parser", choices=["auto", "pymupdf", "paddleocr_vl"], default="auto"
    )
    parse_cmd.add_argument("--ocr", action="store_true", help="Run local OCR HTTP for images")
    parse_cmd.add_argument(
        "--vlm", action="store_true", help="Run OpenAI-compatible VLM captions for images"
    )
    parse_cmd.set_defaults(func=command_parse)

    index_cmd = subparsers.add_parser(
        "index",
        help="Incrementally upsert text and image indexes in Qdrant (skips already-indexed docs)",
    )
    index_cmd.set_defaults(func=command_index)

    reindex_cmd = subparsers.add_parser(
        "reindex",
        help="Drop and rebuild qdrant collections from documents.jsonl (use after changing models)",
    )
    reindex_cmd.add_argument("--text-only", action="store_true")
    reindex_cmd.add_argument("--image-only", action="store_true")
    reindex_cmd.set_defaults(func=command_reindex)

    search_cmd = subparsers.add_parser("search", help="Search indexed assets")
    search_cmd.add_argument("query")
    search_cmd.add_argument(
        "--mode",
        choices=["text", "text-to-image", "image-to-image", "hybrid"],
        default="hybrid",
    )
    search_cmd.add_argument(
        "--image", default="", help="Path to query image (for image-to-image / hybrid)"
    )
    search_cmd.add_argument("--top-k", type=int, default=5)
    search_cmd.set_defaults(func=command_search)

    eval_cmd = subparsers.add_parser("eval", help="Run the small retrieval regression set")
    eval_cmd.add_argument("--top-k", type=int, default=5)
    eval_cmd.set_defaults(func=command_eval)

    answer_cmd = subparsers.add_parser("answer", help="Answer with retrieved multimodal evidence")
    answer_cmd.add_argument("question")
    answer_cmd.add_argument("--top-k", type=int, default=5)
    answer_cmd.set_defaults(func=command_answer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
