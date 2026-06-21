"""Command-line interface for mm-asset-rag."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .answer import answer_json
from .assets import load_assets
from .config import load_env
from .document_store import write_documents
from .evaluation import run_eval, write_eval_report
from .image_parser import parse_image
from .paths import get_documents_jsonl, get_text_index_dir
from .pdf_parser import parse_pdf
from .qdrant_store import (
    build_qdrant_image_index,
    build_qdrant_text_index,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .retrieval import hybrid_search


def command_parse(args: argparse.Namespace) -> None:
    load_env()
    assets = load_assets(limit=args.limit)
    documents = []
    failures: list[tuple[str, str]] = []  # (asset_id, error message)
    for asset in assets:
        try:
            if asset.source_type == "pdf":
                parsed = parse_pdf(asset, parser=args.pdf_parser)
            elif asset.source_type == "image":
                parsed = parse_image(asset, enable_ocr=args.ocr, enable_vlm=args.vlm)
            else:
                parsed = []
        except Exception as exc:
            print(
                f"FAILED asset={asset.asset_id} type={asset.source_type} "
                f"error={type(exc).__name__}: {exc}"
            )
            failures.append((asset.asset_id, str(exc)))
            continue
        documents.extend(parsed)
        print(f"parsed asset={asset.asset_id} type={asset.source_type} documents={len(parsed)}")
    write_documents(documents)
    if failures:
        print(f"WARNING: {len(failures)} asset(s) failed to parse:")
        for asset_id, msg in failures:
            print(f"  - {asset_id}: {msg}")
    print(f"documents={len(documents)}")
    print(f"documents_jsonl={get_documents_jsonl()}")


def command_index(args: argparse.Namespace) -> None:
    load_env()
    text_count, embedding_name = build_qdrant_text_index()
    image_count, image_provider = build_qdrant_image_index()
    print(f"text_documents={text_count}")
    print(f"embedding={embedding_name}")
    print(f"text_index={get_text_index_dir()}")
    print(f"image_records={image_count}")
    print(f"image_provider={image_provider}")


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

    index_cmd = subparsers.add_parser("index", help="Build text and image indexes in Qdrant")
    index_cmd.set_defaults(func=command_index)

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
