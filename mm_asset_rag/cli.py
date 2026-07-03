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
from .paths import get_documents_jsonl
from .service import ParseOptions, dispatch_search, get_service
from .upload_pipeline import UserEdits, get_pipeline


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
    """Sniff, parse and index files passed on the CLI.

    The CLI mirrors the web upload flow without the editable preview UI:
    it previews each file, accepts every supported preview as-is, then
    schedules parse + index through ``IngestService``.
    """
    load_env()
    file_paths = [Path(p).expanduser() for p in args.files]
    missing = [str(p) for p in file_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing file(s): {', '.join(missing)}")

    pipeline = get_pipeline()
    if args.no_auto_meta:
        from .upload_pipeline import disable_auto_meta

        disable_auto_meta()
    previews = pipeline.preview([(p.name, p) for p in file_paths])
    if not previews:
        raise SystemExit("no files to parse")
    cache_id = previews[0].cache_id
    edits = [UserEdits(preview_id=p.preview_id, rejected=not p.is_supported) for p in previews]
    assets = pipeline.confirm(cache_id, edits)
    if not assets:
        raise SystemExit("no supported files to parse")

    options = ParseOptions(
        assets=assets, pdf_parser=args.pdf_parser, enable_ocr=args.ocr, enable_vlm=args.vlm
    )
    rec = get_service().ingest_assets(assets, options)
    print(f"started task {rec.task_id} (parse + index)")
    _wait_for_task(rec.task_id)
    print(f"documents_jsonl={get_documents_jsonl()}")


def command_reindex(args: argparse.Namespace) -> None:
    """Drop and rebuild the qdrant collections from documents.jsonl.

    Routes through :meth:`IngestService.reindex` so the CLI and any
    other caller share one implementation — and one lock-detection
    path. The default ``index`` command is incremental (skips
    already-indexed docs); use ``reindex`` when you want a clean slate
    — e.g. after changing the embedding model or fixing a corrupted
    collection.

    qdrant local mode is single-process: stop the API server (or any other
    mm-asset-rag process) before running this command, otherwise the local
    storage lock will block. Use ``QDRANT_URL`` (server mode) if you need
    concurrent access.

    ``--yes`` skips the interactive confirmation — useful for CI / scripts
    and for the "switch CLIP model" recipe in ``docs/eval-report-v3.md``.
    """
    load_env()
    from .backends.qdrant_backend import QdrantLockHeldError
    from .service import get_service

    if not args.yes:
        targets = []
        if not args.image_only:
            targets.append("text")
        if not args.text_only:
            targets.append("image")
        msg = f"rebuild {', '.join(targets)} collection(s)? [y/N] "
        try:
            ans = input(msg)
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted")

    try:
        names = get_service().reindex(
            text_only=args.text_only,
            image_only=args.image_only,
        )
    except QdrantLockHeldError as exc:
        raise SystemExit(f"error: {exc}") from exc
    for name in names:
        print(f"[reindex] {name}")


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
    if args.mode == "image-to-image" and not args.image:
        raise RuntimeError("--image is required for image-to-image search")
    try:
        hits = dispatch_search(
            query=args.query,
            mode=args.mode,
            image_path=args.image or None,
            top_k=args.top_k,
        )
    except RuntimeError as exc:
        # dispatch_search raises HTTPException for image-to-image without
        # image_path; surface a friendlier message for the CLI.
        raise RuntimeError(str(exc)) from exc
    print_hits(hits)


def command_eval(args: argparse.Namespace) -> None:
    load_env()
    results = run_eval(top_k=args.top_k)
    write_eval_report(results)
    safe_print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


def command_answer(args: argparse.Namespace) -> None:
    load_env()
    safe_print(answer_json(args.question, top_k=args.top_k))


def command_retry(args: argparse.Namespace) -> None:
    """Re-run a previously failed / partial / interrupted task."""
    load_env()
    service = get_service()
    service.load_history()
    try:
        rec = service.retry_task(args.task_id, force=args.force, failed_only=args.failed_only)
    except KeyError as exc:
        raise SystemExit(f"unknown task: {exc}") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"cannot retry task: {exc}") from exc
    flags = []
    if rec.force:
        flags.append("force")
    if rec.failed_only:
        flags.append("failed-only")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    print(
        f"started retry task {rec.task_id} (origin {rec.origin_task_id}, kind={rec.kind}){flag_str}"
    )
    _wait_for_task(rec.task_id)


def command_delete(args: argparse.Namespace) -> None:
    """Delete an asset (best-effort across disk, parsed, captions, Qdrant, index)."""
    load_env()
    if not args.dry_run and not args.yes:
        confirm = input(
            f"Delete asset {args.asset_id}? This will remove its file, parsed/, "
            "captions/, Qdrant points and asset index entry. [y/N] "
        )
        if confirm.strip().lower() not in {"y", "yes"}:
            print("aborted")
            return
    report = get_service().delete_asset(args.asset_id, dry_run=args.dry_run)
    if not report.was_known:
        raise SystemExit(f"unknown asset: {args.asset_id}")
    safe_print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrag",
        description="mm-asset-rag: multimodal asset RAG (Qdrant backend)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse and index PDF/image files")
    parse_cmd.add_argument("files", nargs="+", help="PDF/image files to ingest")
    parse_cmd.add_argument(
        "--pdf-parser", choices=["auto", "pymupdf", "paddleocr_vl"], default="auto"
    )
    parse_cmd.add_argument("--ocr", action="store_true", help="Run local OCR HTTP for images")
    parse_cmd.add_argument(
        "--vlm", action="store_true", help="Run OpenAI-compatible VLM captions for images"
    )
    parse_cmd.add_argument(
        "--no-auto-meta",
        action="store_true",
        help=(
            "Skip the VLM-based title / tags / description extraction in the "
            "preview phase. Useful on slow VLM endpoints or when ingesting a "
            "large batch where the per-file round-trip becomes the bottleneck."
        ),
    )
    parse_cmd.set_defaults(func=command_parse)

    reindex_cmd = subparsers.add_parser(
        "reindex",
        help="Drop and rebuild qdrant collections from documents.jsonl (use after changing models)",
    )
    reindex_cmd.add_argument("--text-only", action="store_true")
    reindex_cmd.add_argument("--image-only", action="store_true")
    reindex_cmd.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation (CI / scripts).",
    )
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

    retry_cmd = subparsers.add_parser(
        "retry",
        help="Re-run a previously failed / partial / interrupted task",
    )
    retry_cmd.add_argument("task_id", help="Task id returned by /upload/confirm or mmrag parse")
    retry_cmd.add_argument(
        "--force",
        action="store_true",
        help="Clear parsed/<id>/ cache before re-running",
    )
    retry_cmd.add_argument(
        "--failed-only",
        action="store_true",
        help=(
            "Only re-run assets that previously failed or were skipped. "
            "Composable with --force: only the failed assets' cache is cleared."
        ),
    )
    retry_cmd.set_defaults(func=command_retry)

    delete_cmd = subparsers.add_parser(
        "delete",
        help="Delete an asset (file, parsed/, captions/, Qdrant points, index entry)",
    )
    delete_cmd.add_argument("asset_id", help="Asset id to delete")
    delete_cmd.add_argument("--yes", action="store_true", help="Skip the interactive confirmation")
    delete_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without touching the disk or Qdrant",
    )
    delete_cmd.set_defaults(func=command_delete)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
