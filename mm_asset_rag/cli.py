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
from .search_service import SearchInputError, dispatch_search, get_search_service
from .service import ParseOptions, get_service
from .upload_pipeline import UserEdits, get_pipeline


def _collect_upload_files(inputs: list[Path]) -> list[tuple[str, Path]]:
    """Expand input directories while preserving their relative labels."""
    files: list[tuple[Path, Path]] = []
    for input_path in inputs:
        if input_path.is_dir():
            files.extend((input_path, path) for path in sorted(input_path.rglob("*")) if path.is_file())
        else:
            files.append((input_path.parent, input_path))
    return [(str(path.relative_to(root)), path) for root, path in files]


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
        if rec.status in ("done", "partial", "failed", "interrupted", "cancelled"):
            return
        time.sleep(poll_interval)


def command_parse(args: argparse.Namespace) -> None:
    """Sniff, parse and index files passed on the CLI.

    The CLI mirrors the web upload flow without the editable preview UI:
    it previews each file, accepts every supported preview as-is, then
    schedules parse + index through ``IngestService``.
    """
    input_paths = [Path(p).expanduser() for p in args.files]
    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing file(s): {', '.join(missing)}")

    pipeline = get_pipeline()
    if args.no_auto_meta:
        from .upload_pipeline import disable_auto_meta

        disable_auto_meta()
    upload_files = _collect_upload_files(input_paths)
    previews = pipeline.preview(upload_files)
    if not previews:
        raise SystemExit("no files to parse")
    cache_id = previews[0].cache_id
    if args.document_id and len(previews) != 1:
        raise SystemExit("--document-id requires exactly one input file")
    edits = [
        UserEdits(
            preview_id=p.preview_id,
            document_id=args.document_id,
            collection=args.collection,
            allowed_principals=list(args.principals),
            rejected=not p.is_supported,
        )
        for p in previews
    ]
    assets = pipeline.confirm(cache_id, edits)
    if not assets:
        raise SystemExit("no supported files to parse")

    options = ParseOptions(
        assets=assets,
        pdf_parser=args.pdf_parser,
        document_parser=args.document_parser,
        enable_ocr=args.ocr,
        enable_vlm=args.vlm,
        contextual=args.contextual,
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
    from .backends.qdrant.client import QdrantLockHeldError
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
    rows = [_serialize_hit(hit) for hit in hits]
    safe_print(json.dumps(rows, ensure_ascii=False, indent=2))


def _without_asset_id(value: object) -> object:
    if isinstance(value, dict):
        return {key: _without_asset_id(item) for key, item in value.items() if key != "asset_id"}
    if isinstance(value, list):
        return [_without_asset_id(item) for item in value]
    return value


def _serialize_hit(hit) -> dict[str, object]:
    """Serialize the public retrieval contract, never the internal DTO."""
    metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
    return {
        "document_id": metadata.get("document_id"),
        "version_id": metadata.get("version_id"),
        "chunk_id": metadata.get("chunk_id"),
        "title": hit.title,
        "source_type": hit.source_type,
        "source_path": hit.source_path,
        "evidence": hit.evidence,
        "score": hit.score,
        "routes": metadata.get("routes", [hit.route]),
        "page": metadata.get("page"),
        "parser": metadata.get("parser") or metadata.get("provider"),
        "images": _without_asset_id(hit.images or metadata.get("images") or []),
    }


def safe_print(text: str) -> None:
    print(
        text.encode("utf-8", errors="replace")
        .decode("utf-8")
        .encode("gbk", errors="replace")
        .decode("gbk")
    )


def command_search(args: argparse.Namespace) -> None:
    if args.mode == "image-to-image" and not args.image:
        raise RuntimeError("--image is required for image-to-image search")
    try:
        hits = dispatch_search(
            query=args.query,
            mode=args.mode,
            image_path=args.image or None,
            top_k=args.top_k,
            collection=args.collection,
            metadata_filter=args.metadata_filter,
            principal=args.principal,
        )
    except SearchInputError as exc:
        raise SystemExit(f"error: {exc}") from exc
    except RuntimeError as exc:
        # dispatch_search raises HTTPException for image-to-image without
        # image_path; surface a friendlier message for the CLI.
        raise RuntimeError(str(exc)) from exc
    print_hits(hits)


def _resolve_cli_cases_path(value: str | None) -> str | Path | None:
    """Resolve ``--cases`` via the shared resolver (``paths.resolve_cases_path``).

    The resolver validates the path (relative, ``.json``, no traversal),
    resolves it under ``eval_cases/`` then ``examples/``, and checks the
    file exists. A bad path raises ``ValueError`` / ``FileNotFoundError``;
    we translate both to a CLI-friendly ``SystemExit``.
    """
    from .paths import resolve_cases_path

    try:
        resolved = resolve_cases_path(value)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    return None if resolved is None else str(resolved)


def command_eval(args: argparse.Namespace) -> None:
    cases_path = _resolve_cli_cases_path(args.cases)
    if args.answer_quality:
        from .answer_evaluation import run_answer_eval, write_answer_eval_report

        results = run_answer_eval(
            top_k=args.top_k,
            cases_path=cases_path,
            collection=args.collection,
            metadata_filter=args.metadata_filter,
            principal=args.principal,
        )
        write_answer_eval_report(results)
        safe_print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
        # Surface the dominant "all fallback" / "all faithfulness skipped"
        # outcomes so a user on a machine without an LLM doesn't think the
        # eval silently broke — the JSON above will read near-zero.
        from .answer_evaluation import _aggregate_answer_metrics

        payload = _aggregate_answer_metrics(results)
        breakdown = payload.get("summary", {}).get("answer_sources") or {}
        skipped = payload.get("metrics", {}).get("all", {}).get("faithfulness_skipped", 0)
        if breakdown.get("fallback") and not breakdown.get("llm"):
            safe_print(
                "\n[answer-quality] no LLM creds configured - every case ran "
                "via fallback_answer (coverage / citation = 0). Set "
                "OPENAI_* / VLM_* to score /answer."
            )
        elif skipped == len(results) and results:
            safe_print(
                "\n[answer-quality] every case skipped faithfulness "
                "(check OPENAI_* / VLM_* creds or raise EVAL_JUDGE_MAX_CASES)."
            )
        return
    if args.v2:
        from .evaluation_v2 import run_eval_v2, write_eval_report_v2

        results = run_eval_v2(
            top_k=args.top_k,
            cases_path=cases_path,
            collection=args.collection,
            metadata_filter=args.metadata_filter,
            principal=args.principal,
        )
        # Only the text→text group runs here; the text→image / image→image
        # groups live in their own one-shots under ``evaluation_v2.__main__``.
        # ``write_eval_report_v2`` expects a ``{group_name: [V2Result, ...]}``
        # mapping, so we wrap the text→text results. The default ``mmrag eval``
        # output compares against v1's ``run_eval``, which is also text→text.
        write_eval_report_v2({"text_to_text": results})
        safe_print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
        return
    results = run_eval(
        top_k=args.top_k,
        cases_path=cases_path,
        collection=args.collection,
        metadata_filter=args.metadata_filter,
        principal=args.principal,
    )
    write_eval_report(results)
    safe_print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


def command_answer(args: argparse.Namespace) -> None:
    safe_print(
        answer_json(
            args.question,
            top_k=args.top_k,
            search_service=get_search_service(),
            collection=args.collection,
            metadata_filter=args.metadata_filter,
            principal=args.principal,
            min_confidence=args.min_confidence,
        )
    )


def command_retry(args: argparse.Namespace) -> None:
    """Re-run a previously failed / partial / interrupted task."""
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


def command_documents(args: argparse.Namespace) -> None:
    """List logical documents and their latest immutable version."""
    from . import asset_index

    latest = {}
    for record in asset_index.load_records():
        policy = record.access_policy
        if policy.collection != args.collection or not policy.allows(args.principal):
            continue
        if any(
            policy.metadata.get(key) != value for key, value in (args.metadata_filter or {}).items()
        ):
            continue
        current = latest.get(record.document.document_id)
        if current is None or record.version.version_number > current.version.version_number:
            latest[record.document.document_id] = record
    safe_print(
        json.dumps(
            [
                {
                    "document_id": record.document.document_id,
                    "title": record.document.title,
                    "source": record.document.source.to_record(),
                    "latest_version": record.version.to_record(),
                }
                for _document_id, record in sorted(latest.items())
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrag",
        description="mm-asset-rag: multimodal asset RAG (Qdrant backend)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse and index PDF/image files")
    parse_cmd.add_argument("files", nargs="+", help="PDF/image files to ingest")
    parse_cmd.add_argument(
        "--document-id",
        help="Stable logical document identity (allowed only for one input file)",
    )
    parse_cmd.add_argument("--collection", required=True, help="Access-policy collection")
    parse_cmd.add_argument(
        "--principal",
        dest="principals",
        action="append",
        required=True,
        help="Allowed principal; repeat to grant more than one",
    )
    parse_cmd.add_argument(
        "--pdf-parser",
        choices=["auto", "pymupdf", "paddleocr_vl", "docling", "ppocr"],
        default="auto",
    )
    parse_cmd.add_argument(
        "--document-parser",
        choices=["markitdown", "docling"],
        default="markitdown",
        help="Backend for office/text documents (docx/pptx/xlsx/html/md): "
        "markitdown (default, core dep) or docling (needs [docling] extra).",
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
    parse_cmd.add_argument(
        "--contextual",
        action="store_true",
        help=(
            "Generate an LLM context preamble per chunk (Contextual Retrieval) "
            "before indexing. Improves precision when queries use generic terms "
            "(e.g. 'diffusion' matching DDPM vs Stable Diffusion). opt-in: costs "
            "~1 LLM call per chunk; cached under parsed/<id>/context.jsonl so "
            "reindex reuses it. Requires OPENAI_* LLM credentials."
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

    search_cmd = subparsers.add_parser("search", help="Search authorized documents")
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
    search_cmd.add_argument("--collection", required=True, help="Access-policy collection")
    search_cmd.add_argument("--principal", required=True, help="Requesting principal")
    search_cmd.add_argument(
        "--metadata-filter",
        type=json.loads,
        default=None,
        help="JSON object of required policy metadata",
    )
    search_cmd.set_defaults(func=command_search)

    eval_cmd = subparsers.add_parser("eval", help="Run the small retrieval regression set")
    eval_cmd.add_argument("--top-k", type=int, default=5)
    eval_cmd.add_argument("--collection", required=True, help="Access-policy collection")
    eval_cmd.add_argument("--principal", required=True, help="Requesting principal")
    eval_cmd.add_argument(
        "--metadata-filter",
        type=json.loads,
        default=None,
        help="JSON object of required policy metadata",
    )
    eval_cmd.add_argument(
        "--cases",
        default=None,
        help=(
            "Path to a case JSON overriding the default "
            "(Settings.EVAL_CASES_PATH → the bundled sample). Schema: "
            '{"version","groups":{group:[{query_id,query}]},'
            '"qrels":{query_id:{document_id:relevance}}}. '
            "Document IDs are matched exactly."
        ),
    )
    # --v2 and --answer-quality are mutually exclusive: they share top_k /
    # cases_path inputs but write different reports (eval_report_v2.json vs
    # eval_report_answer.json) and score different things (retrieval vs
    # generation). argparse requires the mutex group to be created before
    # any of its members, so both flags live here rather than one per branch.
    eval_mode = eval_cmd.add_mutually_exclusive_group()
    eval_mode.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Run the v2 regression set (multi-dimensional, Chinese-primary: "
            "cross-language / multi-relevant / negative) instead of the v1 "
            "set. Writes eval_report_v2.json. Default is v1 so existing "
            "scripts / dashboards keep their numbers."
        ),
    )
    eval_mode.add_argument(
        "--answer-quality",
        action="store_true",
        help=(
            "Run the answer-quality eval (coverage + citation + LLM-judge "
            "faithfulness). Writes eval_report_answer.json. Text→text cases "
            "only in v0; image-route cases raise ValueError. Coverage / "
            "citation always run; faithfulness is skipped when no LLM creds "
            "are configured (set OPENAI_* or VLM_*)."
        ),
    )
    eval_cmd.set_defaults(func=command_eval)

    answer_cmd = subparsers.add_parser("answer", help="Answer with retrieved multimodal evidence")
    answer_cmd.add_argument("question")
    answer_cmd.add_argument("--top-k", type=int, default=5)
    answer_cmd.add_argument("--collection", required=True, help="Access-policy collection")
    answer_cmd.add_argument("--principal", required=True, help="Requesting principal")
    answer_cmd.add_argument(
        "--metadata-filter",
        type=json.loads,
        default=None,
        help="JSON object of required policy metadata",
    )
    answer_cmd.add_argument(
        "--min-confidence",
        type=float,
        required=True,
        help="Mandatory refusal threshold; the LLM is not called below it",
    )
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

    documents_cmd = subparsers.add_parser(
        "documents", help="List logical documents and their latest versions"
    )
    documents_cmd.add_argument("--collection", required=True, help="Access-policy collection")
    documents_cmd.add_argument("--principal", required=True, help="Requesting principal")
    documents_cmd.add_argument(
        "--metadata-filter",
        type=json.loads,
        default=None,
        help="JSON object of required policy metadata",
    )
    documents_cmd.set_defaults(func=command_documents)

    return parser


def main() -> None:
    # Load .env once at the CLI entry point. ``load_dotenv()`` walks up
    # from the cwd to find a ``.env`` file, so this preserves the
    # "run ``mmrag`` from any subdirectory" behaviour. Individual
    # subcommands no longer call ``load_env()`` themselves — pydantic
    # ``Settings`` reads ``.env`` from the cwd automatically, and the
    # single call here populates ``os.environ`` for the residual
    # ``os.environ.get(...)`` sites that still need it.
    load_env()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
