"""Parse, enrich, document-persist, and index orchestration."""

from __future__ import annotations

import json
import shutil
import time
from typing import TYPE_CHECKING

from .document_store import documents_jsonl_lock
from .paths import get_documents_jsonl, get_parsed_dir
from .task_store import TaskRecord

if TYPE_CHECKING:
    from .service import IngestService, ParseOptions


class _TaskCancelled(Exception):
    """Internal signal used to unwind cooperative index cancellation."""


class IngestWorkflow:
    """Own the ingest worker's parse through index sequence."""

    def run(
        self,
        service: IngestService,
        record: TaskRecord,
        options: ParseOptions,
    ) -> None:
        """Parse and index in sequence, preserving cooperative cancellation."""
        # Resolve through the compatibility facade so callers that historically
        # patched ``service._run_parse_task`` retain the same test seam.
        from . import service as service_module

        service_module._run_parse_task(service, record, options)
        if record.status == service_module.TaskStatus.FAILED:
            service._patch(record, finished_at=time.time())
            return
        if record.status == service_module.TaskStatus.CANCELLED:
            return
        parse_status = record.status

        if (
            service._is_cancelled(record.task_id)
            or record.status == service_module.TaskStatus.CANCELLED
        ):
            service._patch(
                record,
                status=service_module.TaskStatus.CANCELLED,
                finished_at=time.time(),
                current="cancelled before index",
            )
            return
        service._patch(
            record,
            status="running",
            finished_at=None,
            current="parse done, building index",
        )

        indexed_targets = {
            asset_id: status
            for asset_id, status in record.asset_statuses.items()
            if status in {service_module.AssetStatus.OK, service_module.AssetStatus.SKIPPED}
        }

        def _progress(done: int, total: int, phase: str) -> None:
            if service._is_cancelled(record.task_id):
                raise _TaskCancelled
            service._patch(
                record,
                processed=done,
                total=total,
                current=f"indexing: {phase}",
            )

        try:
            backend = service_module.get_backend("qdrant")
            text_n, text_name = backend.upsert_text(progress_cb=_progress)
            service._patch(record, current=f"text indexed · {text_name}")
            image_n, _image_name = backend.upsert_image(progress_cb=_progress)
            service._invalidate_search_caches()
            new_statuses = dict(record.asset_statuses)
            for asset_id in indexed_targets:
                new_statuses[asset_id] = "indexed"
            if (
                service._is_cancelled(record.task_id)
                or record.status == service_module.TaskStatus.CANCELLED
            ):
                service._patch(
                    record,
                    status=service_module.TaskStatus.CANCELLED,
                    finished_at=time.time(),
                    current=f"cancelled after index · text={text_n} image={image_n}",
                    asset_statuses=new_statuses,
                )
                return
            service._patch(
                record,
                current=f"index built · text={text_n} image={image_n}",
                status=parse_status,
                finished_at=time.time(),
                asset_statuses=new_statuses,
            )
        except _TaskCancelled:
            service._patch(
                record,
                status=service_module.TaskStatus.CANCELLED,
                finished_at=time.time(),
                current="cancelled during index",
            )
            print(f"[task {record.task_id}] index cancelled by request")
        except BaseException as exc:
            new_statuses = dict(record.asset_statuses)
            for asset_id in indexed_targets:
                new_statuses[asset_id] = "failed_index"
            service._patch(
                record,
                current=f"index crashed: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
                status="failed",
                finished_at=time.time(),
                asset_statuses=new_statuses,
            )
            print(f"[task {record.task_id}] index crashed: {exc!r}")

    def run_parse(
        self,
        service: IngestService,
        record: TaskRecord,
        options: ParseOptions,
    ) -> None:
        """Run the parse stage and surface worker crashes on the task."""
        service._patch(record, status="running", current="starting")
        try:
            self.parse(service, record, options)
        except BaseException as exc:
            service._patch(
                record,
                status="failed",
                current=f"parse crashed: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=time.time(),
            )
            print(f"[task {record.task_id}] parse crashed: {exc!r}")
            return

        from .service import TaskStatus

        if record.status == TaskStatus.FAILED:
            service._patch(record, finished_at=time.time())

    def parse(
        self,
        service: IngestService,
        record: TaskRecord,
        options: ParseOptions,
    ) -> None:
        """Parse assets, enrich chunks, and append them to documents JSONL."""
        from . import service as service_module

        assets = list(options.assets)
        if not assets:
            service._patch(
                record,
                status="done",
                current="no assets to parse",
                finished_at=time.time(),
            )
            return

        if record.force:
            cleared: list[str] = []
            for asset in assets:
                parsed_dir = get_parsed_dir() / asset.asset_id
                if parsed_dir.exists():
                    shutil.rmtree(parsed_dir, ignore_errors=True)
                    cleared.append(asset.asset_id)
            if cleared:
                removed = service_module._remove_asset_rows_from_documents_jsonl(set(cleared))
                scope = "failed" if record.failed_only else "all"
                service._patch(
                    record,
                    current=(
                        f"force: cleared {len(cleared)} {scope} parsed/ cache dir(s) "
                        f"({removed} documents.jsonl rows) before parse"
                    ),
                )

        service._patch(record, total=len(assets), current=f"parsing {len(assets)} asset(s)")

        failed = 0
        skipped = 0
        parsed = 0
        target = get_documents_jsonl()
        target.parent.mkdir(parents=True, exist_ok=True)
        local_statuses: dict[str, str] = {}
        for index, asset in enumerate(assets, start=1):
            if service._is_cancelled(record.task_id):
                service._patch(
                    record,
                    processed=index - 1,
                    current=f"cancelled before asset {asset.asset_id}",
                )
                return
            try:
                raw_path = get_parsed_dir() / asset.asset_id / "raw.jsonl"
                if raw_path.exists() and raw_path.stat().st_size > 0:
                    skipped += 1
                    local_statuses[asset.asset_id] = "skipped"
                    service._patch(
                        record,
                        processed=index,
                        current=f"skip cached: {asset.asset_id}",
                    )
                    continue
                try:
                    if asset.source_type == "pdf":
                        parser = service_module.get_parser("pdf", options.pdf_parser)
                        documents = parser.parse(asset)
                    elif asset.source_type == "image":
                        parser = service_module.get_parser("image", "image")
                        documents = parser.parse(
                            asset,
                            enable_ocr=options.enable_ocr,
                            enable_vlm=options.enable_vlm,
                        )
                    elif asset.source_type == "document":
                        parser = service_module.get_parser("document", options.document_parser)
                        documents = parser.parse(asset)
                    else:
                        documents = []
                except Exception as exc:
                    failed += 1
                    local_statuses[asset.asset_id] = "failed"
                    print(f"parse task failed for {asset.asset_id}: {exc}")
                    service._patch(
                        record,
                        processed=index,
                        current=f"error {asset.asset_id}: {exc}",
                    )
                    continue

                if service._settings.image_caption_enabled and documents:
                    from .image_caption import enrich_docs_with_image_captions
                    from .paths import get_captions_dir

                    caption_cache = get_captions_dir() / f"{asset.asset_id}.jsonl"
                    service._patch(
                        record,
                        current=f"image-caption: {asset.asset_id} ({len(documents)} chunks)",
                    )
                    enrich_docs_with_image_captions(
                        documents,
                        asset_id=asset.asset_id,
                        cache_path=caption_cache,
                    )
                if (options.contextual or service._settings.contextual_enabled) and documents:
                    from .contextual import enrich_docs_with_context

                    context_cache = get_parsed_dir() / asset.asset_id / "context.jsonl"
                    service._patch(
                        record,
                        current=f"contextual: {asset.asset_id} ({len(documents)} chunks)",
                    )
                    enrich_docs_with_context(
                        documents,
                        asset_title=asset.title or asset.asset_id,
                        cache_path=context_cache,
                    )
                with documents_jsonl_lock(target), target.open("a", encoding="utf-8") as file_obj:
                    for document in documents:
                        file_obj.write(json.dumps(document.to_json(), ensure_ascii=False) + "\n")
                parsed += 1
                local_statuses[asset.asset_id] = "ok"
                service._patch(
                    record,
                    processed=index,
                    current=f"parsed {asset.asset_id} ({len(documents)} doc)",
                )
            except Exception as exc:
                failed += 1
                local_statuses[asset.asset_id] = "failed"
                service._patch(
                    record,
                    processed=index,
                    current=f"error {asset.asset_id}: {exc}",
                )

        merged_statuses = {**record.asset_statuses, **local_statuses}
        if (
            service._is_cancelled(record.task_id)
            or record.status == service_module.TaskStatus.CANCELLED
        ):
            service._patch(
                record,
                status=service_module.TaskStatus.CANCELLED,
                finished_at=time.time(),
                current=f"cancelled: parsed={parsed} skipped={skipped} failed={failed}",
                asset_statuses=merged_statuses,
            )
            return
        status = (
            service_module.TaskStatus.DONE
            if failed == 0 and skipped + parsed == len(assets)
            else service_module.TaskStatus.PARTIAL
        )
        service._patch(
            record,
            status=status,
            finished_at=time.time(),
            current=f"parse {status}: parsed={parsed} skipped={skipped} failed={failed}",
            asset_statuses=merged_statuses,
        )
