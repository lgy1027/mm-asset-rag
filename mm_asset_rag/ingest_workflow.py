"""Parse, enrich, document-persist, and index orchestration."""

from __future__ import annotations

import shutil
import time
from typing import TYPE_CHECKING

from .asset_index import find_by_relative_path
from .document_store import append_documents
from .knowledge_models import Chunk
from .paths import get_documents_jsonl, get_parsed_dir, physical_cache_id
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
            version_id: status
            for version_id, status in record.version_statuses.items()
            if status in {"ok", "skipped"}
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
            new_statuses = dict(record.version_statuses)
            for version_id in indexed_targets:
                new_statuses[version_id] = "indexed"
            if (
                service._is_cancelled(record.task_id)
                or record.status == service_module.TaskStatus.CANCELLED
            ):
                service._patch(
                    record,
                    status=service_module.TaskStatus.CANCELLED,
                    finished_at=time.time(),
                    current=f"cancelled after index · text={text_n} image={image_n}",
                    version_statuses=new_statuses,
                )
                return
            service._patch(
                record,
                current=f"index built · text={text_n} image={image_n}",
                status=parse_status,
                finished_at=time.time(),
                version_statuses=new_statuses,
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
            new_statuses = dict(record.version_statuses)
            for version_id in indexed_targets:
                new_statuses[version_id] = "failed_index"
            service._patch(
                record,
                current=f"index crashed: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
                status="failed",
                finished_at=time.time(),
                version_statuses=new_statuses,
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
            versions_by_document: dict[str, set[str]] = {}
            for asset in assets:
                version_record = find_by_relative_path(asset.relative_path)
                if version_record is None:
                    continue
                cache_key = physical_cache_id(version_record.asset.relative_path)
                parsed_dir = get_parsed_dir() / cache_key
                if parsed_dir.exists():
                    shutil.rmtree(parsed_dir, ignore_errors=True)
                    cleared.append(version_record.version.version_id)
                versions_by_document.setdefault(version_record.document.document_id, set()).add(
                    version_record.version.version_id
                )
            if versions_by_document:
                removed = sum(
                    service_module._remove_document_version_rows_from_documents_jsonl(
                        document_id,
                        version_ids,
                    )
                    for document_id, version_ids in versions_by_document.items()
                )
                scope = "failed" if record.failed_only else "all"
                service._patch(
                    record,
                    current=(
                        f"force: refreshed {sum(map(len, versions_by_document.values()))} "
                        f"{scope} document version(s), cleared {len(cleared)} cache dir(s) "
                        f"and removed {removed} chunk row(s) before parse"
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
            version_record = find_by_relative_path(asset.relative_path)
            if version_record is None:
                raise ValueError(
                    f"no persisted document version for asset path {asset.relative_path!r}"
                )
            status_key = version_record.version.version_id
            cache_key = physical_cache_id(version_record.asset.relative_path)
            if service._is_cancelled(record.task_id):
                service._patch(
                    record,
                    processed=index - 1,
                    current=f"cancelled before version {status_key}",
                )
                return
            try:
                raw_path = get_parsed_dir() / cache_key / "raw.jsonl"
                if raw_path.exists() and raw_path.stat().st_size > 0:
                    skipped += 1
                    local_statuses[status_key] = "skipped"
                    service._patch(
                        record,
                        processed=index,
                        current=f"skip cached version: {status_key}",
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
                    local_statuses[status_key] = "failed"
                    print(f"parse task failed for version {status_key}: {exc}")
                    service._patch(
                        record,
                        processed=index,
                        current=f"error {status_key}: {exc}",
                    )
                    continue

                if service._settings.image_caption_enabled and documents:
                    from .image_caption import enrich_docs_with_image_captions
                    from .paths import get_captions_dir

                    caption_cache = get_captions_dir() / f"{cache_key}.jsonl"
                    service._patch(
                        record,
                        current=f"image-caption: {status_key} ({len(documents)} chunks)",
                    )
                    enrich_docs_with_image_captions(
                        documents,
                        asset_id=cache_key,
                        cache_path=caption_cache,
                    )
                if options.contextual and documents:
                    from .contextual import enrich_docs_with_context

                    context_cache = get_parsed_dir() / cache_key / "context.jsonl"
                    service._patch(
                        record,
                        current=f"contextual: {status_key} ({len(documents)} chunks)",
                    )
                    enrich_docs_with_context(
                        documents,
                        asset_title=asset.title or asset.asset_id,
                        cache_path=context_cache,
                    )
                chunks = self._to_chunks(asset, documents, version_record=version_record)
                append_documents(chunks, path=target)
                parsed += 1
                local_statuses[status_key] = "ok"
                service._patch(
                    record,
                    processed=index,
                    current=f"parsed {status_key} ({len(documents)} chunks)",
                )
            except Exception as exc:
                failed += 1
                local_statuses[status_key] = "failed"
                service._patch(
                    record,
                    processed=index,
                    current=f"error {status_key}: {exc}",
                )

        merged_statuses = {**record.version_statuses, **local_statuses}
        if (
            service._is_cancelled(record.task_id)
            or record.status == service_module.TaskStatus.CANCELLED
        ):
            service._patch(
                record,
                status=service_module.TaskStatus.CANCELLED,
                finished_at=time.time(),
                current=f"cancelled: parsed={parsed} skipped={skipped} failed={failed}",
                version_statuses=merged_statuses,
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
            version_statuses=merged_statuses,
        )

    def _to_chunks(self, asset, documents: list, *, version_record=None) -> list[Chunk]:
        """Attach persisted document identity to parser-owned chunks."""
        from .schema import ParsedChunk

        if not documents:
            return []
        record = version_record or find_by_relative_path(asset.relative_path)
        if record is None:
            raise ValueError(
                f"no persisted document version for asset path {asset.relative_path!r}"
            )
        chunks: list[Chunk] = []
        for ordinal, document in enumerate(documents):
            if not isinstance(document, ParsedChunk):
                raise TypeError("parser output must be ParsedChunk")
            chunks.append(
                Chunk.create(
                    document_version=record.version,
                    asset=record.asset,
                    ordinal=ordinal,
                    text=document.text,
                    source=record.document.source,
                    access_policy=record.document.access_policy,
                    metadata=_without_asset_identity(document.metadata),
                )
            )
        return chunks


def _without_asset_identity(value):
    """Remove obsolete physical identity keys before chunk persistence."""
    if isinstance(value, dict):
        return {
            key: _without_asset_identity(item) for key, item in value.items() if key != "asset_id"
        }
    if isinstance(value, list):
        return [_without_asset_identity(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_asset_identity(item) for item in value)
    return value
