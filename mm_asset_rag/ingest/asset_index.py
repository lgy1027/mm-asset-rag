"""Persistence for each document's current physical asset."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..core.knowledge_models import AccessPolicy, Asset, Document, Source
from ..core.paths import get_asset_index_path
from .document_store import _advisory_lock

_INDEX_LOCK = threading.RLock()
_INDEX_GUARD_STATE = threading.local()


@contextmanager
def _index_guard(target: Path, *, exclusive: bool):
    with _INDEX_LOCK:
        depth = getattr(_INDEX_GUARD_STATE, "depth", 0)
        if depth:
            yield
            return
        _INDEX_GUARD_STATE.depth = depth + 1
        try:
            with _advisory_lock(target, exclusive=exclusive):
                yield
        finally:
            _INDEX_GUARD_STATE.depth = depth


@dataclass(frozen=True)
class DocumentRecord:
    """One logical document and its current physical asset."""

    document: Document
    asset: Asset
    created_at: float | None = None
    access_policy: AccessPolicy = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "access_policy", self.document.access_policy)

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "document": self.document.to_record(),
            "asset": self.asset.to_record(),
            "access_policy": self.access_policy.to_record(),
        }
        if self.created_at is not None:
            record["created_at"] = self.created_at
        return record


def _entry_path() -> Path:
    return get_asset_index_path()


def _record_from_dict(payload: object) -> DocumentRecord | None:
    if not isinstance(payload, dict) or "version" in payload or "asset_id" in payload:
        return None
    document_data = payload.get("document")
    asset_data = payload.get("asset")
    policy_data = payload.get("access_policy")
    if not all(isinstance(value, dict) for value in (document_data, asset_data, policy_data)):
        return None
    source_data = document_data.get("source")
    principals = policy_data.get("allowed_principals")
    metadata = policy_data.get("metadata", {})
    if (
        not isinstance(source_data, dict)
        or not isinstance(principals, list)
        or not isinstance(metadata, dict)
    ):
        return None
    try:
        source = Source(
            source_id=str(source_data["source_id"]),
            uri=str(source_data.get("uri", "")),
            provider=str(source_data.get("provider", "upload")),
        )
        policy = AccessPolicy(
            collection=str(policy_data["collection"]),
            allowed_principals=tuple(str(value) for value in principals),
            metadata=metadata,
        )
        return DocumentRecord(
            document=Document(
                document_id=str(document_data["document_id"]),
                title=str(document_data.get("title", "")),
                source=source,
                access_policy=policy,
            ),
            asset=Asset(
                content_hash=str(asset_data["content_hash"]),
                source_type=str(asset_data["source_type"]),
                relative_path=str(asset_data["relative_path"]),
            ),
            created_at=float(payload["created_at"])
            if payload.get("created_at") is not None
            else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_records_unlocked(target: Path) -> list[DocumentRecord]:
    if not target.exists():
        return []
    records: list[DocumentRecord] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = _record_from_dict(json.loads(line))
            except json.JSONDecodeError:
                record = None
            if record is not None:
                records.append(record)
    return records


def load_records(path: Path | None = None) -> list[DocumentRecord]:
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        return _load_records_unlocked(target)


def find_document(document_id: str, *, path: Path | None = None) -> DocumentRecord | None:
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        return next(
            (
                record
                for record in reversed(_load_records_unlocked(target))
                if record.document.document_id == document_id
            ),
            None,
        )


def find_by_content_hash(content_hash: str, *, path: Path | None = None) -> DocumentRecord | None:
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        return next(
            (
                record
                for record in reversed(_load_records_unlocked(target))
                if record.asset.content_hash == content_hash
            ),
            None,
        )


def find_by_relative_path(relative_path: str, *, path: Path | None = None) -> DocumentRecord | None:
    if not relative_path:
        return None
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        return next(
            (
                record
                for record in reversed(_load_records_unlocked(target))
                if record.asset.relative_path == relative_path
            ),
            None,
        )


def _write_records(target: Path, records: list[DocumentRecord]) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_record(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def upsert_record(record: DocumentRecord, path: Path | None = None) -> DocumentRecord:
    """Replace the current asset for a document atomically."""
    target = path or _entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _index_guard(target, exclusive=True):
        records = [
            item
            for item in _load_records_unlocked(target)
            if item.document.document_id != record.document.document_id
        ]
        records.append(record)
        _write_records(target, records)
    return record


def upsert_records(records: list[DocumentRecord], path: Path | None = None) -> list[DocumentRecord]:
    target = path or _entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _index_guard(target, exclusive=True):
        replacement_ids = {record.document.document_id for record in records}
        current = [
            item
            for item in _load_records_unlocked(target)
            if item.document.document_id not in replacement_ids
        ]
        by_document = {record.document.document_id: record for record in records}
        current.extend(by_document.values())
        _write_records(target, current)
    return records
