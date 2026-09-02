"""Append-only persistence for document versions and their physical assets."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .document_store import _advisory_lock
from .knowledge_models import AccessPolicy, Asset, Document, DocumentVersion, Source
from .paths import get_asset_index_path

_INDEX_LOCK = threading.RLock()
_INDEX_GUARD_STATE = threading.local()


@contextmanager
def _index_guard(target: Path, *, exclusive: bool):
    """Guard index I/O with both a process and cross-process lock.

    ``upsert_records`` calls ``upsert_record`` while holding the transaction
    lock. The thread-local depth avoids reopening the same advisory lock in
    that nested call, which would deadlock on POSIX ``flock``.
    """
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
class DocumentVersionRecord:
    """The complete persisted record written after upload confirmation."""

    document: Document
    version: DocumentVersion
    asset: Asset
    created_at: float | None = None
    access_policy: AccessPolicy = field(init=False)

    def __post_init__(self) -> None:
        if self.version.document_id != self.document.document_id:
            raise ValueError("version must belong to document")
        if self.version.content_hash != self.asset.content_hash:
            raise ValueError("version and asset content hashes must match")
        object.__setattr__(self, "access_policy", self.document.access_policy)

    def to_record(self) -> dict[str, object]:
        record = {
            "document": self.document.to_record(),
            "version": self.version.to_record(),
            "asset": self.asset.to_record(),
            "access_policy": self.access_policy.to_record(),
        }
        if self.created_at is not None:
            record["created_at"] = self.created_at
        return record


def _entry_path() -> Path:
    return get_asset_index_path()


def _record_from_dict(payload: object) -> DocumentVersionRecord | None:
    if not isinstance(payload, dict) or "asset_id" in payload:
        return None
    document_data = payload.get("document")
    version_data = payload.get("version")
    asset_data = payload.get("asset")
    policy_data = payload.get("access_policy")
    if not all(
        isinstance(value, dict) for value in (document_data, version_data, asset_data, policy_data)
    ):
        return None
    source_data = document_data.get("source")
    principals = policy_data.get("allowed_principals")
    if not isinstance(source_data, dict) or not isinstance(principals, list):
        return None
    metadata = policy_data.get("metadata", {})
    if not isinstance(metadata, dict):
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
        document = Document(
            document_id=str(document_data["document_id"]),
            title=str(document_data.get("title", "")),
            source=source,
            access_policy=policy,
        )
        version = DocumentVersion(
            document_id=str(version_data["document_id"]),
            version_number=int(version_data["version_number"]),
            content_hash=str(version_data["content_hash"]),
            version_id=str(version_data["version_id"]),
        )
        asset = Asset(
            content_hash=str(asset_data["content_hash"]),
            source_type=str(asset_data["source_type"]),
            relative_path=str(asset_data["relative_path"]),
        )
        created_at = payload.get("created_at")
        return DocumentVersionRecord(
            document=document,
            version=version,
            asset=asset,
            created_at=float(created_at) if created_at is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_records_unlocked(target: Path) -> list[DocumentVersionRecord]:
    records: list[DocumentVersionRecord] = []
    if not target.exists():
        return records
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = _record_from_dict(json.loads(line))
            except json.JSONDecodeError:
                record = None
            if record is not None:
                records.append(record)
    return records


def load_records(path: Path | None = None) -> list[DocumentVersionRecord]:
    """Load only complete v2 records; old asset-only rows are unsupported."""
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        return _load_records_unlocked(target)


def find_version(
    document_id: str, content_hash: str, *, path: Path | None = None
) -> DocumentVersionRecord | None:
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        for record in reversed(_load_records_unlocked(target)):
            if (
                record.document.document_id == document_id
                and record.version.content_hash == content_hash
            ):
                return record
    return None


def find_by_content_hash(
    content_hash: str, *, path: Path | None = None
) -> DocumentVersionRecord | None:
    """Return the latest v2 physical-asset record with this content hash."""
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        for record in reversed(_load_records_unlocked(target)):
            if record.asset.content_hash == content_hash:
                return record
    return None


def find_by_relative_path(
    relative_path: str, *, path: Path | None = None
) -> DocumentVersionRecord | None:
    """Return the latest v2 record for a canonical physical asset path."""
    if not relative_path:
        return None
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        for record in reversed(_load_records_unlocked(target)):
            if record.asset.relative_path == relative_path:
                return record
    return None


def next_version_number(document_id: str, *, path: Path | None = None) -> int:
    target = path or _entry_path()
    with _index_guard(target, exclusive=False):
        versions = [
            record.version.version_number
            for record in _load_records_unlocked(target)
            if record.document.document_id == document_id
        ]
        return max(versions, default=0) + 1


def upsert_record(record: DocumentVersionRecord, path: Path | None = None) -> DocumentVersionRecord:
    """Atomically find-or-create one document/content version record."""
    target = path or _entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _index_guard(target, exclusive=True):
        records = _load_records_unlocked(target)
        for existing in records:
            if (
                existing.document.document_id == record.document.document_id
                and existing.version.content_hash == record.version.content_hash
            ):
                return existing
        version_number = (
            max(
                (
                    item.version.version_number
                    for item in records
                    if item.document.document_id == record.document.document_id
                ),
                default=0,
            )
            + 1
        )
        persisted = DocumentVersionRecord(
            document=record.document,
            version=DocumentVersion.create(
                record.document,
                record.asset.content_hash,
                version_number=version_number,
            ),
            asset=record.asset,
            created_at=record.created_at,
        )
        line = json.dumps(persisted.to_record(), ensure_ascii=False) + "\n"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return persisted


def upsert_records(
    records: list[DocumentVersionRecord], path: Path | None = None
) -> list[DocumentVersionRecord]:
    """Find-or-create records as one atomic index transaction.

    The lock covers every idempotence check and version allocation in the
    batch. If any write fails, restore the exact pre-transaction file so a
    retry can safely move and persist the upload again.
    """
    target = path or _entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _index_guard(target, exclusive=True):
        existed = target.exists()
        snapshot = target.read_bytes() if existed else None
        try:
            return [upsert_record(record, path=target) for record in records]
        except Exception:
            try:
                if snapshot is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(snapshot)
            except OSError as rollback_error:
                raise OSError(
                    f"index transaction failed and rollback failed: {rollback_error}"
                ) from rollback_error
            raise


# ``IngestService`` is migrated in Task 5.  Retaining this import-only alias
# keeps the intermediate v2 branch importable; it does not accept or persist
# the legacy asset-only shape.
AssetIndexEntry = DocumentVersionRecord
