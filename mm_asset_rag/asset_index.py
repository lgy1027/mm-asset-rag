"""Append-only content-hash index for confirmed assets.

Why this exists:

The upload pipeline used to call :func:`UploadPipeline.confirm` and trust
whatever ``asset_id`` and ``relative_path`` it generated. Two identical
PDFs uploaded in different sessions got two distinct ``asset_id``s and
both ended up indexed in Qdrant. With this index in place, the second
``confirm`` discovers the first by content hash and reuses the same
``asset_id`` / ``relative_path`` instead of allocating a new one.

Storage layout: ``$MM_ASSET_RAG_HOME/asset_index.jsonl`` — one JSON
object per line, ``latest row wins`` semantics. ``deleted=True`` is
recorded as another row rather than a mutation so the history is
preserved and a future compaction / migration to SQLite can fold it.

The module is intentionally narrow: it does not touch the assets
directory or Qdrant, it only records the decision. Callers are
``UploadPipeline.confirm`` (writes on confirm) and
``IngestService.delete_asset`` (writes ``deleted=True``).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import get_asset_index_path

_INDEX_LOCK = threading.Lock()


@dataclass
class AssetIndexEntry:
    asset_id: str
    sha256: str
    source_type: str
    relative_path: str
    asset_title: str = ""
    ingested_at: float = field(default_factory=time.time)
    last_task_id: str | None = None
    deleted: bool = False
    deleted_at: float | None = None
    tags: list[str] = field(default_factory=list)


def _entry_path() -> Path:
    return get_asset_index_path()


def load_entries(path: Path | None = None) -> list[AssetIndexEntry]:
    """Read every index row, tolerating corrupt / empty / missing file."""
    target = path or _entry_path()
    if not target.exists():
        return []
    entries: list[AssetIndexEntry] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            try:
                raw_tags = obj.get("tags") or []
                if not isinstance(raw_tags, list):
                    raw_tags = []
                entries.append(
                    AssetIndexEntry(
                        asset_id=str(obj.get("asset_id", "")),
                        sha256=str(obj.get("sha256", "")),
                        source_type=str(obj.get("source_type", "")),
                        relative_path=str(obj.get("relative_path", "")),
                        asset_title=str(obj.get("asset_title", "")),
                        ingested_at=float(obj.get("ingested_at") or 0.0),
                        last_task_id=(
                            str(obj["last_task_id"]) if obj.get("last_task_id") else None
                        ),
                        deleted=bool(obj.get("deleted", False)),
                        deleted_at=(
                            float(obj["deleted_at"]) if obj.get("deleted_at") is not None else None
                        ),
                        tags=[str(t) for t in raw_tags if isinstance(t, (str, int, float))],
                    )
                )
            except (TypeError, ValueError):
                continue
    return entries


def latest_by_asset_id(
    entries: list[AssetIndexEntry] | None = None,
    path: Path | None = None,
) -> dict[str, AssetIndexEntry]:
    """Fold raw rows to the latest entry per ``asset_id``."""
    source = entries if entries is not None else load_entries(path)
    latest: dict[str, AssetIndexEntry] = {}
    for entry in source:
        latest[entry.asset_id] = entry
    return latest


def find_by_sha256(
    sha256: str,
    *,
    include_deleted: bool = False,
    path: Path | None = None,
) -> AssetIndexEntry | None:
    """Return the latest entry matching ``sha256``.

    Iterates from the most recent row backwards so a ``deleted=True``
    tombstone correctly shadows any earlier non-deleted row with the
    same hash. ``include_deleted=True`` ignores tombstones.
    """
    if not sha256:
        return None
    for entry in reversed(load_entries(path)):
        if entry.sha256 != sha256:
            continue
        if not include_deleted and entry.deleted:
            return None
        return entry
    return None


def find_active_by_asset_id(
    asset_id: str,
    *,
    path: Path | None = None,
) -> AssetIndexEntry | None:
    """Return the latest non-deleted entry for ``asset_id``."""
    if not asset_id:
        return None
    for entry in reversed(load_entries(path)):
        if entry.asset_id == asset_id:
            return entry if not entry.deleted else None
    return None


def upsert_entry(entry: AssetIndexEntry, path: Path | None = None) -> None:
    """Append a new index row with fsync durability.

    The caller is expected to have computed a fresh ``ingested_at`` /
    ``last_task_id`` for the new row; we do not deduplicate inside
    ``upsert_entry`` so the JSONL history is preserved.
    """
    target = path or _entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
    with _INDEX_LOCK, target.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def mark_deleted(asset_id: str, *, path: Path | None = None, at: float | None = None) -> bool:
    """Record a ``deleted=True`` row for ``asset_id``.

    Returns ``True`` if a deletion row was actually written, ``False`` if
    the asset is unknown or already tombstoned by a more recent row.
    """
    if not asset_id:
        return False
    entries = load_entries(path)
    current = next((e for e in reversed(entries) if e.asset_id == asset_id), None)
    if current is None or current.deleted:
        return False
    upsert_entry(
        AssetIndexEntry(
            asset_id=asset_id,
            sha256=current.sha256,
            source_type=current.source_type,
            relative_path=current.relative_path,
            asset_title=current.asset_title,
            ingested_at=current.ingested_at,
            last_task_id=current.last_task_id,
            deleted=True,
            deleted_at=at if at is not None else time.time(),
            tags=list(current.tags),
        ),
        path=path,
    )
    return True


def list_active(*, path: Path | None = None) -> list[AssetIndexEntry]:
    """Return one entry per non-deleted asset, latest first."""
    latest = latest_by_asset_id(path=path)
    return sorted(
        (entry for entry in latest.values() if not entry.deleted),
        key=lambda e: e.ingested_at,
        reverse=True,
    )
