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
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import get_asset_index_path, get_indexes_dir

_INDEX_LOCK = threading.Lock()
_EMBEDDING_LOCK = threading.Lock()


def _dedup_threshold() -> float:
    """Read the semantic-dedup cosine threshold from env.

    Kept out of ``Settings`` on purpose: Agent A is reshaping the
    ``Settings`` dataclass in parallel, and a dedicated field here would
    cause merge churn for a single scalar. Default ``0.92`` matches the
    LlamaIndex DeduplicationModule default.
    """
    raw = os.environ.get("DEDUP_SEMANTIC_THRESHOLD", "0.92")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.92
    # Negative / NaN thresholds make no sense; fall back to the default.
    if not math.isfinite(val) or val < -1.0 or val > 1.0:
        return 0.92
    return val


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


# ── Semantic (embedding-similarity) dedup ───────────────────────────────
# Mirrors LlamaIndex's DeduplicationModule: on top of the exact content
# hash (``find_by_sha256``) we keep an asset-level title embedding index
# at ``$MM_ASSET_RAG_HOME/indexes/asset_embeddings.jsonl``. A new asset
# whose title embedding is cosine-close (default > 0.92) to an existing
# active asset — with a *different* sha256 — is treated as the same
# asset so the caller can reuse the existing ``asset_id`` and skip
# re-indexing a near-duplicate.


@dataclass
class AssetEmbeddingEntry:
    asset_id: str
    title: str
    embedding: list[float]
    model: str = ""
    dim: int | None = None
    deleted: bool = False
    deleted_at: float | None = None


def _asset_embeddings_path(explicit: Path | None = None) -> Path:
    """Resolve the asset-embedding index location.

    Defaults to ``$MM_ASSET_RAG_HOME/indexes/asset_embeddings.jsonl``.
    Tests may pass an explicit path to isolate the file from the real
    home directory.
    """
    if explicit is not None:
        return explicit
    return get_indexes_dir() / "asset_embeddings.jsonl"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-python cosine similarity (avoids a numpy hard-dependency)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    import math

    return dot / (math.sqrt(na) * math.sqrt(nb))


def _load_asset_embeddings(path: Path | None = None) -> list[AssetEmbeddingEntry]:
    """Read every embedding row; latest row wins per ``asset_id``."""
    target = _asset_embeddings_path(path)
    if not target.exists():
        return []
    latest: dict[str, AssetEmbeddingEntry] = {}
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
                emb_raw = obj.get("embedding") or []
                if not isinstance(emb_raw, list):
                    continue
                entry = AssetEmbeddingEntry(
                    asset_id=str(obj.get("asset_id", "")),
                    title=str(obj.get("title", "")),
                    embedding=[float(x) for x in emb_raw],
                    model=str(obj.get("model", "")),
                    dim=int(obj["dim"]) if obj.get("dim") is not None else None,
                    deleted=bool(obj.get("deleted", False)),
                    deleted_at=(
                        float(obj["deleted_at"]) if obj.get("deleted_at") is not None else None
                    ),
                )
            except (TypeError, ValueError):
                continue
            if not entry.asset_id:
                continue
            latest[entry.asset_id] = entry
    return [e for e in latest.values() if not e.deleted]


def find_by_semantic(
    embedding: list[float],
    *,
    threshold: float | None = None,
    exclude_sha256: str | None = None,
    path: Path | None = None,
    embeddings_path: Path | None = None,
) -> str | None:
    """Return the ``asset_id`` of an existing asset whose stored title
    embedding has cosine similarity > ``threshold`` with ``embedding``.

    Mirrors LlamaIndex's DeduplicationModule: the caller (ingest) passes
    the new asset's title (or first-chunk) embedding; if a near-duplicate
    existing asset is found, the caller reuses its ``asset_id`` instead
    of allocating a new one. ``exclude_sha256`` skips the caller's own
    content hash so an exact re-parse does not count as a semantic
    duplicate (the exact-hash path ``find_by_sha256`` already handles
    that). Returns ``None`` when no candidate clears the threshold.

    The index lives at ``$MM_ASSET_RAG_HOME/indexes/asset_embeddings.jsonl``
    (one JSON object per line, ``latest row wins`` semantics with
    ``deleted=True`` tombstones, parallel to ``asset_index.jsonl``).
    """
    if not embedding:
        return None
    cutoff = threshold if threshold is not None else _dedup_threshold()
    # Build the set of active asset_ids so we can skip embeddings whose
    # asset was hard-deleted via a tombstone that predates the
    # embedding row (defensive: the tombstone path in mark_deleted
    # already covers this, but a stale file without tombstones still
    # gets filtered here).
    active_ids: set[str] | None = None
    entries = _load_asset_embeddings(embeddings_path)
    best_id: str | None = None
    best_score = cutoff
    for entry in entries:
        if exclude_sha256:
            # Only exclude when we can confirm the candidate's sha256
            # matches the caller's own — look it up lazily once.
            if active_ids is None:
                active_ids = {
                    e.asset_id
                    for e in load_entries(path)
                    if not e.deleted and e.sha256 == exclude_sha256
                }
                # If the caller's own sha is already indexed, the
                # find_by_sha256 path should have handled reuse; here we
                # only want to avoid matching the *same* content under a
                # different asset_id, which can't happen by construction.
            if entry.asset_id in active_ids:
                continue
        score = _cosine_similarity(embedding, entry.embedding)
        if score > best_score:
            best_score = score
            best_id = entry.asset_id
    return best_id


def record_asset_embedding(
    asset_id: str,
    title: str,
    embedding: list[float],
    *,
    model: str = "",
    dim: int | None = None,
    embeddings_path: Path | None = None,
) -> None:
    """Append one embedding row to the asset-embedding index.

    Called after a successful ``upsert_entry`` write so the next ingest
    can find this asset via :func:`find_by_semantic`. ``dim`` defaults
    to ``len(embedding)`` when not supplied.
    """
    if not asset_id or not embedding:
        return
    target = _asset_embeddings_path(embeddings_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "asset_id": asset_id,
        "title": title,
        "embedding": [float(x) for x in embedding],
        "model": model,
        "dim": dim if dim is not None else len(embedding),
        "deleted": False,
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _EMBEDDING_LOCK, target.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _tombstone_asset_embedding(
    asset_id: str,
    *,
    embeddings_path: Path | None = None,
    at: float | None = None,
) -> None:
    """Append a ``deleted=True`` row for ``asset_id`` to the embedding index."""
    if not asset_id:
        return
    target = _asset_embeddings_path(embeddings_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "asset_id": asset_id,
        "title": "",
        "embedding": [],
        "model": "",
        "dim": None,
        "deleted": True,
        "deleted_at": at if at is not None else time.time(),
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _EMBEDDING_LOCK, target.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def upsert_entry(
    entry: AssetIndexEntry,
    path: Path | None = None,
    *,
    semantic_embedding: list[float] | None = None,
    semantic_model: str = "",
    semantic_dim: int | None = None,
    embeddings_path: Path | None = None,
) -> str | None:
    """Append a new index row with fsync durability.

    The caller is expected to have computed a fresh ``ingested_at`` /
    ``last_task_id`` for the new row; we do not deduplicate inside
    ``upsert_entry`` so the JSONL history is preserved.

    When ``semantic_embedding`` is provided (the new asset's title /
    first-chunk embedding), a LlamaIndex-style semantic dedup check runs
    *before* the write: if an existing active asset has cosine > the
    threshold (env ``DEDUP_SEMANTIC_THRESHOLD``, default ``0.92``) and a
    different sha256, the write is skipped and the existing
    ``asset_id`` is returned so the caller can reuse it and skip
    re-indexing a near-duplicate. When no dedup hit occurs the row is
    written and the embedding is recorded for future dedup checks.

    Without ``semantic_embedding`` the call is fully backward-compatible
    — no dedup, no embedding write. Returns the effective ``asset_id``
    (the entry's own on write, the reused one on dedup hit, or the
    entry's own when no semantic check was requested).
    """
    if semantic_embedding:
        existing_id = find_by_semantic(
            semantic_embedding,
            exclude_sha256=entry.sha256,
            path=path,
            embeddings_path=embeddings_path,
        )
        if existing_id and existing_id != entry.asset_id:
            # Semantic duplicate of an existing asset — reuse its id
            # and skip both the index write and the embedding record.
            return existing_id

    target = path or _entry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
    with _INDEX_LOCK, target.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())

    if semantic_embedding:
        record_asset_embedding(
            entry.asset_id,
            entry.asset_title,
            semantic_embedding,
            model=semantic_model,
            dim=semantic_dim,
            embeddings_path=embeddings_path,
        )
    return entry.asset_id


def mark_deleted(
    asset_id: str,
    *,
    path: Path | None = None,
    at: float | None = None,
    embeddings_path: Path | None = None,
) -> bool:
    """Record a ``deleted=True`` row for ``asset_id``.

    Returns ``True`` if a deletion row was actually written, ``False`` if
    the asset is unknown or already tombstoned by a more recent row.
    Also appends a tombstone to the asset-embedding index so future
    :func:`find_by_semantic` calls stop matching the deleted asset.
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
    _tombstone_asset_embedding(asset_id, embeddings_path=embeddings_path, at=at)
    return True


def list_active(*, path: Path | None = None) -> list[AssetIndexEntry]:
    """Return one entry per non-deleted asset, latest first."""
    latest = latest_by_asset_id(path=path)
    return sorted(
        (entry for entry in latest.values() if not entry.deleted),
        key=lambda e: e.ingested_at,
        reverse=True,
    )
