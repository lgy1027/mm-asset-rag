"""Qdrant collection naming, schema validation, and collection lifecycle."""

from __future__ import annotations

import re

from qdrant_client import QdrantClient, models

from ...settings import get_settings
from .client import get_qdrant_client

TEXT_COLLECTION_BASE = get_settings().qdrant_text_collection
IMAGE_COLLECTION_BASE = get_settings().qdrant_image_collection
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
EMBED_SPARSE_VECTOR_NAME = "embed_sparse"
EMBED_COLBERT_VECTOR_NAME = "embed_colbert"

_ACTIVE_TEXT_COLLECTION: str | None = None
_ACTIVE_IMAGE_COLLECTION: str | None = None


def text_collection(vector_size: int | None = None) -> str:
    """Resolve the active text collection name.

    Without ``vector_size`` returns whatever was last set via
    ``text_collection(2560)`` (the ``qdrant_active_text_collection``
    setting, if set, otherwise the base name). With ``vector_size``,
    sets the active collection to ``f"{base}_{vector_size}d"`` and
    returns it.

    The "active collection" is cached in module state instead of
    ``os.environ`` so concurrent threads don't race on a process-wide
    variable, and tests can reset it without touching the real
    environment.
    """
    global _ACTIVE_TEXT_COLLECTION
    if vector_size is None:
        if _ACTIVE_TEXT_COLLECTION is not None:
            return _ACTIVE_TEXT_COLLECTION
        return get_settings().qdrant_active_text_collection or TEXT_COLLECTION_BASE
    name = f"{TEXT_COLLECTION_BASE}_{vector_size}d"
    _ACTIVE_TEXT_COLLECTION = name
    return name


def image_collection(vector_size: int | None = None) -> str:
    """Same contract as :func:`text_collection`, for the image collection."""
    global _ACTIVE_IMAGE_COLLECTION
    if vector_size is None:
        if _ACTIVE_IMAGE_COLLECTION is not None:
            return _ACTIVE_IMAGE_COLLECTION
        return get_settings().qdrant_active_image_collection or IMAGE_COLLECTION_BASE
    name = f"{IMAGE_COLLECTION_BASE}_{vector_size}d"
    _ACTIVE_IMAGE_COLLECTION = name
    return name


def _create_collection(
    client: QdrantClient,
    name: str,
    *,
    vector_size: int,
    sparse: bool = False,
    recreate: bool = False,
    embed_sparse: bool = False,
    embed_colbert: bool = False,
    colbert_dim: int | None = None,
) -> None:
    """Create (or recreate) a Qdrant collection with the standard config.

    - ``recreate=True`` drops the collection first; used by the explicit
      ``mmrag reindex`` command for a full rebuild.
    - ``recreate=False`` (the default) is a no-op if the collection
      already exists; used by the incremental ``build_qdrant_*_index`` path.
    - ``sparse=True`` adds the BM25 sparse vector config (text collection).
      When ``bm25_zh_enabled`` is set on :class:`Settings`, a second
      Chinese sparse vector (``bm25_zh``) is added so Chinese docs and
      queries get token-level recall via a jieba-based Okapi BM25.
    - ``embed_sparse=True`` adds an extra sparse vector field
      (``embed_sparse``) populated from the embedder's native sparse
      output (bge-m3). Only when the embedder supports it; the
      OpenAI-compatible ``TextEmbedder`` never sets this so the
      default schema is unchanged.
    - ``embed_colbert=True`` adds a multi-vector field
      (``embed_colbert``) for late-interaction retrieval. ``colbert_dim``
      is the per-token vector dim (required when ``embed_colbert`` is
      true). Again only when the embedder supports it.
    """
    if recreate:
        if client.collection_exists(name):
            client.delete_collection(name)
    elif client.collection_exists(name):
        # Schema check: the existing collection must carry every sparse
        # and multi-vector the current Settings expect. Catches the
        # silent-skip-after-upgrade footgun where adding a new field
        # (e.g. ``bm25_zh`` / ``embed_sparse`` / ``embed_colbert``)
        # would otherwise leave the old collection in place while the
        # indexer tries to write richer points.
        if sparse:
            info = client.get_collection(name)
            existing_sparse = set((info.config.params.sparse_vectors or {}).keys())
            existing_multi = set((info.config.params.vectors or {}).keys())
            settings = get_settings()
            expected_sparse: set[str] = {SPARSE_VECTOR_NAME}
            if settings.bm25_zh_enabled:
                expected_sparse.add(settings.bm25_zh_vector_name)
            if embed_sparse:
                expected_sparse.add(EMBED_SPARSE_VECTOR_NAME)
            expected_multi: set[str] = {DENSE_VECTOR_NAME}
            if embed_colbert:
                expected_multi.add(EMBED_COLBERT_VECTOR_NAME)
            missing_sparse = sorted(expected_sparse - existing_sparse)
            unexpected_sparse = sorted(existing_sparse - expected_sparse)
            missing_multi = sorted(expected_multi - existing_multi)
            unexpected_multi = sorted(existing_multi - expected_multi)
            if missing_sparse or unexpected_sparse or missing_multi or unexpected_multi:
                raise RuntimeError(
                    f"Qdrant collection '{name}' schema mismatch.\n"
                    f"  expected sparse vectors: {sorted(expected_sparse)}\n"
                    f"  actual sparse vectors:   {sorted(existing_sparse)}\n"
                    f"  missing sparse:   {missing_sparse or '(none)'}\n"
                    f"  unexpected sparse: {unexpected_sparse or '(none)'}\n"
                    f"  expected vectors: {sorted(expected_multi)}\n"
                    f"  actual vectors:   {sorted(existing_multi)}\n"
                    f"  missing vectors:   {missing_multi or '(none)'}\n"
                    f"  unexpected vectors: {unexpected_multi or '(none)'}\n"
                    f"Run `mmrag reindex` to rebuild the collection with the "
                    f"current Settings (it is drop+rebuild by default)."
                )
        return

    if sparse:
        sparse_config: dict[str, models.SparseVectorParams] = {
            SPARSE_VECTOR_NAME: models.SparseVectorParams(),
        }
        settings = get_settings()
        if settings.bm25_zh_enabled:
            sparse_config[settings.bm25_zh_vector_name] = models.SparseVectorParams()
        if embed_sparse:
            sparse_config[EMBED_SPARSE_VECTOR_NAME] = models.SparseVectorParams()
        vectors_config: dict[str, models.VectorParams] = {
            DENSE_VECTOR_NAME: models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        }
        if embed_colbert and colbert_dim:
            vectors_config[EMBED_COLBERT_VECTOR_NAME] = models.VectorParams(
                size=colbert_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


def _existing_collections_for(client: QdrantClient, base: str) -> list[str]:
    """Return the Qdrant collections that exist for a base name.

    Matches the bare base name (``multimodal_text``) and any dim-suffixed
    variant (``multimodal_text_1024d``) produced by
    :func:`text_collection`/`image_collection` when a vector size is set.
    Different embedders over time leave several ``_<dim>d`` collections, so
    this returns a list — callers (delete, count) want to touch *all* of them
    rather than whichever the process happens to have cached as active.

    Resolving from the live server (not the ``_ACTIVE_*`` module cache) is what
    makes ``delete_points_by_asset_id`` correct when run in a process that never
    ingested (e.g. ``mmrag delete``): without it, ``text_collection()`` falls
    back to the bare base name and ``client.delete`` raises "Collection not
    found", silently leaving the points behind.

    Returns an empty list on any error (server down, unexpected response) so
    the caller can decide how to record it rather than raising mid-cleanup.
    """
    if not base:
        return []
    try:
        names = [c.name for c in client.get_collections().collections]
    except Exception as exc:  # pragma: no cover — server-down / network
        print(f"[qdrant] get_collections failed for base={base!r}: {exc}")
        return []
    pattern = re.compile(rf"{re.escape(base)}_(\d+)d")
    matched = [n for n in names if n == base or pattern.fullmatch(n)]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in matched:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def delete_points_by_asset_id(
    asset_id: str,
    *,
    text: bool = True,
    image: bool = True,
) -> dict[str, int]:
    """Delete every Qdrant point whose payload carries ``asset_id``.

    Returns a small ``{"text": N, "image": M}`` map with the number of
    collections that were actually scanned and deleted from. Failures are
    logged but do not raise so the caller's overall ``delete_asset`` cleanup
    can still complete.

    Collections are resolved from the live Qdrant server (via
    :func:`_existing_collections_for`), **not** from the module's active-cache.
    This is the fix for ``text_collections_scanned: 0`` — a ``mmrag delete``
    run in a process that never ingested would otherwise target the bare base
    collection name (``multimodal_text``) which does not exist (the real name
    is ``multimodal_text_<dim>d``), and the resulting "Collection not found"
    was silently swallowed, leaving the points behind to pollute retrieval.

    When ``qdrant_active_text_collection`` / ``qdrant_active_image_collection``
    is set (a user explicitly pinning a collection for migration), that single
    name is used verbatim instead of listing — preserving the pin intent.
    """
    if not asset_id:
        return {"text": 0, "image": 0}
    selector = models.FilterSelector(
        filter=models.Filter(
            must=[models.FieldCondition(key="asset_id", match=models.MatchValue(value=asset_id))]
        )
    )
    counts = {"text": 0, "image": 0}
    client = get_qdrant_client()
    settings = get_settings()
    if text:
        pinned = settings.qdrant_active_text_collection
        cols = [pinned] if pinned else _existing_collections_for(client, TEXT_COLLECTION_BASE)
        for col in cols:
            try:
                client.delete(collection_name=col, points_selector=selector)
                counts["text"] += 1
            except Exception as exc:
                print(f"[qdrant] failed to delete text points for {asset_id} in {col}: {exc}")
    if image:
        pinned = settings.qdrant_active_image_collection
        cols = [pinned] if pinned else _existing_collections_for(client, IMAGE_COLLECTION_BASE)
        for col in cols:
            try:
                client.delete(collection_name=col, points_selector=selector)
                counts["image"] += 1
            except Exception as exc:
                print(f"[qdrant] failed to delete image points for {asset_id} in {col}: {exc}")
    return counts


create_collection = _create_collection
