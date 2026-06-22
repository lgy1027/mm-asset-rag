"""Qdrant-backed vector store for both text and image embeddings.

Text collection carries two vector kinds for the same payload:

* **dense** — ``qwen3-embedding:4b`` (2560d), used for semantic search.
* **bm25** — sparse BM25 vectors from ``fastembed`` / ``Qdrant/bm25``,
  used for exact-token retrieval. Stored alongside dense so a single
  ``query_points`` call can RRF-fuse both ranks.

This module talks directly to ``qdrant-client``. We intentionally avoid
the ``llama-index-vector-stores-qdrant`` integration because:

- It only handles text nodes (BaseNode/TextNode); image vectors are not first-class.
- The hybrid retrieval here crosses multiple collections and image
  vectors are not first-class in LlamaIndex's ``VectorStore`` abstraction.
"""

from __future__ import annotations

import os
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from ..document_store import read_documents
from ..paths import get_assets_dir, get_indexes_dir
from ..embedders.text_embedder import EmbeddingProvider
from ..embedders.image_embedder import ImageEmbeddingProvider, ImageEmbeddingUnavailable
from ..schema import SearchHit

TEXT_COLLECTION_BASE = os.environ.get("QDRANT_TEXT_COLLECTION", "multimodal_text")
IMAGE_COLLECTION_BASE = os.environ.get("QDRANT_IMAGE_COLLECTION", "multimodal_image")

# Hybrid search tuning
BM25_MODEL_NAME = os.environ.get("QDRANT_BM25_MODEL", "Qdrant/bm25")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
HYBRID_PREFETCH_LIMIT = int(os.environ.get("QDRANT_HYBRID_PREFETCH_LIMIT", "20"))


# Module-level cache for the active collection names. Replaces
# ``os.environ["QDRANT_ACTIVE_TEXT_COLLECTION"]`` side effects which
# raced across threads and leaked into child processes.
_ACTIVE_TEXT_COLLECTION: str | None = None
_ACTIVE_IMAGE_COLLECTION: str | None = None


@lru_cache(maxsize=1)
def _bm25_embedder() -> SparseTextEmbedding:
    """Lazily load the BM25 sparse encoder (cached for the process lifetime).

    The first call downloads the ~10MB model from HuggingFace; subsequent
    calls hit the local cache. Thread-safe via a lock because fastembed's
    internal state isn't safe to share across concurrent first-time loads.
    """
    return SparseTextEmbedding(model_name=BM25_MODEL_NAME)


_BM25_LOCK = threading.Lock()


def _embed_bm25(texts: list[str]) -> list[models.SparseVector]:
    """Encode texts into BM25 sparse vectors for Qdrant sparse payload."""
    with _BM25_LOCK:
        embedder = _bm25_embedder()
        result = list(embedder.embed(texts))
    return [
        models.SparseVector(indices=enc.indices.tolist(), values=enc.values.tolist())
        for enc in result
    ]


def text_collection(vector_size: int | None = None) -> str:
    """Resolve the active text collection name.

    Without ``vector_size`` returns whatever was last set via
    ``text_collection(2560)`` (the ``QDRANT_ACTIVE_TEXT_COLLECTION``
    env-var, if set, otherwise the base name). With ``vector_size``,
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
        env = os.environ.get("QDRANT_ACTIVE_TEXT_COLLECTION")
        return env or TEXT_COLLECTION_BASE
    name = f"{TEXT_COLLECTION_BASE}_{vector_size}d"
    _ACTIVE_TEXT_COLLECTION = name
    return name


def image_collection(vector_size: int | None = None) -> str:
    """Same contract as :func:`text_collection`, for the image collection."""
    global _ACTIVE_IMAGE_COLLECTION
    if vector_size is None:
        if _ACTIVE_IMAGE_COLLECTION is not None:
            return _ACTIVE_IMAGE_COLLECTION
        env = os.environ.get("QDRANT_ACTIVE_IMAGE_COLLECTION")
        return env or IMAGE_COLLECTION_BASE
    name = f"{IMAGE_COLLECTION_BASE}_{vector_size}d"
    _ACTIVE_IMAGE_COLLECTION = name
    return name


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key)
    qdrant_path = get_indexes_dir() / "qdrant"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    _clean_stale_lock(qdrant_path)
    return QdrantClient(path=str(qdrant_path))


def _clean_stale_lock(qdrant_path: Path) -> None:
    """Remove a stale ``.lock`` from a previous crashed session.

    qdrant-client's local mode writes ``.lock`` on open and removes it on
    ``close()``. If the process is killed before close() runs (SIGKILL, OOM,
    abrupt interpreter exit), the .lock is left behind and the next startup
    fails with ``Storage folder X is already accessed by another instance of
    Qdrant client``.

    The local client does not check whether the .lock corresponds to a live
    process — it only checks file existence — so a stale lock from any source
    will block startup. We simply unlink it before opening the new client.
    Safe for single-process use; switch to ``QDRANT_URL`` (server mode) for
    concurrent access.
    """
    lock = qdrant_path / ".lock"
    if lock.exists():
        try:
            lock.unlink()
            print(f"[qdrant] removed stale .lock from previous session: {lock.name}")
        except OSError as exc:
            print(f"[qdrant] warning: could not unlink {lock}: {exc}")


def _create_collection(
    client: QdrantClient,
    name: str,
    *,
    vector_size: int,
    sparse: bool = False,
    recreate: bool = False,
) -> None:
    """Create (or recreate) a Qdrant collection with the standard config.

    - ``recreate=True`` drops the collection first; used by the explicit
      ``mmrag reindex`` command for a full rebuild.
    - ``recreate=False`` (the default) is a no-op if the collection
      already exists; used by the incremental ``build_qdrant_*_index`` path.
    - ``sparse=True`` adds the BM25 sparse vector config (text collection).
    """
    if recreate:
        if client.collection_exists(name):
            client.delete_collection(name)
    elif client.collection_exists(name):
        return

    if sparse:
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
        )
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def build_qdrant_text_index(
    batch_size: int | None = None,
    force_recreate: bool = False,
    progress_cb=None,
) -> tuple[int, str]:
    """Incrementally upsert text + BM25 sparse vectors.

    Each point id is ``uuid5("text:{asset_id}:{page}:{idx}")`` so re-running
    the index over the same ``documents.jsonl`` is a no-op for documents that
    are already indexed — only newly added documents are embedded and written.

    Args:
        batch_size: override ``QDRANT_UPSERT_BATCH_SIZE`` (default 16).
        force_recreate: drop the collection first (full rebuild). Use only
            from the explicit ``reindex`` command.
        progress_cb: optional ``callable(done: int, total: int, phase: str)``
            invoked from the worker thread for finer-grained status reporting.
    """
    documents = read_documents()
    if not documents:
        return 0, "qdrant:text:empty"

    batch_size = batch_size or max(1, int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "16")))
    embedder = EmbeddingProvider()

    # One embedding call up front to learn the vector size (= collection name).
    # On a warm cache this doc may already be in qdrant; we still need it.
    first_vector = embedder.embed(documents[0].text)
    client = get_qdrant_client()
    collection_name = text_collection(len(first_vector))

    if force_recreate:
        _create_collection(client, collection_name, vector_size=len(first_vector), sparse=True, recreate=True)
    _create_collection(client, collection_name, vector_size=len(first_vector), sparse=True)

    inserted = 0
    skipped = 0
    pending: list[models.PointStruct] = []

    def _flush() -> None:
        nonlocal inserted
        if not pending:
            return
        client.upsert(collection_name=collection_name, points=pending, wait=True)
        inserted += len(pending)
        pending.clear()

    if progress_cb:
        progress_cb(0, len(documents), "indexing")

    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        doc_keys = [
            f"text:{doc.metadata.get('asset_id', '')}:{doc.metadata.get('page')}:{offset + i}"
            for i, doc in enumerate(batch)
        ]
        point_ids = [stable_point_id(key) for key in doc_keys]

        if force_recreate:
            existing_set: set[str] = set()
        else:
            existing = client.retrieve(
                collection_name=collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False,
            )
            existing_set = {str(p.id) for p in existing}

        to_do = [i for i, pid in enumerate(point_ids) if pid not in existing_set]
        skipped += len(batch) - len(to_do)
        if not to_do:
            if progress_cb:
                progress_cb(offset + len(batch), len(documents), "skipping cached")
            continue

        texts = [batch[i].text for i in to_do]

        # Reuse the probe embedding when offset==0 and doc 0 is in to_do.
        dense_vectors: list[list[float]] = []
        start = 0
        if offset == 0 and 0 in to_do:
            dense_vectors.append(first_vector)
            start = 1
        if start < len(texts):
            dense_vectors.extend(embedder.embed_batch(texts[start:]))

        sparse_vectors = _embed_bm25(texts)

        for j, i in enumerate(to_do):
            payload = {**batch[i].metadata, "text": batch[i].text, "doc_key": doc_keys[i]}
            pending.append(
                models.PointStruct(
                    id=point_ids[i],
                    vector={
                        DENSE_VECTOR_NAME: dense_vectors[j],
                        SPARSE_VECTOR_NAME: sparse_vectors[j],
                    },
                    payload=payload,
                )
            )
        _flush()
        if progress_cb:
            progress_cb(offset + len(batch), len(documents), f"indexed {inserted}")

    return inserted, f"qdrant:{collection_name}:inserted={inserted}:skipped={skipped}"


def build_qdrant_image_index(
    force_recreate: bool = False,
    progress_cb=None,
) -> tuple[int, str]:
    """Incrementally upsert image embeddings.

    Same shape as ``build_qdrant_text_index``: existing points are skipped, only
    new images are embedded and written. ``progress_cb(done, total, phase)``
    fires from the worker thread for status reporting.
    """
    try:
        provider = ImageEmbeddingProvider()
    except ImageEmbeddingUnavailable as exc:
        return 0, f"skipped: {exc}"

    documents = read_documents()
    image_documents = [
        document for document in documents if document.metadata.get("source_type") == "image"
    ]
    if not image_documents:
        return 0, "qdrant:image:empty"

    assets_dir = get_assets_dir()
    first_path = assets_dir / str(image_documents[0].metadata["source_path"])
    first_vector = provider.embed_image(first_path)
    client = get_qdrant_client()
    collection_name = image_collection(len(first_vector))

    if force_recreate:
        _create_collection(client, collection_name, vector_size=len(first_vector), recreate=True)
    _create_collection(client, collection_name, vector_size=len(first_vector))

    # Bulk-load existing point ids (one scroll pass).
    existing_ids: set[str] = set()
    if not force_recreate:
        offset = None
        while True:
            pts, offset = client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids.update(str(p.id) for p in pts)
            if offset is None:
                break

    points: list[models.PointStruct] = []
    inserted = 0
    skipped = 0
    if progress_cb:
        progress_cb(0, len(image_documents), "indexing images")

    for index, document in enumerate(image_documents):
        point_id = stable_point_id(f"image:{document.metadata.get('asset_id')}")
        if point_id in existing_ids:
            skipped += 1
            continue
        image_path = assets_dir / str(document.metadata["source_path"])
        try:
            vector = first_vector if index == 0 else provider.embed_image(image_path)
        except Exception as exc:
            print(
                f"image index skipped ({document.metadata.get('asset_id')}): "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        payload = {**document.metadata, "text": document.text}
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
        inserted = len(points)

    if progress_cb:
        progress_cb(len(image_documents), len(image_documents), f"images indexed {inserted}")

    return inserted, f"qdrant:{collection_name}:inserted={inserted}:skipped={skipped}"


def _hybrid_text_query(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    top_k: int,
) -> list:
    """Issue a single hybrid query (dense + BM25 prefetched, fused via RRF).

    Qdrant ranks each prefetch independently, then ``Fusion.RRF`` combines
    the two ranked lists. The final ``limit`` is the top-k returned to the
    caller.
    """
    return client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=HYBRID_PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=HYBRID_PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points


def qdrant_text_search(query: str, top_k: int = 5) -> list[SearchHit]:
    embedder = EmbeddingProvider()
    client = get_qdrant_client()
    dense_query = embedder.embed(query)
    sparse_queries = _embed_bm25([query])
    sparse_query = sparse_queries[0]

    # Determine the active collection name (Qdrant active-text env var wins).
    results = _hybrid_text_query(
        client,
        text_collection(len(dense_query)),
        dense_query,
        sparse_query,
        top_k,
    )
    return [_point_to_hit("qdrant_text", point) for point in results]


def qdrant_text_to_image_search(query: str, top_k: int = 5) -> list[SearchHit]:
    try:
        provider = ImageEmbeddingProvider()
    except ImageEmbeddingUnavailable:
        return []
    client = get_qdrant_client()
    query_vector = provider.embed_text(query)
    results = client.query_points(
        collection_name=image_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_text_to_image", point) for point in results]


def qdrant_image_to_image_search(image_path: Path, top_k: int = 5) -> list[SearchHit]:
    try:
        provider = ImageEmbeddingProvider()
    except ImageEmbeddingUnavailable:
        return []
    client = get_qdrant_client()
    query_vector = provider.embed_image(image_path)
    results = client.query_points(
        collection_name=image_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_image_to_image", point) for point in results]


def _point_to_hit(route: str, point) -> SearchHit:
    payload = point.payload or {}
    return _payload_to_hit(route, float(point.score or 0.0), payload)


def _payload_to_hit(route: str, score: float, payload: dict[str, object]) -> SearchHit:
    return SearchHit(
        route=route,
        score=score,
        asset_id=str(payload.get("asset_id", "")),
        title=str(payload.get("asset_title") or payload.get("title") or ""),
        source_type=str(payload.get("source_type", "")),
        source_path=str(payload.get("source_path", "")),
        evidence=str(payload.get("text", ""))[:1000],
        metadata=dict(payload),
    )
