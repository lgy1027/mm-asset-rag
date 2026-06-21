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

from .document_store import read_documents
from .paths import get_assets_dir, get_indexes_dir
from .providers import EmbeddingProvider, ImageEmbeddingProvider, ImageEmbeddingUnavailable
from .schema import SearchHit

TEXT_COLLECTION_BASE = os.environ.get("QDRANT_TEXT_COLLECTION", "multimodal_text")
IMAGE_COLLECTION_BASE = os.environ.get("QDRANT_IMAGE_COLLECTION", "multimodal_image")

# Hybrid search tuning
BM25_MODEL_NAME = os.environ.get("QDRANT_BM25_MODEL", "Qdrant/bm25")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
HYBRID_PREFETCH_LIMIT = int(os.environ.get("QDRANT_HYBRID_PREFETCH_LIMIT", "20"))


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
    if vector_size is None:
        return os.environ.get("QDRANT_ACTIVE_TEXT_COLLECTION", TEXT_COLLECTION_BASE)
    name = f"{TEXT_COLLECTION_BASE}_{vector_size}d"
    os.environ["QDRANT_ACTIVE_TEXT_COLLECTION"] = name
    return name


def image_collection(vector_size: int | None = None) -> str:
    if vector_size is None:
        return os.environ.get("QDRANT_ACTIVE_IMAGE_COLLECTION", IMAGE_COLLECTION_BASE)
    name = f"{IMAGE_COLLECTION_BASE}_{vector_size}d"
    os.environ["QDRANT_ACTIVE_IMAGE_COLLECTION"] = name
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


def _recreate_text_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    """Create or replace the text collection with dense + BM25 sparse vectors."""
    if client.collection_exists(name):
        client.delete_collection(name)
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


def _recreate_image_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def build_qdrant_text_index(batch_size: int | None = None) -> tuple[int, str]:
    """Index text + BM25 sparse vectors for every document.

    Each point carries a dense vector (qwen3 embedding) AND a BM25
    sparse vector so a single query_points() can RRF-fuse both ranks.
    """
    documents = read_documents()
    embedder = EmbeddingProvider()
    batch_size = batch_size or max(1, int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "16")))
    first_vector = embedder.embed_text(documents[0].text)
    client = get_qdrant_client()
    collection_name = text_collection(len(first_vector))
    _recreate_text_collection(client, collection_name, len(first_vector))

    points: list[models.PointStruct] = []
    texts_buffer: list[str] = []
    metas_buffer: list[dict] = []
    ids_buffer: list[str] = []

    def _flush() -> None:
        if not points:
            return
        client.upsert(collection_name=collection_name, points=points, wait=True)
        points.clear()
        texts_buffer.clear()
        metas_buffer.clear()
        ids_buffer.clear()

    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        texts = [d.text for d in batch]
        # dense vectors
        if offset == 0:
            dense_vectors = [first_vector] + embedder.embed_texts(texts[1:])
        else:
            dense_vectors = embedder.embed_texts(texts)
        # BM25 sparse vectors (same batch)
        sparse_vectors = _embed_bm25(texts)

        for (doc, dense_vec, sparse_vec), idx in zip(
            zip(batch, dense_vectors, sparse_vectors),
            range(offset, offset + len(batch)),
        ):
            payload = {**doc.metadata, "text": doc.text}
            point_id = stable_point_id(
                f"text:{doc.metadata.get('asset_id')}:{doc.metadata.get('page')}:{idx}"
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload=payload,
                )
            )
        _flush()
    return len(documents), f"qdrant:{collection_name}"


def build_qdrant_image_index() -> tuple[int, str]:
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
    _recreate_image_collection(client, collection_name, len(first_vector))

    points = []
    for index, document in enumerate(image_documents):
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
        point_id = stable_point_id(f"image:{document.metadata.get('asset_id')}")
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(points), f"qdrant:{collection_name}"


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
    dense_query = embedder.embed_text(query)
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
