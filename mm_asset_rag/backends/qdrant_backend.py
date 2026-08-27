"""Compatibility facade for the responsibility-split Qdrant adapter.

New code should import :mod:`mm_asset_rag.backends.qdrant` or depend on the
backend capability ports. Legacy public imports remain available here.
"""

from __future__ import annotations

from pathlib import Path

from ..retrieval import RRF_K
from .qdrant import client, collections, indexing, search

QdrantLockHeldError = client.QdrantLockHeldError

TEXT_COLLECTION_BASE = collections.TEXT_COLLECTION_BASE
IMAGE_COLLECTION_BASE = collections.IMAGE_COLLECTION_BASE
DENSE_VECTOR_NAME = collections.DENSE_VECTOR_NAME
SPARSE_VECTOR_NAME = collections.SPARSE_VECTOR_NAME
EMBED_SPARSE_VECTOR_NAME = collections.EMBED_SPARSE_VECTOR_NAME
EMBED_COLBERT_VECTOR_NAME = collections.EMBED_COLBERT_VECTOR_NAME
BM25_MODEL_NAME = indexing.BM25_MODEL_NAME
HYBRID_PREFETCH_LIMIT = search.HYBRID_PREFETCH_LIMIT


def reset_qdrant_client_cache() -> None:
    client.reset_qdrant_client_cache()


def get_qdrant_client():
    return client.get_qdrant_client()


def invalidate_bm25_zh_idf_cache() -> None:
    indexing.invalidate_bm25_zh_idf_cache()


def text_collection(vector_size: int | None = None) -> str:
    return collections.text_collection(vector_size)


def image_collection(vector_size: int | None = None) -> str:
    return collections.image_collection(vector_size)


def stable_point_id(value: str) -> str:
    return indexing.stable_point_id(value)


def build_qdrant_text_index(
    batch_size: int | None = None,
    force_recreate: bool = False,
    progress_cb=None,
):
    return indexing.build_text_index(
        batch_size=batch_size,
        force_recreate=force_recreate,
        progress_cb=progress_cb,
    )


def build_qdrant_image_index(force_recreate: bool = False, progress_cb=None):
    return indexing.build_image_index(
        force_recreate=force_recreate,
        progress_cb=progress_cb,
    )


def qdrant_text_search(
    query: str,
    top_k: int = 5,
    *,
    include_image_sources: bool = False,
):
    if include_image_sources:
        return search.text_search(query, top_k=top_k, include_image_sources=True)
    return search.text_search(query, top_k=top_k)


def qdrant_text_to_image_search(query: str, top_k: int = 5):
    return search.text_to_image_search(query, top_k=top_k)


def qdrant_image_to_image_search(image_path: Path, top_k: int = 5):
    return search.image_to_image_search(image_path, top_k=top_k)


def delete_points_by_asset_id(
    asset_id: str,
    *,
    text: bool = True,
    image: bool = True,
):
    return collections.delete_points_by_asset_id(asset_id, text=text, image=image)


# Private compatibility aliases retained for existing in-repository callers
# and focused regression tests. New code should import their owning module.
_clean_stale_lock = client._clean_stale_lock
_probe_lock_holder = client._probe_lock_holder
_lock_holder_pid = client._lock_holder_pid
_pid_alive = client._pid_alive
_create_collection = collections._create_collection
_existing_collections_for = collections._existing_collections_for
_bm25_embedder = indexing._bm25_embedder
_embed_bm25 = indexing._embed_bm25
_load_bm25_zh_idf = indexing._load_bm25_zh_idf
_embed_bm25_zh_query = search._embed_bm25_zh_query
_embedder_sparse_capability = indexing._embedder_sparse_capability
_embedder_colbert_capability = indexing._embedder_colbert_capability
_tokenize_for_bm25 = indexing._tokenize_for_bm25
_bm25_okapi_scores = indexing._bm25_okapi_scores
_select_top_chunks_per_pdf = indexing._select_top_chunks_per_pdf
_hybrid_text_query = search._hybrid_text_query
_filter_by_relevance = search._filter_by_relevance
_is_collection_missing = search._is_collection_missing
_point_to_hit = search._point_to_hit
_payload_to_hit = search._payload_to_hit


def __getattr__(name: str):
    """Forward read-only compatibility access to relocated module state."""
    owner = {
        "_QDRANT_CLIENT": client,
        "_QDRANT_CLIENT_KEY": client,
        "_QDRANT_CLIENT_LOCK": client,
        "_ACTIVE_TEXT_COLLECTION": collections,
        "_ACTIVE_IMAGE_COLLECTION": collections,
        "_BM25_ZH_IDF_CACHE": indexing,
        "_BM25_ZH_IDF_LOCK": indexing,
    }.get(name)
    if owner is None:
        raise AttributeError(name)
    return getattr(owner, name)


__all__ = [
    "BM25_MODEL_NAME",
    "DENSE_VECTOR_NAME",
    "EMBED_COLBERT_VECTOR_NAME",
    "EMBED_SPARSE_VECTOR_NAME",
    "HYBRID_PREFETCH_LIMIT",
    "IMAGE_COLLECTION_BASE",
    "RRF_K",
    "SPARSE_VECTOR_NAME",
    "TEXT_COLLECTION_BASE",
    "QdrantLockHeldError",
    "build_qdrant_image_index",
    "build_qdrant_text_index",
    "delete_points_by_asset_id",
    "get_qdrant_client",
    "image_collection",
    "invalidate_bm25_zh_idf_cache",
    "qdrant_image_to_image_search",
    "qdrant_text_search",
    "qdrant_text_to_image_search",
    "reset_qdrant_client_cache",
    "stable_point_id",
    "text_collection",
]
