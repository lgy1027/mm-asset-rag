"""Vector-store backend implementations.

Adding a new backend (Milvus, Pinecone, ...) is:

1. Drop ``milvus_backend.py`` here whose class satisfies
   ``mm_asset_rag.protocols.VectorBackend``.
2. ``register_backend(...)`` below in this ``__init__``.
3. Make the backend selectable via ``VECTOR_BACKEND`` env (handled in
   ``mm_asset_rag.service``).
"""

from __future__ import annotations

from ..registry import register_backend
from .qdrant_backend import (
    build_qdrant_image_index,
    build_qdrant_text_index,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)

__all__ = [
    "build_qdrant_image_index",
    "build_qdrant_text_index",
    "qdrant_image_to_image_search",
    "qdrant_text_search",
    "qdrant_text_to_image_search",
]


class _QdrantBackend:
    """Thin facade around ``qdrant_store`` exposing only the operations the
    codebase needs. Phase 4 (service extraction) will move the actual
    orchestration logic into ``IngestService``; for now this just re-exports
    the free functions so the registry has a single concrete instance to hold.
    """

    name = "qdrant"

    def ensure_collection(self, *, name, dim, sparse=False):
        from .qdrant_backend import _ensure_text_collection, _ensure_image_collection
        if sparse:
            _ensure_text_collection(self._client(), name, dim)
        else:
            _ensure_image_collection(self._client(), name, dim)

    def drop_collection(self, name):
        from qdrant_client import QdrantClient
        from ..paths import get_indexes_dir
        client = QdrantClient(path=str(get_indexes_dir() / "qdrant"))
        client.delete_collection(name)

    def upsert(self, *, collection, points, wait=True):
        from ..paths import get_indexes_dir
        from qdrant_client import QdrantClient
        client = QdrantClient(path=str(get_indexes_dir() / "qdrant"))
        client.upsert(collection_name=collection, points=points, wait=wait)
        return len(points)

    def retrieve_existing_ids(self, *, collection, ids):
        from ..paths import get_indexes_dir
        from qdrant_client import QdrantClient
        client = QdrantClient(path=str(get_indexes_dir() / "qdrant"))
        existing = client.retrieve(
            collection_name=collection, ids=ids, with_payload=False, with_vectors=False
        )
        return {str(p.id) for p in existing}

    def search_points(self, *, collection, query_vector, sparse_vector, vector_name_dense,
                       vector_name_sparse, top_k):
        from .qdrant_backend import _hybrid_text_query, get_qdrant_client
        client = get_qdrant_client()
        return _hybrid_text_query(
            client, collection, query_vector, sparse_vector, top_k
        )

    def search_image_to_image(self, *, collection, image_path, top_k):
        return qdrant_image_to_image_search(image_path, top_k=top_k)

    @staticmethod
    def _client():
        from .qdrant_backend import get_qdrant_client
        return get_qdrant_client()


register_backend(_QdrantBackend())