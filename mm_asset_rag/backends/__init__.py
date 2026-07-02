"""Vector-store backend implementations.

Adding a new backend (Milvus, Pinecone, …) is:

1. Drop ``milvus_backend.py`` here whose class satisfies
   ``mm_asset_rag.protocols.VectorBackend``.
2. ``register_backend(...)`` below in this ``__init__``.

Today only Qdrant is wired. The Qdrant implementation is a thin facade
that delegates every operation to the free functions in
``qdrant_backend.py``, so the index / search logic stays in one place
regardless of how callers reach it (``backend.upsert_text()`` or the
module-level ``build_qdrant_text_index()``).
"""

from __future__ import annotations

from pathlib import Path

from ..registry import register_backend
from .qdrant_backend import (
    build_qdrant_image_index,
    build_qdrant_text_index,
    get_qdrant_client,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)


class _QdrantBackend:
    """Facade that satisfies ``VectorBackend`` (generic) and adds
    modality-specific helpers for the two index paths the codebase uses.

    All methods delegate to free functions in ``qdrant_backend.py`` so the
    index / search logic lives in exactly one place. Adding a new
    vector backend means replacing this class with one whose methods
    call into that backend's SDK.
    """

    name = "qdrant"

    # ─── Generic ``VectorBackend`` Protocol ────────────────────────────

    def ensure_collection(self, *, name, dim, sparse=False):
        from .qdrant_backend import _create_collection

        _create_collection(self._client(), name, vector_size=dim, sparse=sparse)

    def drop_collection(self, name):
        client = self._client()
        client.delete_collection(name)

    def upsert(self, *, collection, points, wait=True):
        client = self._client()
        client.upsert(collection_name=collection, points=points, wait=wait)
        return len(points)

    def retrieve_existing_ids(self, *, collection, ids):
        client = self._client()
        existing = client.retrieve(
            collection_name=collection, ids=ids, with_payload=False, with_vectors=False
        )
        return {str(p.id) for p in existing}

    def search_points(
        self,
        *,
        collection,
        query_vector,
        sparse_vector,
        vector_name_dense,
        vector_name_sparse,
        top_k,
    ):
        from .qdrant_backend import _hybrid_text_query

        return _hybrid_text_query(self._client(), collection, query_vector, sparse_vector, top_k)

    def search_image_to_image(self, *, collection, image_path, top_k):
        return qdrant_image_to_image_search(image_path, top_k=top_k)

    # ─── Modality-specific helpers (called by service + api + cli) ─────

    def upsert_text(self, *, progress_cb=None, force_recreate=False):
        """Build the text index incrementally; returns ``(count, name)``.

        ``force_recreate=True`` drops the collection first — used by the
        ``reindex`` command for a clean rebuild.
        """
        return build_qdrant_text_index(progress_cb=progress_cb, force_recreate=force_recreate)

    def upsert_image(self, *, progress_cb=None, force_recreate=False):
        """Build the image index incrementally; returns ``(count, name)``."""
        return build_qdrant_image_index(progress_cb=progress_cb, force_recreate=force_recreate)

    def search_text(self, *, query, top_k):
        return qdrant_text_search(query, top_k=top_k)

    def search_text_to_image(self, *, query, top_k):
        return qdrant_text_to_image_search(query, top_k=top_k)

    def search_image(self, *, image_path, top_k):
        return qdrant_image_to_image_search(Path(image_path), top_k=top_k)

    # ─── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _client():
        return get_qdrant_client()


register_backend(_QdrantBackend())
