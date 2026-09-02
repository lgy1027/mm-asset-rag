"""Qdrant adapter implementing the search and indexing capability ports."""

from __future__ import annotations

from pathlib import Path

from . import client, collections, indexing, search


class QdrantBackend:
    """Capability-oriented adapter over the responsibility-specific modules."""

    name = "qdrant"

    def ensure_collection(self, *, name, dim, sparse=False):
        collections.create_collection(self._client(), name, vector_size=dim, sparse=sparse)

    def drop_collection(self, name):
        self._client().delete_collection(name)

    def upsert(self, *, collection, points, wait=True):
        self._client().upsert(collection_name=collection, points=points, wait=wait)
        return len(points)

    def retrieve_existing_ids(self, *, collection, ids):
        existing = self._client().retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        return {str(point.id) for point in existing}

    def upsert_text(self, *, progress_cb=None, force_recreate=False):
        return indexing.build_text_index(
            progress_cb=progress_cb,
            force_recreate=force_recreate,
        )

    def upsert_image(self, *, progress_cb=None, force_recreate=False):
        return indexing.build_image_index(
            progress_cb=progress_cb,
            force_recreate=force_recreate,
        )

    def search_text(self, *, query, top_k, search_filter=None):
        return search.text_search(query, top_k=top_k, search_filter=search_filter)

    def search_text_to_image(self, *, query, top_k, search_filter=None):
        return search.text_to_image_search(query, top_k=top_k, search_filter=search_filter)

    def search_image(self, *, image_path, top_k, search_filter=None):
        return search.image_to_image_search(
            Path(image_path), top_k=top_k, search_filter=search_filter
        )

    # Compatibility with the pre-capability aggregate VectorBackend port.
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
        return search.hybrid_text_query(
            self._client(),
            collection,
            query_vector,
            sparse_vector,
            None,
            top_k,
        )

    def search_image_to_image(self, *, collection, image_path, top_k):
        return search.image_to_image_search(Path(image_path), top_k=top_k)

    @staticmethod
    def _client():
        return client.get_qdrant_client()


__all__ = ["QdrantBackend", "client", "collections", "indexing", "search"]
