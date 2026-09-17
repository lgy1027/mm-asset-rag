"""Qdrant adapter implementing the search and indexing capability ports."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import models

from ...core.settings import get_settings
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

    def delete_documents(self, document_ids: set[str]) -> dict[str, int]:
        """Delete exact document payloads from every live Qdrant collection."""
        if not document_ids:
            return {"text": 0, "image": 0}
        settings = get_settings()
        qdrant = self._client()
        groups = {
            "text": [
                *(
                    [settings.qdrant_active_text_collection]
                    if settings.qdrant_active_text_collection
                    else []
                ),
                *collections._strict_existing_collections_for(
                    qdrant, collections.TEXT_COLLECTION_BASE
                ),
            ],
            "image": [
                *(
                    [settings.qdrant_active_image_collection]
                    if settings.qdrant_active_image_collection
                    else []
                ),
                *collections._strict_existing_collections_for(
                    qdrant, collections.IMAGE_COLLECTION_BASE
                ),
            ],
        }
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=sorted(document_ids))
                    )
                ]
            )
        )
        counts = {"text": 0, "image": 0}
        for kind, names in groups.items():
            for name in dict.fromkeys(names):
                qdrant.delete(collection_name=name, points_selector=selector)
                counts[kind] += 1
        return counts

    def index_exists(self, kind: str) -> bool:
        """Check live dimension-suffixed collections without leaking client details."""
        try:
            base = (
                collections.TEXT_COLLECTION_BASE
                if kind == "text"
                else collections.IMAGE_COLLECTION_BASE
            )
            return bool(collections._existing_collections_for(self._client(), base))
        except Exception:
            return False

    def invalidate_caches(self) -> None:
        indexing.invalidate_bm25_zh_idf_cache()

    def close(self) -> None:
        self._client().close()

    def search_text(self, *, query, top_k, search_filter=None):
        return search.text_search(query, top_k=top_k, search_filter=search_filter)

    def search_text_to_image(self, *, query, top_k, search_filter=None):
        return search.text_to_image_search(query, top_k=top_k, search_filter=search_filter)

    def search_image(self, *, image_path, top_k, search_filter=None):
        return search.image_to_image_search(
            Path(image_path), top_k=top_k, search_filter=search_filter
        )

    @staticmethod
    def _client():
        return client.get_qdrant_client()


__all__ = ["QdrantBackend", "client", "collections", "indexing", "search"]
