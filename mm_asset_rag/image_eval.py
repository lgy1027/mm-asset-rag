"""Build repeatable image retrieval qrels from persisted image tags."""

from __future__ import annotations

from collections.abc import Mapping

from .knowledge_models import Chunk


def build_tag_qrels(chunks: list[Chunk], queries: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Map each query id to image documents carrying its required tag."""
    qrels: dict[str, dict[str, int]] = {}
    for query_id, tag in queries.items():
        qrels[query_id] = {
            chunk.document_id: 1
            for chunk in chunks
            if chunk.asset.source_type == "image" and tag in chunk.metadata.get("tags", [])
        }
    return qrels


def build_tag_cases(
    chunks: list[Chunk],
    queries: Mapping[str, str],
    *,
    negative_queries: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a v2 image-evaluation case payload from tags and negative queries."""
    groups: dict[str, list[dict[str, str]]] = {
        "text_to_image": [{"query_id": query_id, "query": tag} for query_id, tag in queries.items()]
    }
    qrels = build_tag_qrels(chunks, queries)
    if negative_queries:
        groups["negative"] = [
            {"query_id": query_id, "query": query} for query_id, query in negative_queries.items()
        ]
        qrels.update({query_id: {} for query_id in negative_queries})
    return {
        "version": "v2",
        "groups": groups,
        "qrels": qrels,
    }
