"""Hybrid retrieval across Qdrant text + image collections."""

from __future__ import annotations

from pathlib import Path

from .backends.qdrant_backend import (
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .schema import SearchHit


def normalize_scores(hits: list[SearchHit]) -> list[SearchHit]:
    """Return new ``SearchHit`` objects with scores divided by ``max(hits)``.

    Pure function: the input list is not mutated. This avoids surprising
    side effects on the upstream ``qdrant_*_search`` result list, which
    is shared across ``hybrid_search`` invocations.
    """
    if not hits:
        return []
    max_score = max(hit.score for hit in hits) or 1.0
    return [
        SearchHit(
            route=hit.route,
            score=hit.score / max_score,
            asset_id=hit.asset_id,
            title=hit.title,
            source_type=hit.source_type,
            source_path=hit.source_path,
            evidence=hit.evidence,
            metadata=dict(hit.metadata),
        )
        for hit in hits
    ]


def _merge_routes(existing: list[str] | None, new_route: str) -> list[str]:
    routes = list(existing or [])
    routes.append(new_route)
    return sorted(set(routes))


def merge_hits(groups: list[list[SearchHit]], weights: list[float], top_k: int) -> list[SearchHit]:
    """Combine per-route hits by ``asset_id`` using weighted scores.

    Pure function: returns a new list of ``SearchHit`` instances; the
    input groups are not mutated.
    """
    merged: dict[str, SearchHit] = {}
    for group, weight in zip(groups, weights):
        for hit in normalize_scores(group):
            if hit.score <= 0:
                continue
            key = hit.asset_id
            weighted_score = hit.score * weight
            if key not in merged:
                merged[key] = SearchHit(
                    route=hit.route,
                    score=weighted_score,
                    asset_id=hit.asset_id,
                    title=hit.title,
                    source_type=hit.source_type,
                    source_path=hit.source_path,
                    evidence=hit.evidence,
                    metadata={**hit.metadata, "routes": [hit.route]},
                )
            else:
                current = merged[key]
                merged[key] = SearchHit(
                    route=current.route,
                    score=current.score + weighted_score,
                    asset_id=current.asset_id,
                    title=current.title,
                    source_type=current.source_type,
                    source_path=current.source_path,
                    evidence=(
                        hit.evidence
                        if len(hit.evidence) > len(current.evidence)
                        else current.evidence
                    ),
                    metadata={
                        **current.metadata,
                        "routes": _merge_routes(
                            current.metadata.get("routes"), hit.route
                        ),
                    },
                )
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:top_k]


def hybrid_search(
    query: str,
    image_path: Path | None = None,
    top_k: int = 5,
) -> list[SearchHit]:
    groups: list[list[SearchHit]] = [
        qdrant_text_search(query, top_k=top_k),
        qdrant_text_to_image_search(query, top_k=top_k),
    ]
    weights = [0.55, 0.30]
    if image_path:
        groups.append(qdrant_image_to_image_search(image_path, top_k=top_k))
        weights.append(0.15)
    return merge_hits(groups, weights, top_k=top_k)
