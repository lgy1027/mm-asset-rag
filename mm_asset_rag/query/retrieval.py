"""Hybrid retrieval across Qdrant text + image collections.

The three routes (``qdrant_text`` / ``qdrant_text_to_image`` /
``qdrant_image_to_image``) are fused with a **rank-based RRF**
(Reciprocal Rank Fusion) strategy: each route's hits are sorted by
their raw score, assigned a 1-based ``rank``, and merged per
``asset_id`` using ``score_rrf = Σ weight * 1/(RRF_K + rank)``. This
decouples the fusion from the raw score scales of each route (CLIP
cosines live in 0.15-0.40, dense embeddings in 0.0-1.0, BM25 scores
can be unbounded) — the historical ``score / max`` normalisation
coupled the routes' scales and let one hot route silence the others.

``RRF_K`` is backend-neutral retrieval policy. Adapters may import it when
their native fusion needs to match the application-level rank fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..core.schema import SearchHit
from ..core.settings import get_settings
from ..embedders import get_default_reranker
from .evidence_policy import _searchable_text, _terms, lexical_coverage
from .query_intent import (
    IntentWeights,
    classify_intent,
    weights_for_intent,
)

if TYPE_CHECKING:
    from ..core.protocols import SearchBackend


RRF_K = 60


def _merge_routes(existing: list[str] | None, new_route: str) -> list[str]:
    routes = list(existing or [])
    routes.append(new_route)
    return sorted(set(routes))


def _merge_images(existing: list, new: list) -> list:
    """Union of two image-ref lists, de-duplicated by ``path``.

    When several chunks of the same asset surface as separate route hits,
    ``merge_hits`` collapses them into one ``SearchHit`` per asset — the
    user-facing hit should carry *all* figures those chunks referenced,
    not just the first chunk's. Order is preserved (existing first), which
    keeps the highest-scoring chunk's figures on top.
    """
    seen: set[str] = set()
    out: list = []
    for img in [*existing, *new]:
        if not isinstance(img, dict):
            continue
        path = img.get("path")
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(img)
    return out


def _rrf_score(rank: int, weight: float) -> float:
    """Weighted reciprocal-rank contribution for a single hit."""
    return weight / (RRF_K + rank)


def _document_id(hit: SearchHit) -> str | None:
    """Return the required nonempty v2 logical identity, if present."""
    value = hit.metadata.get("document_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _rank_hits(hits: list[SearchHit]) -> list[tuple[SearchHit, int]]:
    """Sort ``hits`` by descending raw ``score`` and assign 1-based ranks.

    Ties are broken by ``asset_id`` so the ranking is deterministic when
    many hits share a score (common for BM25 on short queries).
    """
    ordered = sorted(hits, key=lambda h: (-h.score, _document_id(h) or ""))
    return [(hit, rank) for rank, hit in enumerate(ordered, start=1)]


def merge_hits(
    groups: list[list[SearchHit]],
    weights: list[float],
    top_k: int,
    min_score: float = 0.0,
) -> list[SearchHit]:
    """Combine per-route hits by ``asset_id`` using rank-based RRF.

    Each group is independently ranked by its own raw ``score`` (so a
    route with small scores can still contribute high ranks), then
    every hit contributes ``weight / (RRF_K + rank)`` to its
    ``asset_id``. Hits for the same ``asset_id`` across routes sum
    their RRF contributions, which is exactly the standard
    reciprocal-rank fusion formula extended with per-route weights.

    Pure function: returns a new list of ``SearchHit`` instances; the
    input groups are not mutated.

    ``min_score`` is now a *soft* low-end guard on the final RRF score
    (default ``0.0`` disables it). Because RRF scores are tiny by
    design (a single top hit with ``weight=1`` scores ``1/61 ≈ 0.0164``),
    any positive floor should be on the order of ``0.001``; the hard
    score thresholds of the old normalised-score fusion no longer
    apply. Keep ``0.0`` unless you need to trim tiny-tail noise.
    """
    merged: dict[str, SearchHit] = {}
    for group, weight in zip(groups, weights):
        if weight <= 0:
            continue
        valid_hits = [hit for hit in group if _document_id(hit) is not None]
        for hit, rank in _rank_hits(valid_hits):
            if hit.score <= 0:
                # A zero-score hit carries no signal in its route; skip
                # it so it neither contributes RRF weight nor crowds the
                # per-route rank space for downstream hits.
                continue
            key = _document_id(hit)
            if key is None:  # Defensive: ``valid_hits`` establishes this invariant.
                continue
            contribution = _rrf_score(rank, weight)
            if key not in merged:
                merged[key] = SearchHit(
                    route=hit.route,
                    score=contribution,
                    asset_id=key,
                    title=hit.title,
                    source_type=hit.source_type,
                    source_path=hit.source_path,
                    evidence=hit.evidence,
                    metadata={
                        **hit.metadata,
                        "document_id": key,
                        "routes": [hit.route],
                        # Preserve the route's raw score (CLIP cosine for
                        # image routes, dense/BM25 for text) so the reranker
                        # can still read the original signal after RRF
                        # overwrites ``score`` with the fusion contribution.
                        "raw_score": hit.score,
                    },
                    images=list(hit.images),
                    cache_id=hit.cache_id,
                )
            else:
                current = merged[key]
                representative = (
                    hit if hit.score > float(current.metadata.get("raw_score", 0.0)) else current
                )
                merged[key] = SearchHit(
                    route=current.route,
                    score=current.score + contribution,
                    asset_id=key,
                    title=representative.title,
                    source_type=representative.source_type,
                    source_path=representative.source_path,
                    evidence=representative.evidence,
                    metadata={
                        **representative.metadata,
                        "document_id": key,
                        "routes": _merge_routes(current.metadata.get("routes"), hit.route),
                    },
                    images=_merge_images(current.images, hit.images),
                    cache_id=representative.cache_id,
                )
    sorted_hits = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
    if min_score > 0.0:
        sorted_hits = [hit for hit in sorted_hits if hit.score >= min_score]
    return sorted_hits[:top_k]


def filter_low_evidence_hits(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Suppress results that cannot support enough of the query terms.

    Only long evidence is gated: on short segments (audio windows, slide
    bullets) lexical overlap is too sparse to be a reliable relevance
    signal, and the filter would reject correct hits that merely paraphrase
    the query.
    """
    if not _terms(query):
        return hits
    settings = get_settings()
    threshold = settings.retrieval_min_lexical_coverage
    cap = settings.retrieval_lexical_coverage_max_terms
    min_chars = settings.retrieval_min_evidence_chars
    kept: list[SearchHit] = []
    for hit in hits:
        if len(_searchable_text(hit)) < min_chars:
            kept.append(hit)
            continue
        if lexical_coverage(query, [hit], max_terms=cap) >= threshold:
            kept.append(hit)
    return kept


def hybrid_search(
    query: str,
    image_path: Path | None = None,
    top_k: int = 5,
    min_score: float | None = None,
    *,
    weights_override: IntentWeights | None = None,
    backend: SearchBackend | None = None,
) -> list[SearchHit]:
    """Run a hybrid search across text + (optionally) image routes.

    The three routes are fused with rank-based RRF (see :func:`merge_hits`),
    which removes the cross-route score-scale coupling that the previous
    ``score / max`` normalisation introduced.

    ``min_score`` defaults to ``Settings.min_score`` (env-driven) and is
    forwarded to :func:`merge_hits` as a soft low-end guard on the
    final RRF score. The default ``0.0`` keeps every RRF hit; pass an
    explicit value to override per call.

    Weight selection (keyword-only ``weights_override`` first):

    * Explicit ``weights_override`` wins — callers (tests, scripted
      eval) get a stable weight triple without going through the
      ``classify_intent`` heuristic.
    * Otherwise, when ``Settings.hybrid_intent_routing_enabled`` is on,
      ``classify_intent(query)`` picks an :class:`IntentWeights` from
      :data:`query_intent.DEFAULT_INTENT_WEIGHTS` (with the four
      ``Settings.hybrid_intent_weights_*`` fields as overrides).
    * Otherwise, the historical global
      ``Settings.hybrid_weight_text / text_to_image / image_to_image``
      triple is used — preserves the pre-intent-routing behaviour for
      any deployment that hasn't flipped the master switch.

    When ``Settings.reranker_enabled`` is true, a two-stage pipeline runs:
    ``reranker_top_n`` candidates are fetched from each route and merged,
    then a cross-encoder (``bge-reranker-v2-m3``) re-scores each
    ``(query, evidence)`` pair and the top ``reranker_top_k`` (or
    ``top_k`` if None) are returned. The pre-rerank hybrid score is
    preserved in ``metadata["hybrid_score"]``. Image-source hits are
    *not* re-scored by the text cross-encoder — their original CLIP
    score is preserved (see ``Reranker.rerank``). If the reranker
    fails to load (missing dep / model), the search degrades to the
    single-stage path transparently.
    """
    if backend is None:
        from ..core.registry import get_active_backend

        backend = get_active_backend()

    settings = get_settings()
    reranker = get_default_reranker()
    # Fetch a wider candidate pool when reranking; otherwise top_k end-to-end.
    fetch_k = settings.reranker_top_n if reranker is not None else top_k
    if weights_override is not None:
        chosen = weights_override
    elif settings.hybrid_intent_routing_enabled:
        intent = classify_intent(query)
        chosen = weights_for_intent(intent, settings)
    else:
        chosen = IntentWeights(
            text=settings.hybrid_weight_text,
            text_to_image=settings.hybrid_weight_text_to_image,
            image_to_image=settings.hybrid_weight_image_to_image,
        )
    groups: list[list[SearchHit]] = [
        backend.search_text(query=query, top_k=fetch_k),
        backend.search_text_to_image(query=query, top_k=fetch_k),
    ]
    weights = [chosen.text, chosen.text_to_image]
    # Image-to-image is only consulted when an ``image_path`` is supplied
    # *and* its weight is positive — calling it just to multiply by 0
    # wastes a Qdrant round-trip.
    if image_path and chosen.image_to_image > 0:
        groups.append(backend.search_image(image_path=image_path, top_k=fetch_k))
        weights.append(chosen.image_to_image)
    effective_min = settings.min_score if min_score is None else min_score
    merged = merge_hits(groups, weights, top_k=fetch_k, min_score=effective_min)
    if reranker is not None:
        return_k = settings.reranker_top_k if settings.reranker_top_k is not None else top_k
        merged = reranker.rerank(query, merged, top_k=return_k)
    return filter_low_evidence_hits(query, merged)
