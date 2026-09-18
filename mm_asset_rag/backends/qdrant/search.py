"""Qdrant text, text-to-image, and image-to-image query adapter."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from ...core.protocols import SearchFilter
from ...core.schema import SearchHit
from ...core.settings import get_settings
from ...embedders import (
    CnClipImageUnavailable,
    ImageEmbeddingUnavailable,
    get_default_image_embedder,
    get_default_text_embedder,
)
from ...query.retrieval import RRF_K
from .client import get_qdrant_client
from .collections import (
    DENSE_VECTOR_NAME,
    EMBED_COLBERT_VECTOR_NAME,
    EMBED_SPARSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    image_collection,
    text_collection,
)
from .indexing import (
    _embed_bm25,
    _embedder_colbert_capability,
    _embedder_sparse_capability,
    _load_bm25_zh_idf,
)

HYBRID_PREFETCH_LIMIT = get_settings().qdrant_hybrid_prefetch_limit


def _native_policy_filter(search_filter: SearchFilter | None) -> models.Filter:
    """Translate document policy predicates into one Qdrant filter."""
    policy = search_filter or SearchFilter()
    must = [
        models.FieldCondition(key="collection", match=models.MatchValue(value=policy.collection))
    ]
    must.extend(
        models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
        for key, value in policy.metadata.items()
    )
    public = models.Filter(
        must=[
            models.FieldCondition(key="allowed_principals", values_count=models.ValuesCount(lte=0))
        ]
    )
    should: list[models.Condition] = [public]
    if policy.principal:
        should.append(
            models.FieldCondition(
                key="allowed_principals", match=models.MatchAny(any=[policy.principal])
            )
        )
    return models.Filter(must=must, should=should)


def _merge_filters(
    policy_filter: models.Filter, extra: models.Filter | None = None
) -> models.Filter:
    """Combine policy predicates with a route-specific source predicate."""
    if extra is None:
        return policy_filter
    return models.Filter(
        must=[*(policy_filter.must or []), *(extra.must or [])],
        should=policy_filter.should,
        must_not=extra.must_not,
    )


def _embed_bm25_zh_query(query: str) -> models.SparseVector | None:
    """Encode ``query`` as a Chinese BM25 sparse vector, or ``None`` if disabled / no IDF."""
    settings = get_settings()
    if not settings.bm25_zh_enabled:
        return None
    idf = _load_bm25_zh_idf()
    if not idf:
        return None
    from . import bm25_zh as _bm25_zh_mod

    tokens = _bm25_zh_mod.tokenize_zh(query)
    return _bm25_zh_mod.bm25_zh_encode_query(tokens, idf)


# ─── Embedder sparse / ColBERT capability probes ──────────────────────────
# The active text embedder may optionally expose ``embed_text_sparse``
# and ``embed_text_colbert`` (only optional capable embedders
# with bge-m3 does). We probe with ``getattr`` so the OpenAI-compatible
# ``TextEmbedder`` — which does not implement these — returns ``None``
# and the collection schema stays dense + bm25 + bm25_zh (zero schema
# change, no reindex required). Settings flags
# ``embedding_sparse_enabled`` / ``embedding_colbert_enabled`` can
# force-enable / force-disable; ``auto`` (default) follows the probe.


def _hybrid_text_query(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    sparse_vector_en: models.SparseVector,
    sparse_vector_zh: models.SparseVector | None,
    top_k: int,
    text_filter: models.Filter | None = None,
    embed_sparse_vector: dict | None = None,
    embed_colbert_vector: list[list[float]] | None = None,
) -> list:
    """Issue a single hybrid query (dense + BM25(en) + BM25(zh) prefetched, fused via RRF).

    Qdrant ranks each prefetch independently, then RRF combines the
    ranked lists. Per-channel bias is applied via
    ``models.RrfQuery(rrf=models.Rrf(weights=[...]))`` — the weights
    array is positional, one entry per prefetch. The default
    1.0/1.0/1.0 matches the previous uniform-fusion behaviour; raise
    ``Settings.rrf_weight_bm25_zh`` to 1.5 to give Chinese-BM25 more
    weight in the fused ranking.

    When ``bm25_zh_enabled`` is on (and the caller passed a non-empty
    sparse vector), the Chinese channel is included as a third
    prefetch and the weight list grows to match.

    When ``embed_sparse_vector`` / ``embed_colbert_vector`` are
    provided (the embedder supports them — bge-m3), extra prefetches
    for the ``embed_sparse`` sparse field and the ``embed_colbert``
    multi-vector field are appended, and the RRF weights list grows
    to keep the positional mapping intact. The OpenAI-compatible
    embedder passes ``None`` for both so the prefetch list and
    schema are unchanged from the pre-capability behaviour.

    When ``text_filter`` is provided it is applied to every prefetch
    channel — Qdrant's ``query_filter`` parameter is ignored by the
    RRF-fused query path on qdrant-client 1.18, so the filter has to
    live on each ``Prefetch`` to take effect. Used to keep
    image-source placeholders out of text→text recall without
    dropping them from the collection.

    When all weights are 1.0 (the default), we fall back to
    ``models.FusionQuery(fusion=models.Fusion.RRF)`` — equivalent to
    uniform RRF — so the Qdrant server's RRF defaults (k=60) apply
    without the explicit ``RrfQuery`` wrapper.
    """
    settings = get_settings()
    prefetches = [
        models.Prefetch(
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=HYBRID_PREFETCH_LIMIT,
            filter=text_filter,
        ),
        models.Prefetch(
            query=sparse_vector_en,
            using=SPARSE_VECTOR_NAME,
            limit=HYBRID_PREFETCH_LIMIT,
            filter=text_filter,
        ),
    ]
    include_zh = (
        settings.bm25_zh_enabled
        and sparse_vector_zh is not None
        and len(sparse_vector_zh.indices) > 0
    )
    if include_zh:
        prefetches.append(
            models.Prefetch(
                query=sparse_vector_zh,
                using=settings.bm25_zh_vector_name,
                limit=HYBRID_PREFETCH_LIMIT,
                filter=text_filter,
            )
        )
    include_embed_sparse = embed_sparse_vector is not None and bool(
        embed_sparse_vector.get("indices")
    )
    if include_embed_sparse:
        prefetches.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=embed_sparse_vector["indices"],  # type: ignore[index]
                    values=embed_sparse_vector["values"],  # type: ignore[index]
                ),
                using=EMBED_SPARSE_VECTOR_NAME,
                limit=HYBRID_PREFETCH_LIMIT,
                filter=text_filter,
            )
        )
    include_embed_colbert = embed_colbert_vector is not None and len(embed_colbert_vector) > 0
    if include_embed_colbert:
        prefetches.append(
            models.Prefetch(
                query=embed_colbert_vector,
                using=EMBED_COLBERT_VECTOR_NAME,
                limit=HYBRID_PREFETCH_LIMIT,
                filter=text_filter,
            )
        )
    weights = [
        settings.rrf_weight_dense,
        settings.rrf_weight_bm25,
    ]
    if include_zh:
        weights.append(settings.rrf_weight_bm25_zh)
    if include_embed_sparse:
        weights.append(1.0)
    if include_embed_colbert:
        weights.append(1.0)
    # Use the weighted ``RrfQuery`` only when a channel actually
    # diverges from the default — uniform-weight queries use the
    # simpler ``FusionQuery`` so the server's defaults apply.
    if all(abs(w - 1.0) < 1e-9 for w in weights):
        fusion_query: models.FusionQuery | models.RrfQuery = models.FusionQuery(
            fusion=models.Fusion.RRF
        )
    else:
        fusion_query = models.RrfQuery(
            rrf=models.Rrf(weights=weights, k=RRF_K),
        )
    return client.query_points(
        collection_name=collection_name,
        prefetch=prefetches,
        query=fusion_query,
        limit=top_k,
        with_payload=True,
    ).points


def qdrant_text_search(
    query: str,
    top_k: int = 5,
    *,
    include_image_sources: bool = True,
    search_filter: SearchFilter | None = None,
) -> list[SearchHit]:
    """Hybrid text→text search.

    Image-source chunks participate by default so their labels, OCR and
    captions can be recalled by knowledge-base queries.

    When the query preprocessor is enabled (see
    ``Settings.query_fuzzy`` / ``query_lowercase`` / ``query_expansion``),
    the BM25 channels use the preprocessed form (lowercased, typo-corrected,
    expanded) while the dense channel keeps the original query intact —
    multilingual embeddings are case-aware.
    """
    from ...core.observability import get_tracer

    with get_tracer().start_span(
        "qdrant.text_search",
        attributes={"top_k": top_k},
        input=query,
    ) as span:
        hits = _qdrant_text_search_impl(
            query,
            top_k,
            include_image_sources=include_image_sources,
            search_filter=search_filter,
        )
        span.update(
            output={
                "hits": len(hits),
                "top_scores": [round(h.score, 4) for h in hits[:5]],
                "top_assets": [h.asset_id for h in hits[:5]],
            }
        )
        return hits


def _qdrant_text_search_impl(
    query: str,
    top_k: int = 5,
    *,
    include_image_sources: bool = True,
    search_filter: SearchFilter | None = None,
) -> list[SearchHit]:
    from ...query.query_preprocess import preprocess

    pre = preprocess(query)
    embedder = get_default_text_embedder()
    client = get_qdrant_client()
    dense_query = embedder.embed(pre.dense_query)
    sparse_query = _embed_bm25([pre.bm25_query])[0]
    sparse_query_zh = _embed_bm25_zh_query(pre.bm25_query)

    # Probe optional embedder-native sparse / ColBERT query vectors.
    # The OpenAI-compatible embedder does not implement these methods
    # (``getattr`` returns ``None``), so the default configuration
    # keeps the dense + bm25 + bm25_zh prefetch list unchanged.
    embed_sparse_query: dict | None = None
    embed_colbert_query: list[list[float]] | None = None
    if _embedder_sparse_capability(embedder):
        fn_s = getattr(embedder, "embed_text_sparse", None)
        if fn_s is not None:
            try:
                embed_sparse_query = fn_s(pre.dense_query)
            except Exception:
                embed_sparse_query = None
    if _embedder_colbert_capability(embedder):
        fn_c = getattr(embedder, "embed_text_colbert", None)
        if fn_c is not None:
            try:
                embed_colbert_query = fn_c(pre.dense_query)
            except Exception:
                embed_colbert_query = None

    source_filter: models.Filter | None = None
    if not include_image_sources:
        # Exclude image-source chunks only — they carry placeholder text
        # ("图片标题: …") that pollutes text→text recall. The earlier
        # ``must=[source_type == "pdf"]`` form silently dropped every
        # non-PDF source_type (document, …), so a freshly uploaded docx
        # was indexed but never returned by search. ``must_not`` keeps
        # pdf + document and only filters out image.
        source_filter = models.Filter(
            must_not=[
                models.FieldCondition(key="source_type", match=models.MatchValue(value="image"))
            ]
        )

    text_filter = _merge_filters(_native_policy_filter(search_filter), source_filter)
    # Determine the active collection name (Qdrant active-text env var wins).
    try:
        results = _hybrid_text_query(
            client,
            text_collection(len(dense_query)),
            dense_query,
            sparse_query,
            sparse_query_zh,
            top_k,
            text_filter=text_filter,
            embed_sparse_vector=embed_sparse_query,
            embed_colbert_vector=embed_colbert_query,
        )
    except (ValueError, UnexpectedResponse) as exc:
        # The text collection may not exist yet (e.g. a fresh install, or
        # an instance that has only ingested images). Degrade to an empty
        # result instead of crashing hybrid_search — mirrors the image
        # routes' collection-missing handling.
        if not _is_collection_missing(exc):
            raise
        return []
    return [_point_to_hit("qdrant_text", point) for point in results]


def _filter_by_relevance(results, threshold: float) -> list:
    """Drop Qdrant points whose cosine score is below ``threshold``.

    Used by the image search routes to give them a relevance floor:
    off-topic natural-language queries typically score below the floor
    even for the closest image, so filtering returns an empty list
    instead of ten random Picsum photos. ``threshold=0.0`` keeps every
    result (i.e. the previous behaviour).
    """
    if threshold <= 0.0:
        return list(results)
    return [p for p in results if (p.score or 0.0) >= threshold]


def _is_collection_missing(exc: BaseException) -> bool:
    """True iff ``exc`` is Qdrant's "collection not found" error.

    A freshly installed instance (or one that has only ingested images /
    only ingested text) has no matching collection yet; the search routes
    must treat that as a clean empty result rather than crashing
    ``hybrid_search``. Centralised here so all three routes degrade
    symmetrically.

    Two client modes raise different exception types for a missing
    collection, both of which must be recognised:

    * **local file mode** raises ``ValueError`` with ``"not found"`` in
      the message.
    * **remote server mode** (``QDRANT_URL``) raises
      ``UnexpectedResponse`` (an ``ApiException`` subclass, *not* a
      ``ValueError``) with HTTP 404. Before this handled the remote
      case, a remote instance that had only ingested one modality crashed
      ``hybrid_search`` instead of returning an empty route.
    """
    return (isinstance(exc, ValueError) and "not found" in str(exc)) or (
        isinstance(exc, UnexpectedResponse) and getattr(exc, "status_code", None) == 404
    )


def qdrant_text_to_image_search(
    query: str, top_k: int = 5, *, search_filter: SearchFilter | None = None
) -> list[SearchHit]:
    try:
        provider = get_default_image_embedder()
    except (ImageEmbeddingUnavailable, CnClipImageUnavailable):
        return []
    client = get_qdrant_client()
    try:
        query_vector = provider.embed_text(query)
    except (ImageEmbeddingUnavailable, CnClipImageUnavailable):
        # ``embed_text`` itself can surface availability (e.g. cn_clip cache
        # missing tokenizer files → features is a non-tensor object). Treat
        # that the same as "no image embedder available at all" — the image
        # route silently returns empty so text retrieval still works.
        return []
    # The image collection may not exist yet (e.g. user only ingested
    # PDFs). Treat "no image index" as a clean empty result instead of
    # crashing the hybrid_search call.
    try:
        results = client.query_points(
            collection_name=image_collection(len(query_vector)),
            query=query_vector,
            query_filter=_native_policy_filter(search_filter),
            limit=top_k,
            with_payload=True,
        ).points
    except (ValueError, UnexpectedResponse) as exc:
        if not _is_collection_missing(exc):
            raise
        return []
    threshold = get_settings().image_relevance_threshold
    results = _filter_by_relevance(results, threshold)
    return [_point_to_hit("qdrant_text_to_image", point) for point in results]


def qdrant_image_to_image_search(
    image_path: Path, top_k: int = 5, *, search_filter: SearchFilter | None = None
) -> list[SearchHit]:
    try:
        provider = get_default_image_embedder()
    except (ImageEmbeddingUnavailable, CnClipImageUnavailable):
        return []
    client = get_qdrant_client()
    query_vector = provider.embed_image(image_path)
    # ``embed_image`` returns ``None`` if the file can't be opened / encoded.
    # Treat that as an empty result — symmetric with the text→image route.
    if query_vector is None:
        return []
    # The image collection may not exist yet (e.g. the user only ingested
    # PDFs/documents). Degrade to an empty result instead of raising —
    # symmetric with the text→image route and the text route.
    try:
        results = client.query_points(
            collection_name=image_collection(len(query_vector)),
            query=query_vector,
            query_filter=_native_policy_filter(search_filter),
            limit=top_k,
            with_payload=True,
        ).points
    except (ValueError, UnexpectedResponse) as exc:
        if not _is_collection_missing(exc):
            raise
        return []
    threshold = get_settings().image_relevance_threshold
    results = _filter_by_relevance(results, threshold)
    return [_point_to_hit("qdrant_image_to_image", point) for point in results]


def _point_to_hit(route: str, point) -> SearchHit:
    payload = point.payload or {}
    return _payload_to_hit(route, float(point.score or 0.0), payload)


def _payload_to_hit(route: str, score: float, payload: dict[str, object]) -> SearchHit:
    chunk_metadata = payload.get("chunk_metadata")
    if not isinstance(chunk_metadata, dict):
        chunk_metadata = {}
    images = chunk_metadata.get("images")
    title = str(payload.get("title") or "")
    section = str(chunk_metadata.get("section") or "")
    body = str(payload.get("text", ""))
    # Build a richer evidence snippet: "<title> [<section>] <body[:N]>".
    # The previous 1000-char truncation dropped context the reranker
    # cross-encoder needs to separate relevant hits from high-score
    # false positives. 4000 chars keeps the body around a typical
    # cross-encoder context window while bounding payload size.
    prefix_parts = [p for p in (title, section) if p]
    prefix = " | ".join(prefix_parts)
    evidence = f"{prefix}\n\n{body[:4000]}" if prefix else body[:4000]
    return SearchHit(
        route=route,
        score=score,
        asset_id=str(payload.get("document_id", "")),
        title=title,
        source_type=str(payload.get("source_type", "")),
        source_path=str(payload.get("source_path", "")),
        evidence=evidence,
        # Flatten parser-owned chunk metadata into the public hit
        # metadata so consumers (serializers, playback positioning,
        # evidence policy) see page/start/end/kind at the top level.
        # Payload identity fields win collisions; the nested
        # ``chunk_metadata`` copy stays for explicit readers.
        metadata={
            **chunk_metadata,
            **dict(payload),
            "document_id": str(payload.get("document_id", "")),
        },
        images=list(images) if isinstance(images, list) else [],
        cache_id=str(payload.get("cache_id", "")),
    )


hybrid_text_query = _hybrid_text_query
text_search = qdrant_text_search
text_to_image_search = qdrant_text_to_image_search
image_to_image_search = qdrant_image_to_image_search
point_to_hit = _point_to_hit
payload_to_hit = _payload_to_hit
