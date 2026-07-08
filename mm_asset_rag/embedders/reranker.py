"""Cross-encoder reranker for two-stage retrieval.

bge-m3's model card recommends "hybrid retrieval + re-ranking": pull a
candidate pool with dense + BM25, then score each candidate against the
query with a cross-encoder and return the top-k. Cross-encoders see the
query and document jointly (not as independent embeddings), so they catch
"high-score false positives" that threshold filtering cannot — e.g. a
query about "强化学习 PPO" matching an SSD paper that happens to share
tokens. On the v6 corpus the global ``min_score`` floor failed to drop
those (8/8 negatives still returned results); this module is the more
precise replacement.

Backed by ``sentence_transformers.CrossEncoder`` — same dependency family
as ``SentenceTransformerTextEmbedder`` (bge-m3). Runs locally, no API.
Lazy-loaded once per process; ``rerank`` is a pure function that returns
new ``SearchHit`` instances (the hybrid score is preserved in
``metadata["hybrid_score"]`` for debugging).

Degradation: when ``sentence_transformers`` is missing or the model fails
to load, :func:`get_default_reranker` returns ``None`` and
:func:`hybrid_search` silently skips reranking — identical to the
pre-reranker behavior.
"""

from __future__ import annotations

from threading import Lock

from ..schema import SearchHit
from ..settings import get_settings

_LOCK = Lock()
_INSTANCE: Reranker | None = None
_UNAVAILABLE = False  # set True after first failed load so we don't retry


class Reranker:
    """Lazy-loaded cross-encoder reranker.

    Construction is cheap (just stores config); the heavy
    ``CrossEncoder`` model is loaded on the first :meth:`rerank` call and
    cached for the process lifetime.
    """

    def __init__(self, *, model: str | None = None) -> None:
        s = get_settings()
        self.model = model or s.reranker_model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        """Score ``hits`` against ``query`` and return the top-k.

        Uses ``hit.evidence`` (the chunk text payload) as the document
        side of the (query, document) pair. The original hybrid RRF
        score is preserved in ``metadata["hybrid_score"]`` and, by
        default, **blended** with the cross-encoder score rather than
        discarded — a cross-encoder reading a long chunk can over-trust
        token frequency / structural cues (e.g. a well-structured
        abstract of an unrelated paper) and outrank the true match. The
        blend ``blend * norm(ce) + (1-blend) * norm(hybrid)`` keeps the
        reranker in charge while the whole-document dense + BM25 signal
        anchors it. Controlled by ``Settings.reranker_hybrid_blend``
        (1.0 reproduces the old pure-reranker behaviour). Pure: returns
        new ``SearchHit`` instances; inputs are not mutated.

        Image hits (``source_type == "image"``) are **not** re-scored by
        the text cross-encoder — a CLIP cosine is already a
        query-document relevance signal and running it through a
        text-only cross-encoder would only suppress it. Their CLIP score
        is min-max normalised onto the same [0,1] scale as the text
        blend so image and text hits compete on a common axis.
        """
        if not hits:
            return []
        s = get_settings()
        blend = max(0.0, min(1.0, s.reranker_hybrid_blend))
        image_hits = [h for h in hits if h.source_type == "image"]
        text_hits = [h for h in hits if h.source_type != "image"]

        # Image hits are not re-scored by the text cross-encoder — a CLIP
        # cosine is already a query-document relevance signal. We read the
        # *original* CLIP score from ``metadata["raw_score"]`` because
        # ``merge_hits`` overwrites ``hit.score`` with the RRF contribution
        # (~1/(60+rank)); without this the blend below would fuse two RRF
        # signals on image hits and discard the CLIP relevance entirely.
        image_scored = [
            (h, h.metadata.get("raw_score", h.score), h.score)
            for h in image_hits  # (hit, clip_score, hybrid_score)
        ]

        text_scored: list[tuple[SearchHit, float, float]] = []
        if text_hits:
            pairs = [(query, h.evidence or "") for h in text_hits]
            model = self._load()
            scores = model.predict(pairs, show_progress_bar=False)
            # ``predict`` returns a numpy array; normalise to float regardless
            # of whether softmax/softmax-less was used.
            try:
                ce_scores = [float(v) for v in scores]
            except TypeError:  # pragma: no cover — scores already a scalar
                ce_scores = [float(scores)]
            text_scored = [(h, ce_scores[i], h.score) for i, h in enumerate(text_hits)]

        # Min-max normalise each signal to [0,1] across all candidates so
        # the blend is scale-free (CLIP cosine ~0.2-0.4, cross-encoder
        # logits unbounded, RRF ~0.016 all become comparable). When every
        # candidate shares a signal's value (degenerate pool), fall back
        # to 0.0 — no information, no contribution.
        def _norm(values: list[float]) -> list[float]:
            if not values:
                return []
            lo, hi = min(values), max(values)
            if hi <= lo:
                # Degenerate pool (single candidate, or all tied): every
                # member is simultaneously the top of its signal family, so
                # it contributes fully — returning 0.0 would silently zero
                # out a lone image hit's CLIP score and bury it.
                return [1.0] * len(values)
            span = hi - lo
            return [(v - lo) / span for v in values]

        # The cross-encoder signal for text hits; the CLIP score stands
        # in for it on image hits (the text cross-encoder never sees them).
        text_ce = [ce for _, ce, _ in text_scored]
        image_clip = [cl for _, cl, _ in image_scored]
        # Hybrid RRF is normalised over *all* candidates so a text hit and
        # an image hit with the same RRF rank land at the same blend level.
        hy_all = [hy for _, _, hy in text_scored] + [hy for _, _, hy in image_scored]
        ce_norm = _norm(text_ce)
        clip_norm = _norm(image_clip)
        hy_norm = _norm(hy_all)
        text_hy = hy_norm[: len(text_scored)]
        image_hy = hy_norm[len(text_scored) :]

        def _build(h: SearchHit, final: float, ce_raw: float, hy_raw: float) -> SearchHit:
            return SearchHit(
                route=h.route,
                score=final,
                asset_id=h.asset_id,
                title=h.title,
                source_type=h.source_type,
                source_path=h.source_path,
                evidence=h.evidence,
                metadata={
                    **h.metadata,
                    "hybrid_score": hy_raw,
                    "rerank_score": ce_raw,
                    "blended": True,
                },
            )

        ranked: list[SearchHit] = []
        # Text hits: blend normalised cross-encoder + hybrid RRF.
        for (h, ce_raw, hy_raw), ce_n, hy_n in zip(text_scored, ce_norm, text_hy):
            ranked.append(_build(h, blend * ce_n + (1.0 - blend) * hy_n, ce_raw, hy_raw))
        # Image hits: blend normalised CLIP + hybrid RRF (no text CE).
        for (h, clip_raw, hy_raw), clip_n, hy_n in zip(image_scored, clip_norm, image_hy):
            ranked.append(_build(h, blend * clip_n + (1.0 - blend) * hy_n, clip_raw, hy_raw))

        ranked.sort(key=lambda h: h.score, reverse=True)
        return ranked[:top_k]


def get_default_reranker() -> Reranker | None:
    """Return the process-wide :class:`Reranker`, or ``None`` if unavailable.

    Returns ``None`` (so the caller skips reranking) when:
    - ``Settings.reranker_enabled`` is False, or
    - ``sentence_transformers`` is missing, or
    - the model fails to load (network / corrupted download).

    A single failed load is sticky: we set ``_UNAVAILABLE`` so subsequent
    calls don't retry within the same process — keeps retrieval latency
    predictable after a transient failure.
    """
    global _INSTANCE, _UNAVAILABLE
    s = get_settings()
    if not s.reranker_enabled or _UNAVAILABLE:
        return None
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        try:
            _INSTANCE = Reranker()
        except Exception:
            _UNAVAILABLE = True
            return None
    return _INSTANCE


def reset_reranker() -> None:
    """Clear the cached instance + unavailable flag. For tests."""
    global _INSTANCE, _UNAVAILABLE
    _INSTANCE = None
    _UNAVAILABLE = False
