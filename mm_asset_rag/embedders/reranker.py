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
        side of the (query, document) pair. The original hybrid score is
        preserved in ``metadata["hybrid_score"]``; the returned
        ``SearchHit.score`` is the cross-encoder score. Pure: returns a
        new list, the input ``hits`` are not mutated.

        Image hits (``source_type == "image"``) are **not** re-scored by
        the text cross-encoder — a CLIP cosine is already a
        query-document relevance signal and running it through a
        text-only cross-encoder would only suppress it. Their original
        score is preserved as-is and they compete with the text hits'
        cross-encoder scores in the final unified sort. This keeps
        relevant image hits visible when the reranker is on while still
        letting high-scoring text hits rise to the top.
        """
        if not hits:
            return []
        image_hits = [h for h in hits if h.source_type == "image"]
        text_hits = [h for h in hits if h.source_type != "image"]

        # Preserve image hits' original scores; only text/PDF hits go
        # through the cross-encoder.
        image_ranked = [
            SearchHit(
                route=h.route,
                score=h.score,
                asset_id=h.asset_id,
                title=h.title,
                source_type=h.source_type,
                source_path=h.source_path,
                evidence=h.evidence,
                metadata={**h.metadata, "hybrid_score": h.score},
            )
            for h in image_hits
        ]

        if not text_hits:
            ranked = sorted(image_ranked, key=lambda h: h.score, reverse=True)
            return ranked[:top_k]

        pairs = [(query, h.evidence or "") for h in text_hits]
        model = self._load()
        scores = model.predict(pairs, show_progress_bar=False)
        # ``predict`` returns a numpy array; normalise to float regardless
        # of whether softmax/softmax-less was used.
        try:
            score_list = [float(s) for s in scores]
        except TypeError:  # pragma: no cover — scores already a scalar
            score_list = [float(scores)]

        text_ranked = sorted(
            (
                SearchHit(
                    route=h.route,
                    score=score_list[i],
                    asset_id=h.asset_id,
                    title=h.title,
                    source_type=h.source_type,
                    source_path=h.source_path,
                    evidence=h.evidence,
                    metadata={**h.metadata, "hybrid_score": h.score},
                )
                for i, h in enumerate(text_hits)
            ),
            key=lambda h: h.score,
            reverse=True,
        )
        # Merge the two ranked lists by their respective scores and
        # take the top_k. The scores live on different scales (CLIP
        # cosine vs cross-encoder logit) so the ordering is a
        # best-effort interleaving — image hits with a high CLIP score
        # stay visible while text hits with a high cross-encoder score
        # rise to the top. Future work: plug an image reranker here.
        ranked = sorted([*text_ranked, *image_ranked], key=lambda h: h.score, reverse=True)
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
