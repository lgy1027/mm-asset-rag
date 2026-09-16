"""Remote two-stage reranking for hybrid retrieval."""

from __future__ import annotations

import contextlib
import logging
import time
from threading import Lock

from .. import provider_security
from ..schema import SearchHit
from ..settings import get_settings

_LOGGER = logging.getLogger(__name__)
# HTTP errors that are *transient* (server-side / network) get a short retry
# before we give up and sticky-disable the reranker for the process. A 401 /
# 403 / 422 (bad auth or bad model) is a *config* error — retried or not it
# will keep failing, so it goes straight to degrade. See ``_score_text_pairs``.
_HTTP_RETRYABLE = (
    "Timeout",
    "ConnectionError",
    "ConnectionResetError",
    "ChunkedEncodingError",
)
_HTTP_RETRY_COUNT = 1  # one retry on a transient error → 2 attempts total
# Seconds to back off between retry attempts on a transient error. A hosted
# rerank is a single batched call, so a short fixed backoff is enough — it
# gives an overloaded server a breath without adding perceptual latency. Tests
# monkeypatch this to 0 to avoid real sleeps.
_HTTP_RETRY_BACKOFF = 0.5

_LOCK = Lock()
_INSTANCE: Reranker | None = None
_UNAVAILABLE = False
_UNAVAILABLE_UNTIL: float = 0.0
_now = time.monotonic


class RerankerError(RuntimeError):
    """The remote reranking provider could not score a request."""


class Reranker:
    """Provider-independent score blending and ordering."""

    _sticky_ttl: float | None = None

    def _score_text_pairs(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        try:
            scores = self._load().predict(
                [(query, doc) for doc in documents], show_progress_bar=False
            )
            return [float(score) for score in scores]
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerError(f"reranker scorer failed: {exc}") from exc

    def _load(self):
        raise NotImplementedError

    # ── shared pipeline ──────────────────────────────────────────────────
    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        """Score ``hits`` against ``query`` and return the top-k.

        Uses ``hit.evidence`` (the chunk text payload) as the document side of
        the (query, document) pair. The original hybrid RRF score is preserved
        in ``metadata["hybrid_score"]`` and, by default, **blended** with the
        reranker score rather than discarded — a cross-encoder reading a long
        chunk can over-trust token frequency / structural cues (e.g. a
        well-structured abstract of an unrelated paper) and outrank the true
        match. The blend ``blend * norm(ce) + (1-blend) * norm(hybrid)`` keeps
        the reranker in charge while the whole-document dense + BM25 signal
        anchors it. Controlled by ``Settings.reranker_hybrid_blend``
        (1.0 reproduces the old pure-reranker behaviour). Pure: returns new
        ``SearchHit`` instances; inputs are not mutated.

        Image hits (``source_type == "image"``) are **not** re-scored by the
        text cross-encoder — a CLIP cosine is already a query-document
        relevance signal and running it through a text-only cross-encoder
        would only suppress it. Their CLIP score is min-max normalised onto the
        same [0,1] scale as the text blend so image and text hits compete on a
        common axis.

        Only :class:`RerankerError` (a provider failure) is caught and degraded;
        a programming bug propagates so it is visible in dev rather than
        silently sticky-disabling the reranker.
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
            documents = [h.evidence or "" for h in text_hits]
            try:
                ce_scores = self._score_text_pairs(query, documents)
            except RerankerError:
                # Provider failed (corrupted local cache, API 5xx, network,
                # revoked key, bad response body, …). Mark the process-wide
                # instance unavailable using the provider's stickiness policy
                # (HTTP: soft TTL; local: hard) so subsequent searches skip the
                # two-stage path instead of re-attempting the failing call
                # every query, then degrade: return the pre-rerank merged hits
                # in hybrid-score order.
                _mark_unavailable(ttl=self._sticky_ttl)
                return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
            # Defensively align lengths — an HTTP provider that drops a
            # candidate should not crash the blend below.
            if len(ce_scores) != len(text_hits):  # pragma: no cover
                _mark_unavailable(ttl=self._sticky_ttl)
                return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
            text_scored = [(h, ce_scores[i], h.score) for i, h in enumerate(text_hits)]

        # Min-max normalise each signal to [0,1] across all candidates so the
        # blend is scale-free (CLIP cosine ~0.2-0.4, cross-encoder logits
        # unbounded, RRF ~0.016, hosted rerank score 0-1 all become
        # comparable). When every candidate shares a signal's value
        # (degenerate pool), fall back to full contribution — returning 0.0
        # would silently zero out a lone image hit's CLIP score and bury it.
        def _norm(values: list[float]) -> list[float]:
            if not values:
                return []
            lo, hi = min(values), max(values)
            if hi <= lo:
                return [1.0] * len(values)
            span = hi - lo
            return [(v - lo) / span for v in values]

        text_ce = [ce for _, ce, _ in text_scored]
        image_clip = [cl for _, cl, _ in image_scored]
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
                images=list(h.images),
                cache_id=h.cache_id,
            )

        ranked: list[SearchHit] = []
        for (h, ce_raw, hy_raw), ce_n, hy_n in zip(text_scored, ce_norm, text_hy):
            ranked.append(_build(h, blend * ce_n + (1.0 - blend) * hy_n, ce_raw, hy_raw))
        for (h, clip_raw, hy_raw), clip_n, hy_n in zip(image_scored, clip_norm, image_hy):
            ranked.append(_build(h, blend * clip_n + (1.0 - blend) * hy_n, clip_raw, hy_raw))

        ranked.sort(key=lambda h: h.score, reverse=True)
        return ranked[:top_k]


# ─── HTTP rerank API provider ──────────────────────────────────────────


# Provider defaults are ``(api_base, model, form)``. New flat/nested providers
# need one entry and a Settings Literal; a new wire shape needs an adapter.
# DashScope uses its API-key-only nested endpoint by default.
_HTTP_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "siliconflow": ("https://api.siliconflow.cn/v1/rerank", "BAAI/bge-reranker-v2-m3", "flat"),
    "dashscope": (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        "qwen3-rerank",
        "nested",
    ),
}


def _provider_defaults(provider: str) -> tuple[str, str, str]:
    """(api_base, model, form) defaults for an HTTP provider.

    Single source of truth for :meth:`HttpRerankApiReranker._dep_available`
    and :meth:`HttpRerankApiReranker._config` so the two never drift apart.
    Returns ``("", "", "flat")`` for an unknown provider.
    """
    return _HTTP_PROVIDER_DEFAULTS.get(provider, ("", "", "flat"))


class HttpRerankApiReranker(Reranker):
    """Hosted rerank API provider.

    Two wire shapes, selected per provider:

    - **flat** (SiliconFlow, Cohere-form): ``POST {base}`` with
      ``{model, query, documents, top_n}`` → ``{results: [{index,
      relevance_score}, ...]}``.
    - **nested** (百炼 DashScope-native): ``POST {base}`` with
      ``{model, input:{query, documents}, parameters:{top_n}}`` →
      ``{output:{results: [{index, relevance_score}, ...]}}``.

    Both row shapes use ``{index, relevance_score}``; only the wrapper and the
    request body differ. ``results`` is sorted by relevance, not by input
    position, so we reorder by ``index`` back to input order before handing
    scores to the blend pipeline. A missing candidate (server dropped it) is
    scored 0.0 — min-max normalisation then pushes it to the bottom rather
    than crashing.

    .. note:: 百炼's ``gte-rerank-v2`` / ``qwen3-vl-rerank`` also speak the
        nested form at the same endpoint, so they work too — just set
        ``RERANKER_API_MODEL``. The flat OpenAI-compatible endpoint (needs a
        per-user workspaceId) is intentionally not used.
    """

    _sticky_ttl = 60.0

    @staticmethod
    def is_configured() -> bool:
        """Return whether the enabled remote provider has usable credentials."""
        s = get_settings()
        default_base, default_model, _ = _provider_defaults(s.reranker_provider)
        base = s.reranker_api_base or default_base
        model = s.reranker_api_model or default_model
        key = s.reranker_api_key or s.model_api_key
        return bool(base and key and model)

    def _config(self) -> tuple[str, str, str, str, float]:
        """Resolve (api_base, model, form, api_key, timeout) with defaults."""
        s = get_settings()
        provider = s.reranker_provider
        default_base, default_model, default_form = _provider_defaults(provider)
        api_base = s.reranker_api_base or default_base
        model = s.reranker_api_model or default_model
        form = default_form  # form is provider-fixed, not user-tunable
        api_key = s.reranker_api_key or s.model_api_key or ""
        timeout = s.reranker_api_timeout
        return api_base, model, form, api_key, timeout

    def _score_text_pairs(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        api_base, model, form, api_key, timeout = self._config()
        if not api_base or not model:
            raise RerankerError(
                "rerank HTTP provider misconfigured: "
                f"provider={get_settings().reranker_provider} "
                f"api_base={api_base!r} model={model!r}"
            )
        if form not in ("flat", "nested"):
            # Programming error (a bad ``_HTTP_PROVIDER_DEFAULTS`` entry), not a
            # runtime provider failure — let it propagate loudly rather than be
            # swallowed by ``rerank``'s RerankerError handler as a sticky-disable.
            raise ValueError(f"unknown rerank wire form {form!r}")
        # Reuse the project's insecure-URL guard so a plain-HTTP non-loopback
        # ``reranker_api_base`` warns once about Bearer key in cleartext — same
        # pattern as auto_meta / contextual / image_caption.
        with contextlib.suppress(Exception):
            provider_security.warn_insecure_base_url(api_base)
        import requests

        # Provider-specific wire shape. Both end at ``results[].{index,
        # relevance_score}``; only the request body and the results' wrapper
        # differ (flat: top-level ``results``; nested: ``output.results``).
        if form == "nested":
            body = {
                "model": model,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": len(documents), "return_documents": False},
            }
        else:  # "flat" — Cohere form (SiliconFlow)
            body = {
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Transient errors (timeout / connection reset / 5xx / bad body) get
        # one retry with a short backoff — a single blip on a hosted API should
        # not sticky-disable reranking for the whole process. A 4xx (auth / bad
        # model) is a config error and is not retried; it goes straight to the
        # caller's degrade path. ``resp.json()`` is inside the try so a 200 with
        # a non-JSON body degrades as a RerankerError instead of leaking a
        # JSONDecodeError that the broad caller would hard-sticky on.
        attempts = max(1, _HTTP_RETRY_COUNT + 1)
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(api_base, headers=headers, json=body, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                # ``results`` lives at top level (flat) or under ``output``
                # (nested DashScope). ``output`` may carry extra usage/request_id
                # we don't need.
                results = data.get("results") or (data.get("output") or {}).get("results") or []
                # Reorder by the server's ``index`` back to input position. A
                # dropped candidate (no result row) scores 0.0 — min-max
                # normalisation buries it instead of crashing the blend. Use
                # .get so a malformed row (missing index/score) is skipped,
                # not a crash.
                by_index = {
                    r.get("index"): float(r.get("relevance_score", 0.0))
                    for r in results
                    if r.get("index") is not None
                }
                return [by_index.get(i, 0.0) for i in range(len(documents))]
            except RerankerError:
                raise
            except Exception as exc:  # requests.HTTPError / Timeout / JSONDecodeError
                retryable = type(exc).__name__ in _HTTP_RETRYABLE or _is_http_5xx(exc)
                if retryable and attempt < attempts:
                    _LOGGER.warning(
                        "rerank API transient error (attempt %d/%d), retrying: %s %s",
                        attempt,
                        attempts,
                        type(exc).__name__,
                        _exc_status(exc),
                    )
                    if _HTTP_RETRY_BACKOFF > 0:
                        time.sleep(_HTTP_RETRY_BACKOFF)
                    continue
                # Final failure (non-retryable, or retries exhausted): log so a
                # silent degrade is traceable in production, then surface a
                # RerankerError which the caller degrades on (HTTP: soft TTL).
                _LOGGER.warning(
                    "rerank API failed, degrading two-stage rerank for this "
                    "process (provider=%s url=%s): %s %s",
                    get_settings().reranker_provider,
                    api_base,
                    type(exc).__name__,
                    _exc_status(exc),
                )
                raise RerankerError(
                    f"rerank API failed: {type(exc).__name__} {_exc_status(exc)}"
                ) from exc
        raise RerankerError("rerank API exhausted retries")  # pragma: no cover


def _is_http_5xx(exc) -> bool:
    """True if a requests HTTPError carries a 5xx response status."""
    resp = getattr(exc, "response", None)
    return bool(resp and 500 <= getattr(resp, "status_code", 0) < 600)


def _exc_status(exc) -> str:
    """Compact status hint for log lines: HTTP status, else the raw message."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", "?")
        return f"(HTTP {code})"
    return f"({exc})"


def get_default_reranker() -> Reranker | None:
    """Return the enabled remote reranker, if its configuration is valid."""
    global _INSTANCE, _UNAVAILABLE, _UNAVAILABLE_UNTIL
    s = get_settings()
    if not s.reranker_enabled:
        return None
    if _UNAVAILABLE:
        with _LOCK:
            if _UNAVAILABLE_UNTIL and _now() >= _UNAVAILABLE_UNTIL:
                _INSTANCE = None
                _UNAVAILABLE = False
                _UNAVAILABLE_UNTIL = 0.0
            else:
                return None
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        if not HttpRerankApiReranker.is_configured():
            _UNAVAILABLE = True
            _UNAVAILABLE_UNTIL = 0.0
            return None
        try:
            _INSTANCE = HttpRerankApiReranker()
        except Exception:
            _UNAVAILABLE = True
            _UNAVAILABLE_UNTIL = 0.0
            return None
    return _INSTANCE


def reset_reranker() -> None:
    """Clear the process cache. Intended for tests."""
    global _INSTANCE, _UNAVAILABLE, _UNAVAILABLE_UNTIL
    _INSTANCE = None
    _UNAVAILABLE = False
    _UNAVAILABLE_UNTIL = 0.0
    get_settings.cache_clear()


def _mark_unavailable(*, ttl: float | None = None) -> None:
    """Flag the reranker unavailable for the process.

    ``ttl`` is the monotonic expiry time for a temporary provider failure.
    """
    global _INSTANCE, _UNAVAILABLE, _UNAVAILABLE_UNTIL
    with _LOCK:
        _INSTANCE = None
        _UNAVAILABLE = True
        _UNAVAILABLE_UNTIL = (_now() + ttl) if ttl else 0.0
