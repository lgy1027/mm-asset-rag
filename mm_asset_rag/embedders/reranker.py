"""Two-stage reranker for hybrid retrieval.

bge-m3's model card recommends "hybrid retrieval + re-ranking": pull a
candidate pool with dense + BM25, then score each candidate against the
query with a cross-encoder and return the top-k. Cross-encoders see the
query and document jointly (not as independent embeddings), so they catch
"high-score false positives" that threshold filtering cannot — e.g. a
query about "强化学习 PPO" matching an SSD paper that happens to share
tokens.

Two provider backends, selected by ``Settings.reranker_provider``:

- ``local`` (default) — runs ``sentence_transformers.CrossEncoder``
  in-process. Same dep family as the bge-m3 embedder; no network. Needs the
  model downloaded (~2GB first run).
- ``siliconflow`` / ``dashscope`` — call a hosted rerank API. The two providers
  speak *different* wire shapes: ``siliconflow`` uses the flat Cohere form
  (``{model, query, documents, top_n}`` → ``results[].{index,
  relevance_score}``); ``dashscope`` (百炼) uses the DashScope-native *nested*
  form (``{model, input:{query, documents}, parameters:{top_n}}`` →
  ``output.results[].{index, relevance_score}``). Same row shape, different
  wrapper — dispatched per provider in ``HttpRerankApiReranker``. No local model;
  latency is a single round-trip, predictable for interactive search.

Both providers feed raw scores into the same normalise / blend / sort
pipeline in :meth:`Reranker.rerank` — the only per-provider method is
:meth:`_score_text_pairs`, which returns ``len(documents)`` floats aligned
to the input order. Image hits are never re-scored (CLIP is already a
relevance signal).

Degradation: when the provider is unavailable (``local`` dep missing, or an
HTTP provider's API call fails), :func:`get_default_reranker` returns
``None`` / :meth:`rerank` degrades to returning the pre-rerank merged
hits — identical to single-stage behaviour. The stickiness policy is
provider-declared via :attr:`Reranker._sticky_ttl`: ``None`` = hard sticky
(local — a corrupted HF cache / missing dep won't self-heal, needs a restart);
a number of seconds = soft sticky (HTTP — auto-recovers after the TTL so a
transient cloud outage doesn't disable reranking for the whole process life).
A programming bug (``TypeError`` / ``ValueError`` / …) is *not* a provider
failure — it propagates out of :meth:`rerank` rather than being silently
swallowed into a sticky-disable (see :class:`RerankerError`).
"""

from __future__ import annotations

import logging
import time
from threading import Lock

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
_UNAVAILABLE = False  # set True after first failed load so we don't retry
# 0.0 = hard sticky (local: never auto-recover, needs reset_reranker / restart).
# >0.0 = soft-sticky expiry in monotonic seconds (HTTP: auto-recover after TTL).
_UNAVAILABLE_UNTIL: float = 0.0
# Injectable monotonic clock so the soft-sticky TTL is testable without real
# sleeps. ``_now()`` returns seconds; tests monkeypatch this to a controllable
# counter.
_now = time.monotonic


class RerankerError(RuntimeError):
    """A reranker *provider* failed to score (network / API / model / bad body).

    :meth:`Reranker.rerank` catches only this — a programming bug
    (``TypeError`` / ``AttributeError`` / ``ValueError`` from our own blending
    or a misconfigured defaults table) is *not* a ``RerankerError`` and
    propagates, so it surfaces in dev instead of being silently swallowed into
    a sticky-disable. ``_score_text_pairs`` implementations wrap their
    provider interactions in ``try/except`` and re-raise this.
    """


class Reranker:
    """Two-stage reranker; provider-agnostic blend / sort pipeline.

    Construction is cheap (stores config only). The heavy provider resource
    (local CrossEncoder model, or HTTP endpoint) is loaded/called lazily on
    the first :meth:`rerank` via :meth:`_score_text_pairs`.

    Subclasses extending the provider interface override :meth:`_load` and
    :meth:`_score_text_pairs` *as a pair* (the base ``_score_text_pairs`` calls
    ``self._load()``; an HTTP subclass overrides both so the base ``_load`` is
    never reached on it), and declare :attr:`_sticky_ttl` to pick their
    stickiness policy on failure.
    """

    #: Stickiness after a provider failure. ``None`` = hard sticky (local: a
    #: corrupted HF cache / missing dep won't self-heal — needs reset_reranker
    #: or a process restart). A number of seconds = soft sticky that
    #: auto-recovers after the TTL (HTTP: a transient cloud outage shouldn't
    #: disable reranking until restart). New providers set this in one line.
    _sticky_ttl: float | None = None

    def __init__(self, *, model: str | None = None) -> None:
        s = get_settings()
        # Local provider's HF model id. HTTP providers override ``__init__`` so
        # this HF id is never set on them (they resolve their API model via
        # ``_config`` instead) — keeping it here would be dead state that reads
        # as "BAAI/bge-reranker-v2-m3" even on a dashscope instance.
        self.model = model or s.reranker_model
        self._model = None  # local: lazy CrossEncoder; HTTP: unused

    # ── provider interface ──────────────────────────────────────────────
    @staticmethod
    def _dep_available() -> bool:
        """True iff the local provider's dependency is importable.

        Only meaningful for the local backend; HTTP providers override this to
        ``True`` (their "dependency" is ``requests``, a core dep). Probed once
        at :func:`get_default_reranker` so a missing dep short-circuits before
        ``hybrid_search`` commits to the two-stage path.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _score_text_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, aligned to input order.

        Local provider: ``CrossEncoder.predict``. HTTP providers: one
        ``POST /rerank`` whose response is reordered by ``index`` back to the
        input order (the API returns results sorted by relevance, not by
        input position). Provider failures raise :class:`RerankerError`;
        programming bugs propagate.
        """
        if not documents:
            return []
        try:
            model = self._load()
            scores = model.predict([(query, doc) for doc in documents], show_progress_bar=False)
        except RerankerError:
            raise
        except Exception as exc:  # corrupted cache, OOM, revoked weights, …
            raise RerankerError(f"local cross-encoder failed to score: {exc}") from exc
        try:
            return [float(v) for v in scores]
        except TypeError:  # pragma: no cover — scores already a scalar
            return [float(scores)]

    def _load(self):
        """Lazy-load the local CrossEncoder. Overridden by HTTP providers."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model)
        return self._model

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
            )

        ranked: list[SearchHit] = []
        for (h, ce_raw, hy_raw), ce_n, hy_n in zip(text_scored, ce_norm, text_hy):
            ranked.append(_build(h, blend * ce_n + (1.0 - blend) * hy_n, ce_raw, hy_raw))
        for (h, clip_raw, hy_raw), clip_n, hy_n in zip(image_scored, clip_norm, image_hy):
            ranked.append(_build(h, blend * clip_n + (1.0 - blend) * hy_n, clip_raw, hy_raw))

        ranked.sort(key=lambda h: h.score, reverse=True)
        return ranked[:top_k]


# ─── HTTP rerank API provider ──────────────────────────────────────────


# Per-provider defaults. Each entry: (api_base, model, form) where ``form`` is
# the request/response shape the provider speaks — "flat" (Cohere-form) or
# "nested" (DashScope-native). Resolved when the matching settings field is
# None so users only set ``RERANKER_PROVIDER`` + the key.
#
# SiliconFlow: flat Cohere form — one public endpoint.
#
# 百炼 (dashscope): the *flat* OpenAI-compatible endpoint
# (``/compatible-api/v1/reranks``) lives behind a per-user workspaceId maas
# subdomain and is awkward. The **DashScope-native** endpoint at the
# universal ``dashscope.aliyuncs.com`` host works out-of-the-box with just an
# API key (verified: ``qwen3-rerank`` 200, no workspace setup). It speaks a
# field-nested request (``input.query`` / ``input.documents`` /
# ``parameters.top_n``) and wraps results under ``output.results`` — same
# ``{index, relevance_score}`` row shape as the flat form, just nested. So we
# default 百炼 to the native endpoint + nested form rather than the
# compatible flat one.
#
# Adding a new flat/nested HTTP provider = one entry here + one value on the
# ``reranker_provider`` Literal in settings.py. A truly novel wire shape is
# the subclass extension point (override ``_score_text_pairs``).
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

    #: Soft sticky — a transient cloud outage (5xx / timeout / network) auto-
    #: recovers after 60s instead of disabling reranking until process restart.
    _sticky_ttl = 60.0

    def __init__(self, *, model: str | None = None) -> None:
        # ``model`` is the local HF id, unused on the HTTP path (the API model
        # is resolved per-call via ``_config`` from ``reranker_api_model``).
        # Setting ``self.model`` to the HF default here would read as
        # "BAAI/bge-reranker-v2-m3" even on a dashscope instance — misleading.
        self.model = None
        self._model = None  # stateless HTTP provider has no loaded model

    @staticmethod
    def _dep_available() -> bool:
        """``requests`` importable **and** the provider is configured.

        Unlike the local backend (whose only failure is a missing import), an
        HTTP provider that lacks a key / base is *misconfigured*, not
        *temporarily unavailable* — every ``rerank`` would 401 or 404 and
        degrade. Surfacing that here as "unavailable" makes
        :func:`get_default_reranker` return ``None`` so ``hybrid_search``
        skips the two-stage path cleanly, instead of silently retrying a
        failing call every query (the exact trap the local backend's dep
        probe exists to avoid).
        """
        try:
            import requests  # noqa: F401
        except ImportError:  # pragma: no cover
            return False
        s = get_settings()
        default_base, default_model, _ = _provider_defaults(s.reranker_provider)
        base = s.reranker_api_base or default_base
        model = s.reranker_api_model or default_model
        key = s.reranker_api_key or s.openai_api_key
        return bool(base and key and model)

    def _config(self) -> tuple[str, str, str, str, float]:
        """Resolve (api_base, model, form, api_key, timeout) with defaults."""
        s = get_settings()
        provider = s.reranker_provider
        default_base, default_model, default_form = _provider_defaults(provider)
        api_base = s.reranker_api_base or default_base
        model = s.reranker_api_model or default_model
        form = default_form  # form is provider-fixed, not user-tunable
        api_key = s.reranker_api_key or s.openai_api_key or ""
        timeout = s.reranker_api_timeout
        return api_base, model, form, api_key, timeout

    def _load(self):
        # No persistent model to load; HTTP providers are stateless. Kept as a
        # no-op so the base class's _load contract (called nowhere here) holds.
        return None

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
        # pattern as auto_meta / contextual / image_caption. Deferred import
        # to avoid an import cycle through ``answer``.
        try:
            from ..answer import _warn_insecure_base_url

            _warn_insecure_base_url(api_base)
        except Exception:  # pragma: no cover — never let the warning block rerank
            pass
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
    """Return the process-wide :class:`Reranker`, or ``None`` if unavailable.

    Returns ``None`` (so the caller skips reranking) when:
    - ``Settings.reranker_enabled`` is False, or
    - the provider's dependency is missing, or
    - the provider fails to construct (missing config).

    Stickiness on a runtime provider failure is provider-declared
    (:attr:`Reranker._sticky_ttl`): ``None`` (local) = hard sticky — only
    :func:`reset_reranker` or a restart re-enables; a TTL (HTTP) = soft sticky
    that auto-recovers after the TTL so a transient cloud outage doesn't
    disable reranking for the whole process life.
    """
    global _INSTANCE, _UNAVAILABLE, _UNAVAILABLE_UNTIL
    s = get_settings()
    if not s.reranker_enabled:
        return None
    if _UNAVAILABLE:
        # Soft-sticky (HTTP, ``_UNAVAILABLE_UNTIL > 0``): auto-recover after the
        # TTL. Hard-sticky (local, ``_UNAVAILABLE_UNTIL == 0``): never recover.
        # Re-check inside the lock — between our outside read and acquiring the
        # lock, another thread may have just set a fresh TTL via
        # ``_mark_unavailable``; clearing unconditionally would clobber it and
        # immediately re-probe the just-failed provider.
        with _LOCK:
            if _UNAVAILABLE_UNTIL and _now() >= _UNAVAILABLE_UNTIL:
                _INSTANCE = None
                _UNAVAILABLE = False
                _UNAVAILABLE_UNTIL = 0.0
            else:
                return None
        # fall through and re-probe / construct
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        cls = _provider_class(s.reranker_provider)
        if not cls._dep_available():
            # Common-failure path (local dep not installed, or HTTP provider
            # misconfigured). Hard-sticky: a missing dep / bad config won't
            # self-heal, so we don't re-probe on every search call. Explicit
            # ``_UNAVAILABLE_UNTIL = 0.0`` pins the hard-sticky intent so a
            # future change to the soft-sticky path can't turn this into a
            # self-recovering one.
            _UNAVAILABLE = True
            _UNAVAILABLE_UNTIL = 0.0
            return None
        try:
            _INSTANCE = cls()
        except Exception:
            _UNAVAILABLE = True
            _UNAVAILABLE_UNTIL = 0.0
            return None
    return _INSTANCE


def _provider_class(provider: str) -> type[Reranker]:
    """Map ``Settings.reranker_provider`` to a concrete class."""
    if provider in ("siliconflow", "dashscope"):
        return HttpRerankApiReranker
    if provider != "local":
        # Literal validation at Settings layer normally blocks this, but a
        # caller poking ``settings.reranker_provider`` directly (or a future
        # loosened Literal) would otherwise silently run the local backend
        # while the user believes they are calling a cloud API.
        _LOGGER.warning(
            "unknown reranker_provider %r, falling back to local CrossEncoder",
            provider,
        )
    return Reranker  # "local" + any unknown → local backend


def reset_reranker() -> None:
    """Clear the cached instance + unavailable flag. For tests.

    Also clears the ``get_settings`` lru_cache so a test's ``monkeypatch.setenv``
    of ``RERANKER_*`` vars is seen by the next ``get_settings()`` call — without
    this, the first test to touch settings pins a cached instance and later
    tests that only ``setenv`` (no explicit ``cache_clear``) silently read the
    stale provider / base / key. Centralising it here means every test that
    resets the reranker also gets a fresh settings read.
    """
    global _INSTANCE, _UNAVAILABLE, _UNAVAILABLE_UNTIL
    _INSTANCE = None
    _UNAVAILABLE = False
    _UNAVAILABLE_UNTIL = 0.0
    get_settings.cache_clear()


def _mark_unavailable(*, ttl: float | None = None) -> None:
    """Flag the reranker unavailable for the process.

    ``ttl`` (seconds) → soft sticky that auto-recovers after the TTL via
    :func:`get_default_reranker` (HTTP providers, so a transient cloud outage
    self-heals). ``None`` → hard sticky that only :func:`reset_reranker` clears
    (local provider — a missing dep / corrupted HF cache won't self-heal in a
    process lifetime). Called when the provider loads at construction (probe
    passed) but then fails at ``rerank`` time — e.g. a corrupted local HF
    cache, or an HTTP API 5xx / network error / revoked key / bad body.
    """
    global _INSTANCE, _UNAVAILABLE, _UNAVAILABLE_UNTIL
    with _LOCK:
        _INSTANCE = None
        _UNAVAILABLE = True
        _UNAVAILABLE_UNTIL = (_now() + ttl) if ttl else 0.0
