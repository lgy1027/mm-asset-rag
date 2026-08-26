"""LLM-driven query rewriting + multi-query retrieval-augmented generation.

This module layers on top of :func:`mm_asset_rag.retrieval.hybrid_search` —
it is the *entry* point for text / hybrid searches and is responsible
for expanding a single user query into N LLM-generated variants before
firing each one at the indexer. The original
``query_preprocess.preprocess`` (lowercase / fuzzy / synonym expansion)
still runs inside ``hybrid_search`` and stacks with this rewrite — the
two layers are independent.

Two halves, one public surface:

1. :func:`rewrite_query` — single-call LLM expansion. The model is
   asked to produce N variants of the user's query (the original
   included) as JSON ``{"variants": [...]}``. Falls back to the bare
   query on any failure (missing creds, timeout, malformed JSON).
2. :func:`multi_query_search` — fan out :func:`hybrid_search` over
   the variants in a thread pool, then fuse per-route hits with
   :func:`mm_asset_rag.retrieval.merge_hits` (rank-based RRF).
3. :func:`hybrid_search_with_rewrite` — top-level wrapper. ``query_rewrite_enabled=True``
   enables both halves; ``False`` is a transparent pass-through to
   :func:`mm_asset_rag.retrieval.hybrid_search`.

Design notes
------------

- The LLM call is **best-effort**. A rewrite failure (timeout / no
  creds / bad JSON) silently falls back to the original query so a
  search request never 500s because of an LLM glitch. The pattern
  mirrors :func:`mm_asset_rag.answer.fallback_answer` and
  :func:`mm_asset_rag.auto_meta.auto_meta_image`.
- Image routes (``text-to-image`` / ``image-to-image``) skip rewrite
  entirely — variants only help the text side; an LLM-generated
  variant does not improve a CLIP cosine match. The decision lives
  in :func:`mm_asset_rag.service.dispatch_search`, not here.
- ``hybrid_search`` is a blocking Qdrant round-trip; we use
  ``concurrent.futures.ThreadPoolExecutor`` (not asyncio) so the
  rewrite path composes with the rest of the synchronous service
  layer (CLI, ``dispatch_search``, ``IngestService``).
- The rewrite HTTP call is its own private helper
  (:func:`_post_chat_json`) — not a reuse of
  :func:`mm_asset_rag.answer._post_chat` — because rewrite only
  needs the JSON content shape (one-shot, no streaming) and we'd
  otherwise pull answer.py's image-mode retry + ``<think>`` strip
  paths along for no reason.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from .retrieval import hybrid_search, merge_hits
from .schema import SearchHit
from .settings import Settings, get_settings

log = logging.getLogger(__name__)


# ─── LLM transport ─────────────────────────────────────────────────────


_REWRITE_SYSTEM_PROMPT = (
    "你是一个检索查询改写助手。用户给出一个查询,请生成多个语义相近但"
    "措辞不同的检索变体,目的是让基于向量 + BM25 的混合检索能从不同"
    "角度匹配到相关文档。\n"
    "严格要求:\n"
    "1. 输出必须是合法 JSON,不要任何额外解释、markdown 代码块或前后缀。\n"
    "2. 第一个变体必须是用户原始查询(逐字保留)。\n"
    "3. 其余变体在保留核心意图的前提下做同义改写、关键词扩展、视角切换。\n"
    "4. 变体数量 = 用户在请求里指定的数字;不要多也不要少。\n"
    "5. 变体之间不要重复。"
)


def _build_user_prompt(query: str, n_variants: int) -> str:
    """Build the user-side prompt asking for ``n_variants`` rewrites of ``query``.

    The prompt deliberately references ``n_variants`` so the model has
    an exact count to honour. We've seen off-by-one without an explicit
    number ("give me several variants" → 2, 3, 4, or 7 depending on the
    model).
    """
    return (
        f"原始查询:{query}\n\n"
        f"请输出恰好 {n_variants} 个改写变体,JSON 格式:\n"
        '{"variants": ["原句", "改写1", "改写2", ...]}\n'
        "原句必须出现在第 1 位,逐字保留。"
    )


def _extract_json(content: str) -> Any:
    """Tolerant JSON extraction for model responses.

    Tries strict :func:`json.loads` first (works when the model
    cooperates with ``response_format=json_object``). If that raises
    :class:`json.JSONDecodeError`, falls back to scanning for the first
    ``{...}`` or ``[...]`` block — common when a chat template
    accidentally wraps the JSON in a chat-tag preamble or a stray
    reasoning token sneaks past :func:`_strip_think_tags`. The block
    boundary is matched greedily because model-emitted JSON often
    nests arrays of objects.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try object block first (more common shape); fall back to array.
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"model response is not valid JSON: {content[:120]!r}")


def _strip_think_tags(content: str) -> str:
    """Drop ``<think>...</think>`` reasoning-model preambles.

    Some hosted models (Qwen3, DeepSeek-R1, …) wrap their actual answer
    in ``<think>...</think>`` reasoning tokens before the JSON payload.
    The LLM channel in :func:`mm_asset_rag.answer.llm_answer` strips
    these from the *answer*; we replicate here so the rewrite JSON
    parser doesn't trip on the leading ``<think>`` block.
    """
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _post_chat_json(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    timeout: float,
    max_tokens: int = 800,
) -> Any:
    """Single chat-completion POST that returns a parsed JSON payload.

    Asks the model for ``response_format={"type": "json_object"}`` —
    most OpenAI-compatible providers (and the VLM fallback) honour it,
    so we get a clean JSON string most of the time. The tolerant
    :func:`_extract_json` then handles the edge cases (markdown fences,
    reasoning tags, truncated output).

    Raises:
        requests.HTTPError: non-2xx response from the server.
        ValueError: response body was not parseable as JSON.
        KeyError: response was missing the ``choices[0].message.content``
            field (malformed OpenAI shape).
        requests.Timeout: the call exceeded ``timeout`` seconds.
    """
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "temperature": 0.3,  # low but non-zero so the variants aren't all identical
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError(f"chat completion returned non-string content: {type(content).__name__}")
    content = _strip_think_tags(content)
    return _extract_json(content)


# ─── Public: rewrite_query ─────────────────────────────────────────────


def _clamp_n_variants(n: int) -> int:
    """Clamp ``n`` to ``[1, 5]`` and coerce non-positive ints upward."""
    # ``bool`` is a subclass of ``int`` in Python; without this guard,
    # ``True`` and ``False`` would silently coerce to 1 and 0.
    if isinstance(n, bool):
        return 1
    if not isinstance(n, int):
        try:
            n = int(n)
        except (TypeError, ValueError):
            return 1
    return max(1, min(5, n))


def _coerce_variants(parsed: Any, original: str, n: int) -> list[str]:
    """Turn whatever the LLM returned into a clean ``[str, str, ...]`` list.

    Accepts three shapes the model emits in practice:

    - ``{"variants": [...]}`` — the documented contract.
    - ``{"queries": [...]}`` / ``{"rewrites": [...]}`` — synonyms some
      models reach for; we accept them defensively.
    - A bare JSON array ``[...]`` — also common. The
      :func:`_extract_json` regex picks the first ``[]`` block when the
      model wraps the array instead of an object.

    The returned list is deduped (case-sensitive, preserves order) and
    length-clamped to ``n``. The original ``query`` is always prepended
    so even a one-element LLM response yields a usable list.
    """
    candidates: list[str] = []
    if isinstance(parsed, dict):
        # Prefer "variants" (the documented key), then fall back to common synonyms.
        for key in ("variants", "queries", "rewrites", "alternatives"):
            value = parsed.get(key)
            if isinstance(value, list):
                candidates = [str(v).strip() for v in value if str(v).strip()]
                break
    elif isinstance(parsed, list):
        candidates = [str(v).strip() for v in parsed if str(v).strip()]
    else:
        candidates = []

    # Dedupe (preserve order) and clamp length.
    seen: set[str] = set()
    deduped: list[str] = []
    for v in candidates:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
        if len(deduped) >= n:
            break

    # Always keep the original query as the first entry — even if the
    # model omitted it, deduped it away, or returned garbage.
    if not deduped or deduped[0] != original:
        deduped = [original, *[v for v in deduped if v != original]]
    return deduped[:n] or [original]


def rewrite_query(
    query: str,
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Expand ``query`` into ``N`` LLM-generated variants.

    Returns a list of length ``[1, query_rewrite_n_variants]``. The
    original query is always present at index 0 (the LLM is explicitly
    prompted to keep it; we also enforce it defensively). Failure
    modes (no creds, timeout, HTTP error, malformed JSON) all return
    ``[query]`` so callers can treat the return value as "ready to
    search, even if the rewrite did nothing".

    Args:
        query: The user-submitted query. Empty string is short-circuited
            to ``[""]`` (hybrid_search handles empty queries itself).
        settings: Optional pre-built :class:`Settings` override for
            tests / scripted jobs. Defaults to :func:`get_settings`.

    Returns:
        A non-empty list of variant strings. Length is 1 (failure or
        rewrite disabled) up to ``query_rewrite_n_variants`` (clamped
        to 5).
    """
    if not query:
        return [""]

    settings = settings or get_settings()
    if not settings.query_rewrite_enabled:
        return [query]

    creds = settings.llm_creds
    if not (creds[0] and creds[1] and creds[2]):
        # No LLM configured: silent fallback. Log at debug to avoid
        # spamming logs every search when the user hasn't set up an LLM.
        log.debug("rewrite_query skipped: no LLM creds (set OPENAI_* or VLM_*)")
        return [query]

    n = _clamp_n_variants(settings.query_rewrite_n_variants)
    base_url, api_key, model = creds
    prompt = _build_user_prompt(query, n)

    try:
        parsed = _post_chat_json(
            base_url,
            api_key,
            model,
            prompt,
            timeout=float(settings.query_rewrite_timeout),
        )
    except (requests.Timeout, requests.HTTPError, requests.ConnectionError) as exc:
        log.warning("rewrite_query HTTP failure (%s): %s", type(exc).__name__, exc)
        return [query]
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        log.warning("rewrite_query parse failure: %s", exc)
        return [query]
    except Exception as exc:  # last-resort guard, never crash search
        log.warning("rewrite_query unexpected failure: %s", exc)
        return [query]

    return _coerce_variants(parsed, query, n)


# ─── Public: multi_query_search ────────────────────────────────────────


def _search_one(args: tuple[str, Path | None, int, float | None]) -> list[SearchHit]:
    """Worker for the thread pool: run ``hybrid_search`` for one variant.

    Module-level (not a lambda) so the worker is picklable across
    thread boundaries — some Python runtimes require pool callables to
    be top-level named functions, not lambdas. Returns the raw hits
    (no top_k slicing here; the caller merges and re-slices).
    """
    query, image_path, top_k, min_score = args
    return hybrid_search(query, image_path=image_path, top_k=top_k, min_score=min_score)


def multi_query_search(
    queries: list[str],
    *,
    top_k: int,
    image_path: Path | None = None,
    min_score: float | None = None,
    n_parallel: int = 4,
) -> list[SearchHit]:
    """Run :func:`hybrid_search` for each variant, fuse via rank-based RRF.

    The N variants are searched in parallel (a thread pool of size
    ``n_parallel``) because each ``hybrid_search`` is a blocking Qdrant
    round-trip. The hits are then merged with
    :func:`mm_asset_rag.retrieval.merge_hits` using uniform per-variant
    weights — rank-based RRF means the asset appearing in *multiple*
    variants accumulates higher score than the asset appearing in one
    (this is the whole point of multi-query RAG).

    Args:
        queries: Non-empty list of query variants. An empty list
            returns ``[]`` (defensive — caller-side guard).
        top_k: Final result length after fusion. Passed as the
            ``top_k`` of each per-variant ``hybrid_search`` *and* the
            ``top_k`` of the RRF merge.
        image_path: Optional image query for the image-to-image route.
            When ``N > 1``, the image-to-image route is computed **once**
            (variants don't change CLIP cosine) and added as a single
            group instead of being replicated N times — avoids N× CLIP
            encode + N× Qdrant round-trip waste.
        min_score: Forwarded to ``hybrid_search`` per variant and to the
            RRF merge. None lets each layer pick its own default
            (settings.min_score).
        n_parallel: Max concurrent ``hybrid_search`` invocations.
            Clamped to ``[1, max(1, len(queries))]``; a single-query
            input skips the pool entirely.

    Returns:
        RRF-fused search hits, ordered by descending fused score,
        length ``<= top_k``.
    """
    if not queries:
        return []

    if len(queries) == 1:
        # Avoid the pool overhead for the trivial case — the
        # ``query_rewrite_enabled=False`` path lands here after
        # ``rewrite_query`` returns ``[query]``.
        return hybrid_search(
            queries[0],
            image_path=image_path,
            top_k=top_k,
            min_score=min_score,
        )

    # Cap pool size at the number of variants; threads beyond that
    # number just sit idle. Floor at 1 so a 0 / negative ``n_parallel``
    # doesn't crash with ``max_workers <= 0``.
    workers = max(1, min(int(n_parallel), len(queries)))

    # Image-to-image dedup: the CLIP cosine match is invariant to text
    # variants, so ``image_path`` produces the same i2i hits for every
    # variant. Run it once outside the pool and reuse as a single RRF
    # group, otherwise we'd pay N× CLIP encode + N× Qdrant round-trip
    # for nothing. Single-query path keeps i2i inline inside
    # ``hybrid_search`` so behaviour is byte-identical for that case.
    groups: list[list[SearchHit]] = []
    weights: list[float] = []
    if image_path is not None:
        from .registry import get_backend

        try:
            i2i_hits = get_backend("qdrant").search_image(image_path=image_path, top_k=top_k)
        except Exception as exc:
            log.warning("multi_query_search i2i fetch failed (%s): %s", type(exc).__name__, exc)
            i2i_hits = []
        if i2i_hits:
            groups.append(i2i_hits)
            weights.append(1.0)

    # Per-variant text search: skip the image side (``image_path=None``)
    # because we already pulled i2i once above.
    args_list = [(q, None, top_k, min_score) for q in queries]

    # Use ``as_completed`` + per-future ``result()`` so a single variant's
    # exception (Qdrant 5xx, reranker load failure, timeout) is logged and
    # replaced with an empty list — the other variants still contribute.
    # ``pool.map`` propagates the first exception and aborts the merge.
    # We index results back by submission order (not completion order) so
    # the RRF tie-break matches the previous ``pool.map`` behaviour —
    # all RRF scores are equal across single-hit groups, so output order
    # is determined by group ordering.
    from concurrent.futures import as_completed

    variant_hits: list[list[SearchHit] | None] = [None] * len(args_list)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qr-rewrite") as pool:
        future_to_idx = {pool.submit(_search_one, args): i for i, args in enumerate(args_list)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            variant_q = queries[idx]
            try:
                variant_hits[idx] = fut.result()
            except Exception as exc:
                log.warning(
                    "multi_query_search variant failed (query=%r, %s): %s",
                    variant_q, type(exc).__name__, exc,
                )
                variant_hits[idx] = []

    groups.extend(h if h is not None else [] for h in variant_hits)
    weights.extend(1.0 for _ in variant_hits)

    # Uniform per-variant weight — rank-based RRF means the per-asset
    # contribution is ``weight / (RRF_K + rank)`` regardless of how the
    # route scaled its raw score. Equal weights give each variant an
    # equal voice; tweaking this would re-introduce the same
    # cross-variant score-scale coupling the route fusion was designed
    # to remove.
    return merge_hits(groups, weights, top_k=top_k, min_score=min_score or 0.0)


# ─── Public: hybrid_search_with_rewrite ────────────────────────────────


def hybrid_search_with_rewrite(
    query: str,
    *,
    image_path: Path | None = None,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[SearchHit]:
    """Top-level wrapper combining rewrite + multi-query fusion.

    When ``Settings.query_rewrite_enabled`` is ``False``, this is a
    transparent pass-through to :func:`mm_asset_rag.retrieval.hybrid_search`
    (single query, no rewrite, no fusion). When it's ``True``, the
    flow is:

    1. :func:`rewrite_query` → list of variants (``[query]`` on failure).
    2. :func:`multi_query_search` → RRF-fused hits.

    The two-step failure mode means a broken rewrite never costs the
    user anything — the worst case is identical to running
    ``hybrid_search(query)`` directly.
    """
    settings = get_settings()
    queries = rewrite_query(query, settings=settings)
    return multi_query_search(
        queries,
        top_k=top_k,
        image_path=image_path,
        min_score=min_score,
        n_parallel=settings.query_rewrite_concurrency,
    )


# ─── Public: text_search_with_rewrite ──────────────────────────────────


def _text_search_one(args: tuple[str, int]) -> list[SearchHit]:
    """Worker for the text-route thread pool — runs ``backend.search_text``.

    Module-level (not a lambda) so the worker is picklable across thread
    boundaries; mirrors :func:`_search_one` for the hybrid path.
    """
    from .registry import get_backend

    query, top_k = args
    return get_backend("qdrant").search_text(query=query, top_k=top_k)


def _multi_query_text(
    queries: list[str],
    *,
    top_k: int,
    n_parallel: int = 4,
    min_score: float | None = None,
) -> list[SearchHit]:
    """Run ``backend.search_text`` for each variant, fuse via rank-based RRF.

    Same shape as :func:`multi_query_search` but stays on the **text route
    only** — ``backend.search_text`` is the dense + BM25-en + BM25-zh
    prefetch inside Qdrant, so the fusion stays single-route even though
    we still get the multi-query RRF cross-variant accumulation. This is
    what :func:`mm_asset_rag.service.dispatch_search` routes
    ``mode="text"`` through when rewrite is enabled, preserving the
    pre-rewrite "text mode = text-only" semantics.

    ``min_score`` is forwarded to the final RRF merge so an upstream
    ``Settings.min_score`` (or a per-call override) actually filters the
    fused set — without this param the text path silently diverged from
    the hybrid path which does accept ``min_score``.
    """
    if not queries:
        return []
    if len(queries) == 1:
        from .registry import get_backend

        return get_backend("qdrant").search_text(query=queries[0], top_k=top_k)

    workers = max(1, min(int(n_parallel), len(queries)))

    # ``as_completed`` + per-future try/except so a single variant's
    # exception (Qdrant 5xx, reranker load failure) is logged and
    # replaced with an empty list — the other variants still contribute.
    # Index by submission order so RRF tie-break stays stable across
    # ``pool.map`` → ``as_completed`` migration.
    from concurrent.futures import as_completed

    args_list = [(q, top_k) for q in queries]
    variant_hits: list[list[SearchHit] | None] = [None] * len(args_list)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qr-text") as pool:
        future_to_idx = {pool.submit(_text_search_one, args): i for i, args in enumerate(args_list)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            variant_q = queries[idx]
            try:
                variant_hits[idx] = fut.result()
            except Exception as exc:
                log.warning(
                    "_multi_query_text variant failed (query=%r, %s): %s",
                    variant_q, type(exc).__name__, exc,
                )
                variant_hits[idx] = []

    groups = [h if h is not None else [] for h in variant_hits]
    weights = [1.0] * len(groups)
    return merge_hits(groups, weights, top_k=top_k, min_score=min_score or 0.0)


def text_search_with_rewrite(
    query: str,
    *,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[SearchHit]:
    """Single-route text wrapper for the rewrite path.

    Used by ``dispatch_search`` when ``mode="text"``: rewrite the query
    to N variants and run ``backend.search_text`` per variant, fusing
    via RRF. When ``Settings.query_rewrite_enabled`` is off this is a
    transparent pass-through to ``backend.search_text(query)`` — same
    behaviour as the pre-rewrite code path.

    Splitting this from :func:`hybrid_search_with_rewrite` matters
    because ``mode="text"`` historically maps to the text-only route
    (dense + BM25 channels) — silently switching it to the full hybrid
    would drag image routes into a "text-only" caller, breaking the
    ``mode`` contract that the API and CLI expose.

    ``min_score`` is forwarded to :func:`_multi_query_text` so the
    text-mode rewrite path applies the same ``Settings.min_score``
    filter as the hybrid-mode path and the pre-rewrite text path.
    """
    settings = get_settings()
    queries = rewrite_query(query, settings=settings)
    return _multi_query_text(
        queries,
        top_k=top_k,
        n_parallel=settings.query_rewrite_concurrency,
        min_score=min_score,
    )
