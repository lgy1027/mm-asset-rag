"""Unit tests for ``mm_asset_rag.query_rewrite``.

Covers the three public surfaces (``rewrite_query``,
``multi_query_search``, ``hybrid_search_with_rewrite``) plus the
``dispatch_search`` hook that routes text / hybrid queries through
``hybrid_search_with_rewrite`` when ``query_rewrite_enabled`` is on.

Pattern: ``monkeypatch.setattr`` on the module-level helper(s)
(``_post_chat_json``, ``hybrid_search``) so the LLM and the Qdrant
backend never run in tests. Mirrors the DI-hook style in
``test_answer_eval.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import requests

from mm_asset_rag import query_rewrite as qr
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.service import dispatch_search
from mm_asset_rag.settings import Settings, get_settings

# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def settings_no_llm(monkeypatch) -> Settings:
    """Settings with rewrite enabled but no LLM creds set.

    Forces the "missing creds → fall back to [query]" path. Override
    ``query_rewrite_enabled`` per-test via this fixture's instance
    fields.
    """
    get_settings.cache_clear()
    s = Settings(
        query_rewrite_enabled=True,
        query_rewrite_n_variants=3,
    )
    return s


@pytest.fixture
def settings_with_llm(monkeypatch) -> Settings:
    """Settings with rewrite enabled and a (fake) LLM triple set."""
    get_settings.cache_clear()
    s = Settings(
        query_rewrite_enabled=True,
        query_rewrite_n_variants=3,
        openai_api_key="sk-test",
        openai_base_url="https://api.example.com/v1",
        openai_model="gpt-test",
    )
    return s


def _stub_post_chat_json(payload: object):
    """Build a stub for ``qr._post_chat_json`` that ignores all args and returns ``payload``."""

    def _stub(*args, **kwargs):
        return payload

    return _stub


def _hit(asset_id: str, score: float, *, route: str = "text") -> SearchHit:
    """Build a SearchHit for test fixtures."""
    return SearchHit(
        route=route,
        score=score,
        asset_id=asset_id,
        title=asset_id,
        source_type="pdf",
        source_path=f"{asset_id}.pdf",
        evidence=f"evidence for {asset_id}",
        metadata={"page": 1},
    )


# ─── 1. rewrite_query ───────────────────────────────────────────────────


def test_rewrite_disabled_returns_original_only(monkeypatch) -> None:
    """``query_rewrite_enabled=False`` skips the LLM call entirely and
    returns ``[query]`` without touching ``_post_chat_json``."""
    monkeypatch.setattr(qr, "_post_chat_json", _stub_post_chat_json({"variants": ["x", "y"]}))
    settings = Settings(query_rewrite_enabled=False)
    out = qr.rewrite_query("hello", settings=settings)
    assert out == ["hello"]


def test_rewrite_no_llm_returns_original_only(monkeypatch) -> None:
    """Enabled but no LLM creds: silent fallback to ``[query]``, no raise."""
    calls: list[tuple] = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))
        return {"variants": ["x", "y"]}

    monkeypatch.setattr(qr, "_post_chat_json", _record)
    settings = Settings(query_rewrite_enabled=True)  # no openai_* set
    out = qr.rewrite_query("hello", settings=settings)
    assert out == ["hello"]
    assert calls == [], "LLM should not be called when creds are missing"


def test_rewrite_parses_json_variants(monkeypatch) -> None:
    """The documented contract shape ``{"variants": [...]}`` is parsed,
    length-clamped, and deduped."""
    monkeypatch.setattr(
        qr,
        "_post_chat_json",
        _stub_post_chat_json({"variants": ["用户原句", "改写 1", "改写 2", "改写 3"]}),
    )
    settings = Settings(
        query_rewrite_enabled=True,
        query_rewrite_n_variants=3,
        openai_api_key="sk-test",
        openai_base_url="https://api.example.com/v1",
        openai_model="gpt-test",
    )
    out = qr.rewrite_query("用户原句", settings=settings)
    # Original is preserved at index 0, three variants total.
    assert out[0] == "用户原句"
    assert len(out) == 3
    assert out == ["用户原句", "改写 1", "改写 2"]


def test_rewrite_parses_bare_list(monkeypatch) -> None:
    """Bare JSON array (not wrapped in an object) is accepted defensively."""
    monkeypatch.setattr(qr, "_post_chat_json", _stub_post_chat_json(["q1", "q2", "q3"]))
    settings = Settings(
        query_rewrite_enabled=True,
        query_rewrite_n_variants=3,
        openai_api_key="sk-test",
        openai_base_url="https://api.example.com/v1",
        openai_model="gpt-test",
    )
    out = qr.rewrite_query("orig", settings=settings)
    assert out[0] == "orig"
    assert "q1" in out and "q2" in out


def test_rewrite_parses_garbage_returns_original(monkeypatch) -> None:
    """Garbage JSON / missing key → ``[query]``, no exception propagates."""
    monkeypatch.setattr(qr, "_post_chat_json", _stub_post_chat_json({"unrelated": "garbage"}))
    settings = Settings(
        query_rewrite_enabled=True,
        query_rewrite_n_variants=3,
        openai_api_key="sk-test",
        openai_base_url="https://api.example.com/v1",
        openai_model="gpt-test",
    )
    out = qr.rewrite_query("原始查询", settings=settings)
    assert out == ["原始查询"]


def test_rewrite_timeout_returns_original(monkeypatch) -> None:
    """``requests.Timeout`` from the LLM → silent fallback to ``[query]``."""

    def _raise_timeout(*args, **kwargs):
        raise requests.Timeout("upstream slow")

    monkeypatch.setattr(qr, "_post_chat_json", _raise_timeout)
    settings = Settings(
        query_rewrite_enabled=True,
        query_rewrite_n_variants=3,
        openai_api_key="sk-test",
        openai_base_url="https://api.example.com/v1",
        openai_model="gpt-test",
    )
    out = qr.rewrite_query("hello", settings=settings)
    assert out == ["hello"]


def test_rewrite_n_variants_clamps_to_range() -> None:
    """``query_rewrite_n_variants`` outside ``[1, 5]`` is clamped, not respected literally."""
    # 0 → 1
    assert qr._clamp_n_variants(0) == 1
    # negative → 1
    assert qr._clamp_n_variants(-7) == 1
    # 100 → 5
    assert qr._clamp_n_variants(100) == 5
    # 3 stays 3
    assert qr._clamp_n_variants(3) == 3
    # 5 stays 5
    assert qr._clamp_n_variants(5) == 5
    # Non-int (string) coerces
    assert qr._clamp_n_variants("3") == 3
    # Garbage → 1 (defensive default)
    assert qr._clamp_n_variants("garbage") == 1
    # bool is a subclass of int — without the guard, ``True`` (=1) would
    # slip through and ``False`` (=0) would coerce to 1 too. Both pinned.
    assert qr._clamp_n_variants(True) == 1
    assert qr._clamp_n_variants(False) == 1


def test_rewrite_empty_query_returns_empty_string(monkeypatch) -> None:
    """Empty query short-circuits to ``[""]`` without any LLM call."""
    calls: list[tuple] = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))
        return {"variants": ["x"]}

    monkeypatch.setattr(qr, "_post_chat_json", _record)
    settings = Settings(
        query_rewrite_enabled=True,
        query_rewrite_n_variants=3,
        openai_api_key="sk-test",
        openai_base_url="https://api.example.com/v1",
        openai_model="gpt-test",
    )
    out = qr.rewrite_query("", settings=settings)
    assert out == [""]
    assert calls == [], "LLM should not be called for empty query"


# ─── 2. multi_query_search ─────────────────────────────────────────────


def test_multi_query_single_returns_hybrid_search(monkeypatch) -> None:
    """Single-element list short-circuits to one ``hybrid_search`` call
    (no thread pool)."""
    fake_hits = [_hit("a", 0.9), _hit("b", 0.8)]
    calls: list[tuple] = []

    def _fake_hybrid(query, *, image_path=None, top_k=5, min_score=None):
        calls.append((query, image_path, top_k, min_score))
        return fake_hits

    monkeypatch.setattr(qr, "hybrid_search", _fake_hybrid)
    out = qr.multi_query_search(["only"], top_k=5)
    assert out == fake_hits
    assert len(calls) == 1
    assert calls[0][0] == "only"


def test_multi_query_fuses_via_rrf(monkeypatch) -> None:
    """Two variants with overlapping hits → RRF merge lifts the overlapping asset."""
    # Variant 1 returns A (top) + C (lower).
    # Variant 2 returns A (top) + B (lower). After RRF, A appears in both,
    # so it should accumulate a higher score than B or C which appear once.
    calls: list[str] = []

    def _fake_hybrid(query, *, image_path=None, top_k=5, min_score=None):
        calls.append(query)
        if "alpha" in query:
            return [_hit("a", 0.9), _hit("c", 0.5)]
        return [_hit("a", 0.8), _hit("b", 0.7)]

    monkeypatch.setattr(qr, "hybrid_search", _fake_hybrid)
    out = qr.multi_query_search(["alpha query", "beta query"], top_k=5)

    # A is the only asset in *both* variant result sets → top-1 after RRF.
    assert out[0].asset_id == "a"
    # B and C each appear in only one variant → below A.
    asset_ids = [h.asset_id for h in out]
    assert set(asset_ids) == {"a", "b", "c"}
    assert len(calls) == 2
    # A's fused RRF score must beat either single-variant asset's contribution.
    a_score = next(h.score for h in out if h.asset_id == "a")
    b_score = next(h.score for h in out if h.asset_id == "b")
    c_score = next(h.score for h in out if h.asset_id == "c")
    assert a_score > b_score
    assert a_score > c_score


def test_multi_query_respects_n_parallel(monkeypatch) -> None:
    """``n_parallel`` caps the worker pool; ``hybrid_search`` is called exactly N times."""
    queries = [f"q{i}" for i in range(6)]
    fake_hits_per_query = {q: [_hit(f"asset-{i}", 0.5)] for i, q in enumerate(queries)}

    def _fake_hybrid(query, *, image_path=None, top_k=5, min_score=None):
        return fake_hits_per_query[query]

    # Patch the pool class to inspect its ``max_workers`` so we can
    # assert the cap without needing real concurrency semantics.
    real_pool = ThreadPoolExecutor
    captured: dict[str, int] = {}

    class _Pool(real_pool):
        def __init__(self, max_workers=None, **kw):
            captured["max_workers"] = int(max_workers)
            super().__init__(max_workers=max_workers, **kw)

    monkeypatch.setattr(qr, "hybrid_search", _fake_hybrid)
    monkeypatch.setattr(qr, "ThreadPoolExecutor", _Pool)

    out = qr.multi_query_search(queries, top_k=5, n_parallel=2)
    # 6 unique assets, but top_k=5 caps the merged output at 5 — proves
    # top_k is honoured by the RRF merge, not just by the per-variant
    # hybrid_search.
    assert len(out) == 5
    assert {h.asset_id for h in out} == {f"asset-{i}" for i in range(5)}
    assert captured["max_workers"] == 2

    # n_parallel > len(queries) is also clamped to len(queries).
    captured.clear()
    qr.multi_query_search(queries[:2], top_k=5, n_parallel=8)
    assert captured["max_workers"] == 2


# ─── 2.5. text_search_with_rewrite (single-route text variant) ─────────


def test_text_search_with_rewrite_disabled_passes_through(monkeypatch) -> None:
    """``QUERY_REWRITE_ENABLED=false`` → ``text_search_with_rewrite`` calls
    ``backend.search_text`` exactly once with the original query. No rewrite
    LLM call, no multi-query fanout.
    """
    calls: list[str] = []

    class _Backend:
        def search_text(self, *, query, top_k):
            calls.append(query)
            return [_hit("a", 0.9)]

    monkeypatch.setattr("mm_asset_rag.registry.get_backend", lambda name: _Backend())
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "false")
    get_settings.cache_clear()

    out = qr.text_search_with_rewrite("q", top_k=3)
    assert [h.asset_id for h in out] == ["a"]
    assert calls == ["q"], (
        "rewrite off should hit backend.search_text exactly once with the original"
    )


def test_text_search_with_rewrite_enabled_fans_out_per_variant(monkeypatch) -> None:
    """``QUERY_REWRITE_ENABLED=true`` + 3 variants → ``backend.search_text``
    is called 3 times (once per variant) and the results are RRF-merged.
    """
    from mm_asset_rag import query_rewrite as qr_mod

    # Stub the LLM rewrite to produce 3 variants.
    monkeypatch.setattr(
        qr_mod,
        "rewrite_query",
        lambda query, settings=None: [query, f"{query} alpha", f"{query} beta"],
    )
    # Track which variants were searched and return hits that surface
    # ``a`` as a recurring asset across variants — RRF should rank it top.
    calls: list[str] = []

    class _Backend:
        def search_text(self, *, query, top_k):
            calls.append(query)
            if "alpha" in query:
                return [_hit("a", 0.9), _hit("c", 0.5)]
            if "beta" in query:
                return [_hit("a", 0.8), _hit("b", 0.7)]
            return [_hit("a", 0.95)]

    monkeypatch.setattr("mm_asset_rag.registry.get_backend", lambda name: _Backend())
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    monkeypatch.setenv("QUERY_REWRITE_N_VARIANTS", "3")
    get_settings.cache_clear()

    out = qr.text_search_with_rewrite("q", top_k=5)
    assert len(calls) == 3
    assert calls[0] == "q"
    assert set(calls[1:]) == {"q alpha", "q beta"}
    # ``a`` appears in all 3 variants → top of the fused list.
    assert out[0].asset_id == "a"


def test_text_search_with_rewrite_never_touches_hybrid(monkeypatch) -> None:
    """Lock down the text-only semantics: ``text_search_with_rewrite`` must
    not invoke :func:`mm_asset_rag.retrieval.hybrid_search` even when the
    rewrite LLM returns N variants. (Otherwise ``mode="text"`` callers would
    silently gain image routes, breaking the API contract.)
    """
    from mm_asset_rag import query_rewrite as qr_mod

    monkeypatch.setattr(
        qr_mod, "rewrite_query", lambda query, settings=None: [query, f"{query} alt"]
    )

    class _Backend:
        def search_text(self, *, query, top_k):
            return [_hit("a", 0.9)]

    monkeypatch.setattr("mm_asset_rag.registry.get_backend", lambda name: _Backend())
    # Trip-wire: if ``hybrid_search`` is called, the test fails loudly.
    monkeypatch.setattr(
        qr_mod,
        "hybrid_search",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("hybrid_search must not be called")),
    )
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    get_settings.cache_clear()

    out = qr.text_search_with_rewrite("q", top_k=3)
    assert [h.asset_id for h in out] == ["a"]


# ─── 3. dispatch_search hook ───────────────────────────────────────────


def test_dispatch_search_text_uses_text_rewrite_wrapper(monkeypatch) -> None:
    """``mode=text`` routes through the **text-only** rewrite wrapper
    (``text_search_with_rewrite``) — never the hybrid wrapper. This
    preserves the historical ``mode=text`` = text-only contract: a
    user / API caller that picks ``text`` must not have image routes
    silently pulled into their result list. The hybrid wrapper stays
    for ``mode=hybrid`` (see
    :func:`test_dispatch_search_hybrid_uses_rewrite_when_enabled`).
    """
    from mm_asset_rag import search_service as service_mod

    text_calls: list[tuple] = []
    hybrid_calls: list[tuple] = []

    def _fake_text(query, *, top_k=5, min_score=None, backend=None):
        text_calls.append((query, top_k, min_score))
        return []

    def _fake_hybrid(query, *, image_path=None, top_k=5, min_score=None, backend=None):
        hybrid_calls.append((query, image_path, top_k, min_score))
        return []

    monkeypatch.setattr(service_mod, "text_search_with_rewrite", _fake_text)
    monkeypatch.setattr(service_mod, "hybrid_search_with_rewrite", _fake_hybrid)

    # Enabled: text wrapper is called; hybrid wrapper is NOT.
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    get_settings.cache_clear()
    dispatch_search(query="q", mode="text", image_path=None, top_k=5)
    assert len(text_calls) == 1, "text mode must funnel through text_search_with_rewrite"
    assert hybrid_calls == [], "text mode must NOT touch hybrid_search_with_rewrite"

    # Disabled: wrapper is *still* called — the pass-through to
    # ``backend.search_text`` happens inside the wrapper, not at dispatch.
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "false")
    get_settings.cache_clear()
    text_calls.clear()
    dispatch_search(query="q", mode="text", image_path=None, top_k=5)
    assert len(text_calls) == 1, (
        "text mode always goes through the text wrapper (rewrite is internal)"
    )
    assert hybrid_calls == [], "text mode must never reach the hybrid wrapper"


def test_dispatch_search_image_modes_skip_rewrite(monkeypatch) -> None:
    """``text-to-image`` and ``image-to-image`` bypass ``hybrid_search_with_rewrite``
    even when ``query_rewrite_enabled=True`` — the rewrite only helps the text side."""
    from pathlib import Path

    from mm_asset_rag import search_service as service_mod

    rewrite_calls: list[tuple] = []

    def _fake_rewrite(query, *, image_path=None, top_k=5, min_score=None, backend=None):
        rewrite_calls.append((query, image_path, top_k, min_score))
        return []

    # Stub the sandbox resolver so we don't need a real on-disk asset
    # for the image-to-image path — the test is about routing, not
    # about file existence checks (those have their own suite).
    monkeypatch.setattr(
        service_mod, "resolve_sandboxed_image_path", lambda p: Path("/fake/img.png") if p else None
    )
    monkeypatch.setattr(service_mod, "hybrid_search_with_rewrite", _fake_rewrite)
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    get_settings.cache_clear()

    # text-to-image should go through the backend, not the rewrite wrapper.
    with patch("mm_asset_rag.search_service.get_backend") as get_backend:
        backend = get_backend.return_value
        backend.search_text_to_image.return_value = []
        dispatch_search(query="q", mode="text-to-image", image_path=None, top_k=5)
        backend.search_text_to_image.assert_called_once()
    assert rewrite_calls == [], "text-to-image must not invoke the rewrite wrapper"

    # image-to-image (with sandboxed path stubbed) — rewrite is also skipped here.
    with patch("mm_asset_rag.search_service.get_backend") as get_backend:
        backend = get_backend.return_value
        backend.search_image.return_value = []
        dispatch_search(query="q", mode="image-to-image", image_path="img.png", top_k=5)
        backend.search_image.assert_called_once()
    assert rewrite_calls == [], "image-to-image must not invoke the rewrite wrapper"


def test_dispatch_search_hybrid_uses_rewrite_when_enabled(monkeypatch) -> None:
    """``mode=hybrid`` (the default for many endpoints) also funnels
    through ``hybrid_search_with_rewrite`` when enabled."""
    from mm_asset_rag import search_service as service_mod

    rewrite_calls: list[tuple] = []

    def _fake_rewrite(query, *, image_path=None, top_k=5, min_score=None, backend=None):
        rewrite_calls.append((query, image_path, top_k, min_score))
        return []

    monkeypatch.setattr(service_mod, "hybrid_search_with_rewrite", _fake_rewrite)
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    get_settings.cache_clear()

    dispatch_search(query="q", mode="hybrid", image_path=None, top_k=5)
    assert len(rewrite_calls) == 1


# ─── 4. Reviewer-fix regression tests ──────────────────────────────────


def test_multi_query_dedupes_image_to_image(monkeypatch) -> None:
    """``multi_query_search`` calls ``backend.search_image`` exactly once
    even when N variants are fanned out — i2i is invariant to text
    variants and would otherwise be Nx wasted CLIP encode + Qdrant
    round-trip."""
    from mm_asset_rag import registry

    queries = ["q0", "q1", "q2", "q3"]
    fake_text_hits = {q: [_hit(f"text-{q}", 0.5, route="text")] for q in queries}

    def _fake_hybrid(query, *, image_path=None, top_k=5, min_score=None):
        # Image side should already be None here — dedup pulls it
        # out before fanning out text searches.
        assert image_path is None, "i2i should be pulled by multi_query_search, not hybrid_search"
        return fake_text_hits[query]

    i2i_calls: list[int] = []

    class _Backend:
        def search_image(self, *, image_path, top_k):
            i2i_calls.append(1)
            return [_hit("i2i-asset", 0.9, route="image-to-image")]

        def search_text(self, *, query, top_k):
            return fake_text_hits[query]

    monkeypatch.setattr(qr, "hybrid_search", _fake_hybrid)
    monkeypatch.setattr(registry, "get_backend", lambda name: _Backend())

    out = qr.multi_query_search(queries, top_k=5, image_path="img.png")

    # i2i pulled exactly once even with 4 variants.
    assert len(i2i_calls) == 1
    # i2i asset surfaces in the fused output.
    asset_ids = {h.asset_id for h in out}
    assert "i2i-asset" in asset_ids


def test_multi_query_isolates_variant_exceptions(monkeypatch) -> None:
    """A single variant's ``hybrid_search`` exception is logged + that
    variant contributes an empty list — the other variants still
    contribute. Without this guard a single Qdrant 5xx / reranker load
    failure on variant 1 would 500 the whole search."""
    queries = ["ok-a", "broken", "ok-c"]

    def _fake_hybrid(query, *, image_path=None, top_k=5, min_score=None):
        if query == "broken":
            raise RuntimeError("simulated Qdrant 5xx")
        return [_hit(f"asset-{query}", 0.5)]

    monkeypatch.setattr(qr, "hybrid_search", _fake_hybrid)

    out = qr.multi_query_search(queries, top_k=5)

    # Two surviving variants contribute; the third's slot is an empty list.
    asset_ids = {h.asset_id for h in out}
    assert "asset-ok-a" in asset_ids
    assert "asset-ok-c" in asset_ids
    assert not any("asset-broken" in a for a in asset_ids)


def test_text_search_with_rewrite_forwards_min_score(monkeypatch) -> None:
    """``text_search_with_rewrite`` accepts ``min_score`` and forwards it
    through to the RRF merge — closes the MEDIUM-1 asymmetric gap
    where ``mode='text'`` silently dropped the settings floor."""
    captured: list[dict] = []

    def _fake_multi_text(queries, *, top_k, n_parallel=4, min_score=None):
        captured.append({"min_score": min_score, "n_groups": len(queries)})
        return [_hit("only-asset", 0.5)]

    monkeypatch.setattr(qr, "_multi_query_text", _fake_multi_text)
    monkeypatch.setattr(qr, "rewrite_query", lambda q, settings=None: [q])

    qr.text_search_with_rewrite("hello", top_k=5, min_score=0.01)
    assert captured[0]["min_score"] == 0.01
    assert captured[0]["n_groups"] == 1


def test_dispatch_search_rejects_unknown_mode() -> None:
    """LOW-7: a typo'd ``mode`` (``"typo"``) raises ``ValueError``
    instead of silently falling through to ``hybrid``."""
    with pytest.raises(ValueError, match="unknown mode"):
        dispatch_search(query="q", mode="typo", image_path=None, top_k=5)
