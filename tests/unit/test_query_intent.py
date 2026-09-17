"""Tests for ``mm_asset_rag.query_intent`` + ``hybrid_search`` intent routing.

The Qdrant client is mocked via the shared ``fake_qdrant_client`` /
``fixed_vector`` fixtures so we can exercise the intent → RRF weight
plumbing end-to-end without a live embedder or backend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mm_asset_rag.core.schema import SearchHit
from mm_asset_rag.query import retrieval
from mm_asset_rag.query.query_intent import (
    DEFAULT_INTENT_WEIGHTS,
    IntentWeights,
    QueryIntent,
    _parse_intent_weights_json,
    classify_intent,
    weights_for_intent,
)


def _make_hit(asset_id: str, route: str, score: float) -> SearchHit:
    return SearchHit(
        route=route,
        score=score,
        asset_id=asset_id,
        title=asset_id,
        source_type="pdf",
        source_path=f"{asset_id}.pdf",
        evidence=f"evidence-for-{asset_id}",
    )


# ─── classify_intent rules ───────────────────────────────────────────────


def test_classify_intent_chinese_cjk_heavy() -> None:
    """A 30-char all-CJK query → CHINESE regardless of length."""
    q = "请介绍一下联宝科技在ESG方面的实践和成果以及未来发展计划。"
    assert len(q) >= 30
    assert classify_intent(q) == QueryIntent.CHINESE


def test_classify_intent_short_keyword() -> None:
    """A 5-char keyword like '联宝 ESG' is precise (no stopword, ≤12 chars)."""
    assert classify_intent("联宝 ESG") == QueryIntent.PRECISE_KEYWORD


def test_classify_intent_descriptive_long() -> None:
    """A long descriptive English sentence → DESCRIPTIVE (≥30 chars after strip).

    Per the classifier's rule order, CJK ratio ≥ 70% fires the CHINESE
    branch first, so a long descriptive must be primarily Latin to
    exercise the DESCRIPTIVE branch. (The semantically-descriptive
    all-CJK query "请详细介绍联宝科技..." goes to CHINESE — the
    CHINESE intent is *both* a language profile and a length-tag,
    so it covers all-CJK regardless of length.)
    """
    q = "Please describe in detail the ESG practices and achievements of the company and its future plans"
    assert len(q.replace(" ", "")) >= 30
    assert classify_intent(q) == QueryIntent.DESCRIPTIVE


def test_classify_intent_entity_default() -> None:
    """A 13-char English query (no stopword, not CJK-heavy) → ENTITY_LOOKUP."""
    # "BERT" (4 chars) and "Diffusion" (9 chars) both fit the PRECISE branch
    # (≤12 chars, no stopword). Pick something in the 13-29 char window so
    # it's neither PRECISE (too long) nor DESCRIPTIVE (too short).
    assert classify_intent("large language model") == QueryIntent.ENTITY_LOOKUP


def test_classify_intent_mixed_cjk_falls_to_descriptive() -> None:
    """Mixed CJK < 70% AND long-ish → DESCRIPTIVE (the length branch wins)."""
    q = "对比 BERT 与 GPT 这两个 transformer 模型在 NLP 任务上的表现差异、应用场景分析以及未来发展趋势"
    # 30+ chars after strip; deliberately mixed CJK + Latin, with Latin
    # pushing CJK ratio < 70% so the length branch (≥30 chars) wins.
    compact = q.replace(" ", "")
    assert len(compact) >= 30
    intent = classify_intent(q)
    assert intent == QueryIntent.DESCRIPTIVE, (
        f"expected DESCRIPTIVE (CJK<70% and ≥30 chars), got {intent}"
    )


def test_classify_intent_stops_words_not_precise() -> None:
    """A short query containing a Chinese stopword is not PRECISE_KEYWORD."""
    # "什么是 ESG" → 8 chars but contains "是"; should NOT be PRECISE_KEYWORD.
    intent = classify_intent("什么是 ESG")
    assert intent != QueryIntent.PRECISE_KEYWORD
    # 8 chars < 30 so not DESCRIPTIVE; not CJK-heavy enough so not CHINESE.
    assert intent == QueryIntent.ENTITY_LOOKUP


def test_classify_intent_handles_whitespace_strip() -> None:
    """Leading/trailing whitespace is stripped before the length check."""
    # "  ESG  " → stripped is "ESG" → 3 chars, no stopword → PRECISE_KEYWORD.
    assert classify_intent("  ESG  ") == QueryIntent.PRECISE_KEYWORD


def test_classify_intent_short_english_question_is_not_precise() -> None:
    """MEDIUM-3: short English interrogative queries (``how to use the
    API``, ``what is AI``) are NOT PRECISE_KEYWORD — they'd otherwise
    down-weight the dense channel which is the opposite of what a
    paraphrase query wants. Should fall to ENTITY_LOOKUP."""
    # 14 chars no whitespace → 14 ≤ 12? No (with space). Without space
    # the compact form is 14 chars, also > 12. But the rule order still
    # routes them via step 4 (entity fallback) once the English stopword
    # short-circuits step 2. Pick queries where the compact length is
    # *also* ≤ 12 so the precise rule actually fires when stopwords
    # weren't checked.
    assert classify_intent("how to use") != QueryIntent.PRECISE_KEYWORD
    assert classify_intent("how to use") == QueryIntent.ENTITY_LOOKUP

    # "what is AI" — 9 chars after stripping space (compact), contains
    # "what"/"is" → must NOT be PRECISE_KEYWORD, falls to ENTITY_LOOKUP.
    assert classify_intent("what is AI") != QueryIntent.PRECISE_KEYWORD

    # Bare entity — still PRECISE_KEYWORD (no question token).
    assert classify_intent("BERT") == QueryIntent.PRECISE_KEYWORD
    assert classify_intent("Diffusion") == QueryIntent.PRECISE_KEYWORD


def test_parse_intent_weights_rejects_negative(caplog) -> None:
    """LOW-8: a negative weight is meaningless (merge_hits would silently
    drop the route); parser must refuse and fall back to defaults so the
    deployer gets a visible log entry rather than a silently-degraded
    search."""
    import logging

    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.query.query_intent"):
        # CSV shape
        assert _parse_intent_weights_json("-0.5,0.2,0.15") is None
        # JSON shape
        assert (
            _parse_intent_weights_json('{"text":-1,"text_to_image":0.2,"image_to_image":0.15}')
            is None
        )
    # And the warning was logged at least once.
    assert any("must be non-negative" in rec.message for rec in caplog.records)


def test_classify_intent_empty_falls_to_entity() -> None:
    """Empty / whitespace-only query is the safe default (ENTITY_LOOKUP)."""
    assert classify_intent("") == QueryIntent.ENTITY_LOOKUP
    assert classify_intent("   ") == QueryIntent.ENTITY_LOOKUP


# ─── _parse_intent_weights_json ──────────────────────────────────────────


def test_weights_override_parses_json() -> None:
    """Both JSON object and CSV triple parse to IntentWeights."""
    parsed = _parse_intent_weights_json('{"text":0.7,"text_to_image":0.2,"image_to_image":0.15}')
    assert parsed == IntentWeights(text=0.7, text_to_image=0.2, image_to_image=0.15)
    parsed_csv = _parse_intent_weights_json("0.7,0.2,0.15")
    assert parsed_csv == IntentWeights(text=0.7, text_to_image=0.2, image_to_image=0.15)


def test_weights_override_invalid_json_falls_to_default() -> None:
    """Garbage JSON logs a warning and returns None (caller falls back)."""
    assert _parse_intent_weights_json("garbage") is None
    # CSV with wrong number of fields
    assert _parse_intent_weights_json("0.5,0.5") is None
    # JSON missing required field
    assert _parse_intent_weights_json('{"text":0.7}') is None
    # JSON with non-numeric value
    assert (
        _parse_intent_weights_json('{"text":"abc","text_to_image":0.2,"image_to_image":0.15}')
        is None
    )


def test_weights_for_intent_picks_settings_override() -> None:
    """When settings has a JSON override, weights_for_intent uses it."""
    from types import SimpleNamespace

    settings = SimpleNamespace(
        hybrid_intent_weights_precise_keyword='{"text":0.5,"text_to_image":0.5,"image_to_image":0.0}',
        hybrid_intent_weights_descriptive=None,
        hybrid_intent_weights_entity_lookup=None,
        hybrid_intent_weights_chinese="0.9,0.05,0.05",  # CSV override
    )
    precise = weights_for_intent(QueryIntent.PRECISE_KEYWORD, settings)
    assert precise == IntentWeights(text=0.5, text_to_image=0.5, image_to_image=0.0)
    chinese = weights_for_intent(QueryIntent.CHINESE, settings)
    assert chinese == IntentWeights(text=0.9, text_to_image=0.05, image_to_image=0.05)
    # Unset falls back to default
    desc = weights_for_intent(QueryIntent.DESCRIPTIVE, settings)
    assert desc == DEFAULT_INTENT_WEIGHTS[QueryIntent.DESCRIPTIVE]


def test_weights_for_intent_invalid_json_falls_to_default() -> None:
    """Invalid JSON on a settings field → log warning + use default."""
    from types import SimpleNamespace

    settings = SimpleNamespace(
        hybrid_intent_weights_precise_keyword="garbage",
        hybrid_intent_weights_descriptive=None,
        hybrid_intent_weights_entity_lookup=None,
        hybrid_intent_weights_chinese=None,
    )
    # Should still return the default, not raise
    assert (
        weights_for_intent(QueryIntent.PRECISE_KEYWORD, settings)
        == DEFAULT_INTENT_WEIGHTS[QueryIntent.PRECISE_KEYWORD]
    )


def test_weights_for_intent_no_settings() -> None:
    """When settings is None, fall back to the hardcoded default table."""
    assert (
        weights_for_intent(QueryIntent.ENTITY_LOOKUP, None)
        == DEFAULT_INTENT_WEIGHTS[QueryIntent.ENTITY_LOOKUP]
    )


# ─── hybrid_search → per-intent weights ─────────────────────────────────


@pytest.fixture
def _stub_qdrant(monkeypatch):
    """Pin the three Qdrant backends to empty / canned hits for hybrid_search tests."""
    backend = SimpleNamespace(
        search_text=lambda *, query, top_k: [_make_hit("a", "qdrant_text", 1.0)],
        search_text_to_image=lambda *, query, top_k: [_make_hit("b", "qdrant_text_to_image", 1.0)],
        search_image=lambda *, image_path, top_k: [_make_hit("c", "qdrant_image_to_image", 1.0)],
    )
    monkeypatch.setattr("mm_asset_rag.core.registry.get_backend", lambda name: backend)


def test_hybrid_search_picks_weights_per_intent(monkeypatch, fixed_vector, _stub_qdrant) -> None:
    """With intent routing ON, classify_intent() drives the captured weights."""
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)
    monkeypatch.setattr(settings, "hybrid_intent_routing_enabled", True)

    captured: dict[str, list[float]] = {}

    real_merge = retrieval.merge_hits

    def _fake_merge(groups, weights, top_k, **kwargs):
        captured["weights"] = list(weights)
        return real_merge(groups, weights, top_k, **kwargs)

    monkeypatch.setattr("mm_asset_rag.query.retrieval.merge_hits", _fake_merge)

    # A descriptive Chinese sentence (CJK-heavy + long) → CHINESE intent.
    # CHINESE weights: 0.70 / 0.20 / 0.15 — all three > 0 so all routes fetched.
    q = "请详细介绍联宝科技在 ESG 方面的实践和成果以及未来发展计划"
    retrieval.hybrid_search(q, image_path=__import__("pathlib").Path("/tmp/x.png"))

    # Weights list passed to merge_hits: [text, text_to_image, image_to_image]
    assert captured["weights"] == [
        DEFAULT_INTENT_WEIGHTS[QueryIntent.CHINESE].text,
        DEFAULT_INTENT_WEIGHTS[QueryIntent.CHINESE].text_to_image,
        DEFAULT_INTENT_WEIGHTS[QueryIntent.CHINESE].image_to_image,
    ]


def test_hybrid_search_weights_override_bypasses_classify(
    monkeypatch, fixed_vector, _stub_qdrant
) -> None:
    """Explicit weights_override skips classify_intent and the routing switch."""
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)
    # Even with intent routing ON, the override should win.
    monkeypatch.setattr(settings, "hybrid_intent_routing_enabled", True)

    captured: dict[str, list[float]] = {}
    real_merge = retrieval.merge_hits

    def _fake_merge(groups, weights, top_k, **kwargs):
        captured["weights"] = list(weights)
        return real_merge(groups, weights, top_k, **kwargs)

    monkeypatch.setattr("mm_asset_rag.query.retrieval.merge_hits", _fake_merge)

    # If classify_intent ran, "联宝 ESG" → PRECISE_KEYWORD (0.60/0.20/0.15).
    # The override forces a different triple; that triple should be captured.
    # image_path is supplied so all 3 weights are forwarded to merge_hits.
    from pathlib import Path

    override = IntentWeights(text=0.55, text_to_image=0.30, image_to_image=0.15)
    retrieval.hybrid_search("联宝 ESG", image_path=Path("/tmp/x.png"), weights_override=override)

    assert captured["weights"] == [0.55, 0.30, 0.15]


def test_hybrid_search_image_to_image_weight_uses_intent(monkeypatch, fixed_vector) -> None:
    """When the chosen intent's image_to_image weight is 0, the i2i route is skipped."""
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)

    called = {"i2i": 0}

    def _track(*, image_path, top_k):
        called["i2i"] += 1
        return []

    backend = SimpleNamespace(
        search_text=lambda *, query, top_k: [],
        search_text_to_image=lambda *, query, top_k: [],
        search_image=_track,
    )

    # IntentWeights with image_to_image=0 — even with an image_path, the route
    # must NOT be called (mirrors the historical `weight<=0` skip behaviour).
    override = IntentWeights(text=0.80, text_to_image=0.20, image_to_image=0.0)

    from pathlib import Path

    retrieval.hybrid_search(
        "anything",
        image_path=Path("/tmp/none.png"),
        weights_override=override,
        backend=backend,
    )
    assert called["i2i"] == 0


def test_hybrid_search_default_routing_off(monkeypatch, fixed_vector, _stub_qdrant) -> None:
    """With routing OFF (default), the global hybrid_weight_* triple is used."""
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)
    monkeypatch.setattr(settings, "hybrid_intent_routing_enabled", False)
    monkeypatch.setattr(settings, "hybrid_weight_text", 0.71)
    monkeypatch.setattr(settings, "hybrid_weight_text_to_image", 0.29)
    # i2i weight default 0.15 > 0, so route will be appended.

    captured: dict[str, list[float]] = {}
    real_merge = retrieval.merge_hits

    def _fake_merge(groups, weights, top_k, **kwargs):
        captured["weights"] = list(weights)
        return real_merge(groups, weights, top_k, **kwargs)

    monkeypatch.setattr("mm_asset_rag.query.retrieval.merge_hits", _fake_merge)

    from pathlib import Path

    retrieval.hybrid_search("anything", image_path=Path("/tmp/x.png"))

    # Global triple: text=0.71, text_to_image=0.29, image_to_image=0.15.
    assert captured["weights"] == [0.71, 0.29, 0.15]


def test_hybrid_search_routing_off_with_override(monkeypatch, fixed_vector, _stub_qdrant) -> None:
    """Override beats the routing switch (so tests don't need to flip the switch)."""
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", False)
    monkeypatch.setattr(settings, "hybrid_intent_routing_enabled", False)

    captured: dict[str, list[float]] = {}
    real_merge = retrieval.merge_hits

    def _fake_merge(groups, weights, top_k, **kwargs):
        captured["weights"] = list(weights)
        return real_merge(groups, weights, top_k, **kwargs)

    monkeypatch.setattr("mm_asset_rag.query.retrieval.merge_hits", _fake_merge)

    # Override with non-default triple; routing is OFF so global weights are
    # irrelevant. No image_path → 2 weights.
    override = IntentWeights(text=0.99, text_to_image=0.01, image_to_image=0.50)
    retrieval.hybrid_search("anything", weights_override=override)

    assert captured["weights"] == [0.99, 0.01]
