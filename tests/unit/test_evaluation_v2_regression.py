"""Auto-eval regression test for ``evaluation_v2``.

This is the CI gate: when the retriever changes, this test runs in
under a second against a fixed mock corpus and asserts that hit_rate
does not regress below a recorded floor. The full v2 eval (with a
real Qdrant collection + ollama embedding) is the manual
``/tmp/run_v2_eval.py`` path; this test exists to catch the obvious
"someone changed ``_match`` and now nothing hits" class of bug
before it lands.

How it works
============

* A small fixed corpus of 20 assets (``MOCK_FULL_IDS``) is passed via
  the ``full_ids`` kwarg ``run_text_to_text_eval_v2`` accepts. No
  ``asset_index.jsonl`` is read.
* A canned ``search_fn`` returns deterministic top-k results for a
  small set of "golden" queries. These results were captured from
  the live retriever at v4 baseline.
* Per-group hit_rate@5 is asserted against a minimum threshold. If
  someone breaks ``_match`` / ``_expand`` / ``_title_of`` such that
  these golden queries stop hitting, the test fails.

Adding a new query
==================

1. Run ``/tmp/run_v2_eval.py`` against the live retriever.
2. Pick a query whose current ``actual_asset_ids`` is "what we want".
3. Add the query + expected + actual to ``GOLDEN_QUERIES``.
4. Re-baseline ``MIN_HIT_RATE_PER_GROUP`` based on the new group
   hit counts.
"""

from __future__ import annotations

import pytest

from mm_asset_rag import evaluation_v2
from mm_asset_rag.evaluation_v2 import (
    V2Result,
    run_text_to_image_eval_v2,
    run_text_to_text_eval_v2,
)
from mm_asset_rag.schema import SearchHit

# 20 mock asset_ids spanning 5 groups: english papers, chinese
# papers, mixed (with hash variants), Picsum-style images, Caltech.
MOCK_FULL_IDS: set[str] = {
    # English arxiv papers (each has 1-2 hash variants to exercise
    # the ``_title_of`` cross-hash matcher).
    "Learning Transferable Visual Models From Natural Language Supervision_6ea9db01",
    "Learning Transferable Visual Models From Natural Language Supervision_79e328a2",
    "Bert B42C52E2_5e8f0e8e",
    "Bert_ec793c5d",
    "Alexnet_0c1c2b23",
    "Alexnet Caaa534B_12f94731",
    "You Only Look Once_4582d878",
    "Detr_4582d878",
    "Attention Is All You Need_23e87012",
    "Attention Is All You Need 2A6E3761_86e3baff",
    "Ddpm_598d0928",
    "Ddpm D6E2716C_b7029c9a",
    # Chinese arxiv-style
    "所有深度用 AI 编程的朋友，这篇 Codex 全景指南值得存好，架构生态横评和最佳实践一次讲透_c1cf02d1",
    "所有深度用 AI 编程的朋友，这篇 Codex 全景指南值得存好，架构生态横评和最佳实践一次讲透_0363cb35",
    "CES 2026再绽光芒！ 联想两大“未来PC”背后的联宝智造力量_7df7f3f8",
    # Picsum-style images (positive-control for the text→image mocks)
    "Caltech Panda 01_7d47dc53",
    "Caltech Panda 01 3443A5D5_0698851d",
    # Negative-control distractors
    "Picsum 240 A3C86556_5747a9a9",
    "Picsum 291 9E581Fa7_e180b972",
    "Densenet_330fe977",
}

# Golden queries — captured from v4 baseline against the bundled
# corpus. ``actual_top_k`` is what the live retriever returned; the
# auto-eval asserts the same shape against the same search_fn stub so
# any regression in ``_match`` / ``_expand`` / ``_title_of`` is caught.
GOLDEN_QUERIES: dict[str, list[str]] = {
    # Bare title with no hash — _expand should resolve via prefix.
    "Bert": [
        "Bert B42C52E2_5e8f0e8e",
        "Bert_ec793c5d",
    ],
    # Different hash variant of the same content — _match should still
    # accept via _title_of stripping.
    "Learning Transferable Visual Models From Natural Language Supervision": [
        "Learning Transferable Visual Models From Natural Language Supervision_6ea9db01",
    ],
    # Cross-language: Chinese query, English expected.
    "CLIP 模型": [
        "Learning Transferable Visual Models From Natural Language Supervision_6ea9db01",
    ],
    # Chinese-only corpus (Codex).
    "Codex 全景指南": [
        "所有深度用 AI 编程的朋友，这篇 Codex 全景指南值得存好，架构生态横评和最佳实践一次讲透_c1cf02d1",
    ],
    # Negative — empty expected (designed to return nothing or be
    # over-recalled; we only assert the structure here, not the hit).
    "强化学习算法": [
        # Whatever the retriever returns is fine; we just don't want
        # an exception. The mock returns a Picsum + Caltech distractor.
        "Picsum 240 A3C86556_5747a9a9",
    ],
}


def _stub_search_fn(query: str, top_k: int) -> list[SearchHit]:
    """Return canned results for the golden queries, or empty otherwise."""
    actuals = GOLDEN_QUERIES.get(query, [])
    return [
        SearchHit(
            route="mock_text",
            score=1.0 / (1 + i),
            asset_id=aid,
            title=aid,
            source_type="pdf",
            source_path="",
            evidence="",
        )
        for i, aid in enumerate(actuals[:top_k])
    ]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point the eval at a temp ``$MM_ASSET_RAG_HOME`` so no real
    ``asset_index.jsonl`` is read. The tests inject ``full_ids``
    directly so the file is never opened.
    """
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(tmp_path))
    yield


# Per-group minimum hit_rate@5. Calibrated from the v4 mock corpus
# against the live retriever — adjust when intentionally retuning
# the retriever, never silently.
MIN_HIT_RATE = {
    "zh_on_en": 0.50,  # 1/2 golden queries should hit (CLIP / Bert)
    "en_on_en": 0.50,  # 1/2 (Bert, CLIP)
    "zh_on_zh": 0.50,  # 1/2 (Codex)
    # negative group is structural — we don't assert a hit_rate floor
    # because the design intent is "return nothing"; we only verify
    # the runner completes and respects the empty-expected contract.
}


def test_text_to_text_runner_uses_injected_search_fn() -> None:
    """The runner must honour the ``search_fn`` injection — no live Qdrant call."""
    results = run_text_to_text_eval_v2(top_k=5, search_fn=_stub_search_fn, full_ids=MOCK_FULL_IDS)
    # Run is non-empty (50+ v2 cases produce many per_query entries).
    assert len(results) > 0
    # Each result has a valid group label.
    assert {r.group for r in results} <= {
        "zh_on_en",
        "en_on_en",
        "zh_on_zh",
        "negative",
    }
    # No exception leaked from the runner.
    for r in results:
        assert isinstance(r, V2Result)


def test_golden_queries_hit_at_expected_rate() -> None:
    """Cross-check that the v2 case fixtures + matcher still resolve the
    bare ids we care about. If this fails, someone changed ``_match``
    / ``_expand`` / ``_title_of`` in a way that breaks hash-variant
    or bare-prefix resolution.
    """
    # Filter the runner output to only the queries that appear in
    # GOLDEN_QUERIES, then assert per-group hit_rate floors.
    results = run_text_to_text_eval_v2(top_k=5, search_fn=_stub_search_fn, full_ids=MOCK_FULL_IDS)
    by_query = {r.query: r for r in results}

    by_group_hits: dict[str, tuple[int, int]] = {}
    for q, _expected_assets in GOLDEN_QUERIES.items():
        r = by_query.get(q)
        if r is None:
            continue
        hit = 1 if r.hit else 0
        prev_h, prev_t = by_group_hits.get(r.group, (0, 0))
        by_group_hits[r.group] = (prev_h + hit, prev_t + 1)

    for group, (hits, total) in by_group_hits.items():
        rate = hits / total if total else 0.0
        floor = MIN_HIT_RATE.get(group)
        if floor is None:
            continue  # group intentionally not floored (e.g. negative)
        assert rate >= floor, (
            f"hit_rate regressed for group {group!r}: {rate:.3f} < {floor:.3f} "
            f"({hits}/{total}). If intentional, recalibrate MIN_HIT_RATE."
        )


def test_negative_queries_run_without_crash() -> None:
    """Negative samples should always finish (no exception) even if
    the mock returns distractor results — the runner is robust to
    over-recall.
    """
    results = run_text_to_text_eval_v2(top_k=5, search_fn=_stub_search_fn, full_ids=MOCK_FULL_IDS)
    negative = [r for r in results if r.group == "negative"]
    assert negative, "expected at least one negative case to run"
    # The runner must report hit=False when expected is empty,
    # regardless of how many actual_asset_ids the search_fn returned.
    for r in negative:
        assert r.expected_asset_ids == []
        assert r.hit is False


def test_text_to_image_runner_uses_injected_search_fn() -> None:
    """Same DI contract for the text→image runner."""

    def _t2i_stub(query: str, top_k: int) -> list[SearchHit]:
        return _stub_search_fn(query, top_k)

    results = run_text_to_image_eval_v2(top_k=5, search_fn=_t2i_stub, full_ids=MOCK_FULL_IDS)
    # Some V2_TEXT_TO_IMAGE cases expect Caltech Panda hits; the mock
    # returns the same canned results regardless of query. The runner
    # should still finish and report the right group.
    for r in results:
        assert r.group == "text_to_image"


def test_run_text_to_text_eval_v2_signature_includes_search_fn() -> None:
    """Sanity: the production runner exposes the DI hook. This guards
    against accidental signature changes that would break CI mocks.
    """
    import inspect

    sig = inspect.signature(run_text_to_text_eval_v2)
    assert "search_fn" in sig.parameters
    assert "full_ids" in sig.parameters


def test_evaluation_module_unchanged_construction() -> None:
    """Sanity: the module-level case lists still exist and have the
    expected sizes (guards against accidental reformatting that would
    shrink the eval set silently).
    """
    assert len(evaluation_v2.V2_ZH_ON_EN_PAPERS) >= 15
    assert len(evaluation_v2.V2_EN_ON_EN_PAPERS) >= 8
    assert len(evaluation_v2.V2_ZH_ON_ZH_CORPUS) >= 8
    assert len(evaluation_v2.V2_NEGATIVE_QUERIES) >= 6
    assert len(evaluation_v2.V2_TEXT_TO_IMAGE) >= 15
    assert len(evaluation_v2.V2_IMAGE_TO_IMAGE) >= 5
