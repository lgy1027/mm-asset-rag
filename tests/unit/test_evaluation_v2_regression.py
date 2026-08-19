"""Auto-eval regression test for ``evaluation_v2``.

This is the CI gate: when the retriever changes, this test runs in
under a second against a fixed mock corpus and asserts that hit_rate
does not regress below a recorded floor. The full v2 eval (with a
real Qdrant collection + ollama embedding) is a manual path against a
real corpus; this test exists to catch the obvious "someone changed
``_match`` and now nothing hits" class of bug before it lands.

How it works
============

* A small fixed corpus (``MOCK_FULL_IDS``) is passed via the
  ``full_ids`` kwarg ``run_text_to_text_eval_v2`` accepts. No
  ``asset_index.jsonl`` is read.
* A canned ``search_fn`` returns deterministic top-k results for the
  "golden" queries drawn from the bundled default v2 case set.
* Per-group hit_rate@5 is asserted against a minimum threshold. If
  someone breaks ``_match`` / ``_expand`` / ``_title_of`` such that
  these golden queries stop hitting, the test fails.

Adding a new query
==================

1. Add a case to ``mm_asset_rag/eval_data/v2_cases.json``.
2. Add the query + the canned ``actual_top_k`` to ``GOLDEN_QUERIES``
   and an id it should resolve to ``MOCK_FULL_IDS``.
3. Re-baseline ``MIN_HIT_RATE`` based on the new group hit counts.
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

# Mock asset_ids spanning the bundled default v2 case set (zh_on_en +
# en_on_en + negative) plus a distractor. Each expected bare title has a
# matching ``_<hash>`` variant so ``_expand`` resolves it.
MOCK_FULL_IDS: set[str] = {
    "Learning Transferable Visual Models From Natural Language Supervision_6ea9db01",
    "Bert_ec793c5d",
    "Retrieval Augmented Generation_caaa534b",
    "Resnet_0c1c2b23",
    "Alexnet_0c1c2b23",
    "You Only Look Once_4582d878",
    "Gan_caaa534b",
    # Negative-control distractor.
    "Picsum 240 A3C86556_5747a9a9",
}

# Golden queries — drawn from the bundled default v2 case set.
# ``actual_top_k`` is the canned list the stub returns; the auto-eval
# asserts ``_match`` / ``_expand`` / ``_title_of`` still resolve them.
GOLDEN_QUERIES: dict[str, list[str]] = {
    # zh_on_en — Chinese query, English paper expected.
    "CLIP 模型": [
        "Learning Transferable Visual Models From Natural Language Supervision_6ea9db01",
    ],
    "BERT 预训练双向 transformer": ["Bert_ec793c5d"],
    "RAG 检索增强生成": ["Retrieval Augmented Generation_caaa534b"],
    # en_on_en — English query, English paper.
    "image classification deep learning": ["Alexnet_0c1c2b23"],
    "real-time object detection": ["You Only Look Once_4582d878"],
    # Negative — empty expected; the mock returns a distractor. We only
    # assert the runner completes and respects the empty-expected
    # contract (hit=False), not a hit_rate floor.
    "强化学习算法 PPO DQN": ["Picsum 240 A3C86556_5747a9a9"],
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


# Per-group minimum hit_rate@5. Calibrated against the golden queries
# above — adjust when intentionally retuning the retriever or the case
# set, never silently.
MIN_HIT_RATE = {
    "zh_on_en": 1.0,  # 3/3 golden (CLIP / BERT / RAG)
    "en_on_en": 1.0,  # 2/2 (Alexnet, YOLO)
    # negative group is structural — not floored: the design intent is
    # "return nothing"; we only verify the runner completes and respects
    # the empty-expected contract.
}


def test_text_to_text_runner_uses_injected_search_fn() -> None:
    """The runner must honour the ``search_fn`` injection — no live Qdrant call."""
    results = run_text_to_text_eval_v2(top_k=5, search_fn=_stub_search_fn, full_ids=MOCK_FULL_IDS)
    # Run is non-empty (the bundled default v2 set has 9 cases).
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


def test_text_to_image_runner_absent_group_returns_empty() -> None:
    """The bundled default v2 case file has no ``text_to_image`` group
    (image routes live in the opt-in chapter11 file). The runner must
    return ``[]`` rather than crash when the group is absent — this is
    the contract ``mmrag eval --v2`` relies on by default.
    """

    def _t2i_stub(query: str, top_k: int) -> list[SearchHit]:
        return _stub_search_fn(query, top_k)

    results = run_text_to_image_eval_v2(top_k=5, search_fn=_t2i_stub, full_ids=MOCK_FULL_IDS)
    assert results == []


def test_run_text_to_text_eval_v2_signature_includes_search_fn() -> None:
    """Sanity: the production runner exposes the DI hook. This guards
    against accidental signature changes that would break CI mocks.
    """
    import inspect

    sig = inspect.signature(run_text_to_text_eval_v2)
    assert "search_fn" in sig.parameters
    assert "full_ids" in sig.parameters
    assert "cases_path" in sig.parameters


def test_default_case_set_has_expected_groups() -> None:
    """Sanity: the bundled default v2 case set has the expected groups
    and sizes (guards against accidental reformatting that would shrink
    the eval set silently).
    """
    groups = evaluation_v2.load_cases(version="v2")
    assert len(groups["zh_on_en"]) == 4
    assert len(groups["en_on_en"]) == 3
    assert len(groups["negative"]) == 2
    # Image groups belong to the opt-in chapter11 file, not the default.
    assert groups.get("text_to_image", []) == []
    assert groups.get("image_to_image", []) == []
