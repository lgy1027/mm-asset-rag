"""Unit + regression tests for ``mm_asset_rag.answer_evaluation``.

Mirrors :mod:`tests.unit.test_evaluation_v2_regression` style:
``MOCK_*`` fixtures + ``_stub_*_fn`` DI hooks + per-scorer floor assertions +
``inspect.signature`` guard. The runner takes DI hooks (``search_fn`` /
``answer_fn`` / ``judge_fn`` / ``full_ids`` / ``max_judge_cases``) so every
layer - retrieval, answer generation, judge - is mocked out: no live Qdrant,
no live LLM, no live corpus required.

Layout
------
1. Signature + dataclass shape
2. Offline graceful degradation (no LLM -> coverage + citation run, faithfulness skipped)
3. Per-scorer (coverage / citation / faithfulness) tests
4. Back-compat (v1 / v2 case files without new fields still load)
5. Report writer + CLI + API surface
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from mm_asset_rag.answer import answer_evaluation as ae
from mm_asset_rag.answer.answer_evaluation import (
    AnswerEvalResult,
    _citation_metrics,
    _coverage,
    _judge_one,
    _normalize_for_coverage,
    _parse_faithfulness,
    run_answer_eval,
    write_answer_eval_report,
)
from mm_asset_rag.core.schema import SearchHit
from mm_asset_rag.query.search_service import SearchCommand, SearchMode

# ─── Mocks ────────────────────────────────────────────────────────────────


HITS: list[SearchHit] = [
    SearchHit(
        route="text",
        score=0.9,
        asset_id="Retrieval Augmented Generation_caaa534b",
        title="Retrieval Augmented Generation",
        source_type="pdf",
        source_path="rag.pdf",
        evidence="RAG combines a retriever with a generator to ground answers in context.",
        metadata={"page": 1},
    ),
    SearchHit(
        route="text",
        score=0.8,
        asset_id="Resnet_0c1c2b23",
        title="Resnet",
        source_type="pdf",
        source_path="resnet.pdf",
        evidence="Deep residual networks use skip connections to mitigate vanishing gradients.",
        metadata={"page": 3},
    ),
    SearchHit(
        route="text",
        score=0.7,
        asset_id="Gan_caaa534b",
        title="Gan",
        source_type="pdf",
        source_path="gan.pdf",
        evidence="A generator network competes with a discriminator in a minimax game.",
        metadata={"page": 5},
    ),
]


MOCK_FULL_IDS: set[str] = {h.asset_id for h in HITS}


def _stub_search_fn(command: SearchCommand) -> list[SearchHit]:
    return HITS[: command.top_k]


def _stub_answer_fn(query: str, hits: list[SearchHit]) -> dict:
    return {
        "question": query,
        "answer": "stub answer text",
        "sources": [{"asset_id": h.asset_id, "title": h.title} for h in hits],
    }


def _stub_fallback_answer_fn(query: str, hits: list[SearchHit]) -> dict:
    """Mirrors the real ``fallback_answer`` shape (sets ``_fallback=True``)
    so the CLI test exercises the "all fallback" branch in the report
    and the user-facing warning gets printed."""
    return {
        "question": query,
        "answer": "当前未配置 LLM stub answer",
        "sources": [{"asset_id": h.asset_id, "title": h.title} for h in hits],
        "_fallback": True,
    }


def _stub_judge_fn(question: str, hits: list[SearchHit], answer: str) -> float:
    return 0.9


# ─── 1. Signature + dataclass shape ───────────────────────────────────────


def test_signature_includes_di_hooks() -> None:
    """``run_answer_eval`` exposes the DI kwargs used by the test suite below."""
    sig = inspect.signature(run_answer_eval)
    for name in (
        "cases_path",
        "top_k",
        "search_fn",
        "answer_fn",
        "judge_fn",
        "full_ids",
        "max_judge_cases",
    ):
        assert name in sig.parameters, f"missing DI kwarg {name!r}"


def test_answer_eval_result_carries_all_score_fields() -> None:
    """Dataclass fields include all per-case scorers + provenance."""
    fields = {f.name for f in AnswerEvalResult.__dataclass_fields__.values()}
    for name in (
        "query",
        "expected_asset_ids",
        "expected_answer_assets",
        "actual_asset_ids",
        "answer_text",
        "answer_source",
        "coverage",
        "citation_precision",
        "citation_recall",
        "citation_present",
        "faithfulness",
        "faithfulness_skipped",
        "faithfulness_error",
        "group",
    ):
        assert name in fields, f"AnswerEvalResult missing {name!r}"


def test_documents_100_answer_cases_are_loadable_and_scored() -> None:
    path = Path(__file__).parents[2] / "examples" / "eval_cases_documents_100_answer_v1.json"

    groups, version = ae._load_cases(path)
    cases = [case for group in groups.values() for case in group]

    assert version == "answer_v1"
    assert len(cases) == 10
    assert all(case["expected_asset_ids"] for case in cases)
    assert all(case["expected_answer_assets"] for case in cases)
    assert all(case["expected_answer_keywords"] for case in cases)


# ─── 2. Offline graceful degradation ─────────────────────────────────────


def test_offline_no_llm_runs_gracefully(tmp_home, monkeypatch) -> None:
    """Without LLM credentials, coverage + citation still run; faithfulness
    is ``skipped=True``; ``answer_source="fallback"`` on every row.

    Uses the bundled default ``answer_v1_cases.json`` (4 EN + 4 ZH cases).
    """
    cases_path = (
        Path(__file__).resolve().parents[2]
        / "mm_asset_rag"
        / "eval"
        / "eval_data"
        / "answer_v1_cases.json"
    )
    # Force llm_creds -> (None, None, None) so fallback_answer path triggers.
    monkeypatch.setattr(
        "mm_asset_rag.answer.answer_evaluation.get_settings",
        lambda: _FakeSettings(llm_creds=(None, None, None)),
    )

    results = run_answer_eval(
        cases_path=str(cases_path),
        search_fn=_stub_search_fn,
        full_ids=MOCK_FULL_IDS,
        collection="team",
        principal="alice",
    )

    assert results, "no cases produced"
    for r in results:
        assert r.faithfulness_skipped, r
        assert r.faithfulness is None
        assert r.faithfulness_error, "error message should be populated"
        assert r.answer_source == "fallback"
        # coverage / citation still computed even when LLM is absent
        assert 0.0 <= r.coverage <= 1.0
        assert 0.0 <= r.citation_precision <= 1.0
        assert 0.0 <= r.citation_recall <= 1.0


def test_answer_eval_defaults_to_search_service(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = Mock()
    backend.execute.return_value = HITS
    monkeypatch.setattr(ae, "get_search_service", lambda: backend, raising=False)
    monkeypatch.setattr(
        "mm_asset_rag.query.retrieval.hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy retrieval ran")),
    )

    run_answer_eval(
        search_fn=None,
        collection="team",
        principal="alice",
        answer_fn=_stub_answer_fn,
        judge_fn=_stub_judge_fn,
        full_ids=MOCK_FULL_IDS,
    )

    assert backend.execute.call_args_list[0].args[0] == SearchCommand(
        query="retrieval augmented generation RAG",
        mode=SearchMode.HYBRID,
        top_k=5,
        collection="team",
        principal="alice",
    )


# ─── 3. Per-scorer tests ──────────────────────────────────────────────────


# Coverage


def test_coverage_basic_match() -> None:
    assert _coverage("PyTorch is great", ["pytorch", "great"]) == 1.0
    assert _coverage("PyTorch is great", ["pytorch", "tensorflow"]) == 0.5


def test_coverage_chinese_mixed_with_nfc_and_whitespace() -> None:
    """Normalization: NFC + casefold + whitespace collapse + ZW strip.

    Note the keyword keeps the same intra-CJK whitespace as the answer so
    the substring check passes after whitespace collapse; cross-CJK space
    stripping isn't part of v0's normalization (would require jieba or a
    segmentation pass — too heavy for an eval helper).
    """
    assert _coverage("双碳 目标 与 pytorch", ["双碳 目标", "PyTorch"]) == 1.0


def test_coverage_zero_width_chars() -> None:
    assert _coverage("看看双碳目标", ["双碳目标"]) == 1.0


def test_coverage_missing_keywords() -> None:
    assert _coverage("only one match", ["alpha", "beta", "gamma"]) == 0.0


def test_coverage_empty_keywords() -> None:
    """Empty keywords -> 0.0 (not division-by-zero)."""
    assert _coverage("anything", []) == 0.0


def test_normalize_for_coverage_strips_ideographic_space() -> None:
    """U+3000 (ideographic space) is stripped outright, not collapsed to ASCII."""
    s = "abc　def"  # contains U+3000
    out = _normalize_for_coverage(s)
    assert "　" not in out
    # U+3000 was stripped, not converted; the latin letters are now adjacent.
    assert out == "abcdef"


def test_normalize_for_coverage_collapses_whitespace_runs() -> None:
    """Runs of ASCII whitespace collapse to a single space."""
    s = "foo   bar\tbaz"
    assert _normalize_for_coverage(s) == "foo bar baz"


# Citation


def test_citation_precision_recall_with_markers() -> None:
    answer = "见 [1] 和 [2]。"
    expected = [HITS[0].asset_id, HITS[1].asset_id]
    prec, rec, present = _citation_metrics(answer, HITS, expected, MOCK_FULL_IDS)
    assert present is True
    assert prec == 1.0
    assert rec == 1.0


def test_citation_no_markers_in_answer() -> None:
    answer = "没有任何脚注。"
    prec, rec, present = _citation_metrics(answer, HITS, [HITS[0].asset_id], MOCK_FULL_IDS)
    assert present is False
    assert prec == 0.0
    assert rec == 0.0


def test_citation_partial_overlap() -> None:
    """Only one of two expected assets cited -> precision 1.0 (correct cite),
    recall 0.5 (missed one)."""
    answer = "见 [1]。"
    expected = [HITS[0].asset_id, HITS[1].asset_id]
    prec, rec, present = _citation_metrics(answer, HITS, expected, MOCK_FULL_IDS)
    assert present is True
    assert prec == 1.0
    assert rec == 0.5


def test_citation_wrong_rank_silently_dropped() -> None:
    """``[99]`` -> ``hits[98]`` doesn't exist -> ignored, ``present=False``."""
    answer = "见 [99]。"
    prec, rec, present = _citation_metrics(answer, HITS, [HITS[0].asset_id], MOCK_FULL_IDS)
    assert present is False
    assert prec == 0.0
    assert rec == 0.0


def test_citation_falls_back_to_expected_asset_ids() -> None:
    """Direct call with only ``expected_asset_ids`` (no separate
    ``expected_answer_assets``) works - the runner does the fallback in
    ``run_answer_eval``. Here we verify the metrics function itself accepts
    a single expected list and reports correctly."""
    answer = "见 [1]。"
    expected = [HITS[0].asset_id]
    prec, rec, present = _citation_metrics(answer, HITS, expected, MOCK_FULL_IDS)
    assert present is True
    assert prec == 1.0
    assert rec == 1.0


def test_citation_uses_normalised_ids() -> None:
    """Bare expected id ``"Resnet"`` matches actual ``"Resnet_<hash>"`` via
    ``_normalize_id`` slug normalisation + bidirectional substring."""
    answer = "见 [2]。"
    expected = ["Resnet"]  # bare, no hash
    prec, rec, present = _citation_metrics(answer, HITS, expected, MOCK_FULL_IDS)
    assert present is True
    assert prec == 1.0
    assert rec == 1.0


# Faithfulness


def test_faithfulness_skipped_when_no_llm(monkeypatch) -> None:
    """``llm_creds`` empty -> judge raises ``_JudgeUnavailable`` -> row records skip."""
    monkeypatch.setattr(
        "mm_asset_rag.answer.answer_evaluation.get_settings",
        lambda: _FakeSettings(llm_creds=(None, None, None)),
    )
    score, skipped, err = _judge_one("q", HITS, "a", ae.faithfulness_judge, 0, None)
    assert skipped is True
    assert score is None
    assert err


def test_faithfulness_runs_with_stubbed_judge() -> None:
    _score, skipped, err = _judge_one("q", HITS, "a", _stub_judge_fn, 0, None)
    assert skipped is False
    assert _score == 0.9
    assert err is None


def test_faithfulness_timeout_records_skip() -> None:
    """Judge raises ``requests.Timeout`` -> row records skip, runner continues."""

    def _timeout_judge(q: str, h: list[SearchHit], a: str) -> float:
        raise requests.Timeout("judge timeout")

    _score, skipped, err = _judge_one("q", HITS, "a", _timeout_judge, 0, None)
    assert skipped is True
    assert "Timeout" in (err or "")


def test_faithfulness_capped_by_max_judge_cases() -> None:
    """``max_judge_cases=2`` -> first 2 calls score, 3rd is skipped."""
    counts: list[int] = []

    def _counting_judge(q: str, h: list[SearchHit], a: str) -> float:
        counts.append(1)
        return 0.5

    for i in range(5):
        _judge_one("q", HITS, "a", _counting_judge, i, max_judge_cases=2)
    # i=0 -> judge_count=0 < 2 -> runs, judge_count -> 1 (external counter)
    # i=1 -> judge_count=1 < 2 -> runs, judge_count -> 2
    # i=2..4 -> judge_count >= 2 -> skipped
    assert len(counts) == 2


def test_faithfulness_parses_json_and_bare_float() -> None:
    """Parser: JSON ``{"faithfulness": 0.7}`` -> 0.7; bare ``0.65`` -> 0.65;
    garbage -> ValueError."""
    assert _parse_faithfulness('{"faithfulness": 0.7}') == 0.7
    assert _parse_faithfulness("  0.65 ") == 0.65
    with pytest.raises(ValueError):
        _parse_faithfulness("not a number")


def test_faithfulness_clamps_out_of_range() -> None:
    """Scores outside [0, 1] are clamped (judge may overshoot)."""

    def _high(q: str, h: list[SearchHit], a: str) -> float:
        return 1.5

    def _low(q: str, h: list[SearchHit], a: str) -> float:
        return -0.3

    score_hi, _, _ = _judge_one("q", HITS, "a", _high, 0, None)
    score_lo, _, _ = _judge_one("q", HITS, "a", _low, 0, None)
    assert score_hi == 1.0
    assert score_lo == 0.0


# ─── 4. Full pipeline (every layer stubbed) ──────────────────────────────


def test_full_pipeline_with_mocked_full_stack(tmp_home) -> None:
    """search + answer + judge all stubbed -> all scorers populated,
    ``answer_source="llm"``."""
    cases_path = (
        Path(__file__).resolve().parents[2]
        / "mm_asset_rag"
        / "eval"
        / "eval_data"
        / "answer_v1_cases.json"
    )

    def _answer_with_citation(query: str, hits: list[SearchHit]) -> dict:
        return {
            "question": query,
            "answer": "见 [1] retrieval augmented generation combines retrieval and context.",
            "sources": [{"asset_id": hits[0].asset_id, "title": hits[0].title}],
        }

    results = run_answer_eval(
        cases_path=str(cases_path),
        collection="team",
        principal="alice",
        search_fn=_stub_search_fn,
        answer_fn=_answer_with_citation,
        judge_fn=_stub_judge_fn,
        full_ids=MOCK_FULL_IDS,
    )

    assert results
    for r in results:
        assert r.answer_source == "llm"
        assert r.faithfulness_skipped is False
        assert r.faithfulness == 0.9
        assert r.citation_present is True
        assert 0.0 <= r.coverage <= 1.0


def test_back_compat_v1_v2_cases_load_without_new_fields(tmp_home, monkeypatch) -> None:
    """Cases from ``v1_cases.json`` (no ``expected_answer_keywords`` /
    ``expected_answer_assets``) still load - coverage defaults to 0.0,
    citation falls back to ``expected_asset_ids``."""
    v1_path = (
        Path(__file__).resolve().parents[2]
        / "mm_asset_rag"
        / "eval"
        / "eval_data"
        / "v1_cases.json"
    )

    monkeypatch.setattr(
        "mm_asset_rag.answer.answer_evaluation.get_settings",
        lambda: _FakeSettings(llm_creds=(None, None, None)),
    )

    results = run_answer_eval(
        cases_path=str(v1_path),
        collection="team",
        principal="alice",
        search_fn=_stub_search_fn,
        answer_fn=_stub_answer_fn,
        full_ids=MOCK_FULL_IDS,
    )
    assert results
    for r in results:
        assert r.coverage == 0.0, "no expected_answer_keywords -> 0.0"
        assert r.expected_answer_assets == r.expected_asset_ids, (
            "expected_answer_assets falls back to expected_asset_ids"
        )


def test_image_route_cases_raise(tmp_home) -> None:
    """Cases with ``image_path`` or groups named ``text_to_image`` /
    ``image_to_image`` raise a clear ``ValueError``."""
    bad_path = tmp_home / "bad.json"
    bad_path.write_text(
        json.dumps(
            {
                "version": "answer_v1",
                "groups": {
                    "text_to_image": [
                        {"query": "x", "image_path": "/tmp/x.png", "expected_asset_ids": ["a"]}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="text→text only"):
        run_answer_eval(
            cases_path=str(bad_path),
            search_fn=_stub_search_fn,
            collection="team",
            principal="alice",
        )


# ─── 5. Report writer + CLI + API surface ────────────────────────────────


def test_report_writes_to_eval_report_answer_json(tmp_home) -> None:
    """Report lands at ``$MM_ASSET_RAG_HOME/eval_report_answer.json`` with the
    shared evaluation report envelope."""
    cases_path = (
        Path(__file__).resolve().parents[2]
        / "mm_asset_rag"
        / "eval"
        / "eval_data"
        / "answer_v1_cases.json"
    )
    results = run_answer_eval(
        cases_path=str(cases_path),
        collection="team",
        principal="alice",
        search_fn=_stub_search_fn,
        answer_fn=_stub_answer_fn,
        judge_fn=_stub_judge_fn,
        full_ids=MOCK_FULL_IDS,
    )
    write_answer_eval_report(results)
    out = tmp_home / "eval_report_answer.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "evaluation.v1"
    assert payload["kind"] == "answer_quality"
    assert "answer_sources" in payload["summary"]
    assert "groups" in payload
    assert "metrics" in payload
    assert "all" in payload["metrics"]


def test_cli_eval_answer_quality_flag(tmp_home, monkeypatch, capsys) -> None:
    """``command_eval --answer-quality`` runs end-to-end via the CLI and
    prints the [answer-quality] hint when no LLM is configured.

    Stubs the command-level search service and ``llm_answer`` at the runner
    module level
    so the CLI path doesn't need real embedding creds or a corpus - we
    only want to assert the CLI plumbing (flag routing, report write,
    friendly warning).
    """
    from mm_asset_rag.cli import command_eval

    monkeypatch.setattr(
        "mm_asset_rag.answer.answer_evaluation.get_search_service",
        lambda: type("StubSearchService", (), {"execute": staticmethod(_stub_search_fn)})(),
    )
    monkeypatch.setattr(
        "mm_asset_rag.answer.answer.llm_answer",
        _stub_fallback_answer_fn,
        raising=False,
    )
    monkeypatch.setattr(
        "mm_asset_rag.answer.answer_evaluation.get_settings",
        lambda: _FakeSettings(llm_creds=(None, None, None)),
    )

    args = argparse.Namespace(
        top_k=5,
        cases=None,
        answer_quality=True,
        v2=False,
        collection="team",
        principal="alice",
        metadata_filter=None,
    )
    command_eval(args)
    captured = capsys.readouterr()
    assert "no LLM creds configured" in captured.out
    assert (tmp_home / "eval_report_answer.json").exists()


def test_api_eval_request_rejects_v2_and_answer_quality_together() -> None:
    """Pydantic mutex validator raises ``ValidationError`` for v2 ∧ answer_quality."""
    from pydantic import ValidationError

    from mm_asset_rag.api.api import EvalRequest

    with pytest.raises(ValidationError):
        EvalRequest(
            v2=True,
            answer_quality=True,
            collection="team",
            principal="alice",
        )
    # Either flag alone is fine.
    EvalRequest(v2=True, collection="team", principal="alice")
    EvalRequest(answer_quality=True, collection="team", principal="alice")
    EvalRequest(collection="team", principal="alice")


# ─── Helpers ──────────────────────────────────────────────────────────────


class _FakeSettings:
    """Minimal Settings stand-in for tests that read ``llm_creds`` /
    ``eval_judge_max_cases``."""

    def __init__(self, *, llm_creds: tuple, eval_judge_max_cases: int | None = None):
        self._llm_creds = llm_creds
        self.eval_judge_max_cases = eval_judge_max_cases

    @property
    def llm_creds(self) -> tuple:
        return self._llm_creds
