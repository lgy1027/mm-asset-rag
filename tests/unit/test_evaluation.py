"""Tests for the qrels-only v1 evaluation entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.core.schema import SearchHit
from mm_asset_rag.eval.evaluation import (
    EvalResult,
    aggregate_retrieval_scenarios,
    load_cases,
    load_qrels,
    run_eval,
    write_eval_report,
)
from mm_asset_rag.query.search_service import SearchCommand, SearchMode


def _write_cases(path: Path, *, qrels: dict[str, dict[str, int]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "v1",
                "groups": {
                    "en": [
                        {"query_id": "q1", "query": "handbook"},
                        {"query_id": "q2", "query": "unknown"},
                    ]
                },
                "qrels": qrels,
            }
        ),
        encoding="utf-8",
    )
    return path


def _hit(
    document_id: str, *, asset_id: str = "physical-file", title: str = "handbook"
) -> SearchHit:
    return SearchHit(
        route="text",
        score=0.9,
        asset_id=asset_id,
        title=title,
        source_type="pdf",
        source_path="file.pdf",
        metadata={"document_id": document_id},
    )


def test_load_qrels_accepts_graded_document_relevance(tmp_path: Path) -> None:
    path = tmp_path / "qrels.json"
    path.write_text(
        json.dumps({"qrels": {"q1": {"handbook": 2, "ignore": 0}, "bad": "x"}}),
        encoding="utf-8",
    )

    assert load_qrels(path) == {"q1": {"handbook": 2}}


def test_load_cases_attaches_qrels_to_query_cases(tmp_path: Path) -> None:
    groups = load_cases(
        _write_cases(tmp_path / "cases.json", qrels={"q1": {"doc-handbook": 3}, "q2": {}})
    )

    assert groups == {
        "en": [
            {"query_id": "q1", "query": "handbook", "qrels": {"doc-handbook": 3}},
            {"query_id": "q2", "query": "unknown", "qrels": {}},
        ]
    }


def test_load_cases_rejects_legacy_expected_asset_ids(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "version": "v1",
                "groups": {"en": [{"query": "handbook", "expected_asset_ids": ["handbook"]}]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="qrels"):
        load_cases(path)


def test_load_cases_requires_explicit_qrel_for_every_query(tmp_path: Path) -> None:
    path = _write_cases(tmp_path / "cases.json", qrels={"q1": {"doc-handbook": 1}})

    with pytest.raises(ValueError, match="q2"):
        load_cases(path)


def test_bundled_v1_default_runs_with_qrels_only() -> None:
    results = run_eval(search_fn=lambda _command: [], collection="team", principal="alice")

    assert len(results) == 8
    assert len({result.query_id for result in results}) == 8
    assert all(result.qrels for result in results)
    assert all(result.actual_document_ids == [] for result in results)


def test_documents_100_qrels_are_loadable_and_explicit() -> None:
    path = Path(__file__).parents[2] / "examples" / "eval_cases_documents_100_v1.json"

    groups = load_cases(path)
    cases = [case for group in groups.values() for case in group]

    assert len(cases) == 30
    assert {case["query_id"] for case in cases} == set(load_qrels(path))
    assert all("query" in case and "qrels" in case for case in cases)
    assert len(groups["negative"]) == 5
    assert all(not case["qrels"] for case in groups["negative"])


def test_api_v1_default_uses_bundled_qrels(tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from mm_asset_rag.api.api import app

    monkeypatch.setattr(
        "mm_asset_rag.eval.evaluation.get_search_service",
        lambda: SimpleNamespace(execute=lambda _command: []),
    )

    response = TestClient(app, base_url="http://127.0.0.1").post(
        "/eval", json={"collection": "team", "principal": "alice"}
    )

    assert response.status_code == 200
    rows = response.json()["results"]
    assert len(rows) == 8
    assert all(set(row) >= {"query_id", "qrels", "actual_document_ids"} for row in rows)


def test_run_eval_uses_only_exact_metadata_document_id(tmp_path: Path) -> None:
    path = _write_cases(tmp_path / "cases.json", qrels={"q1": {"doc-handbook": 2}, "q2": {}})
    commands: list[SearchCommand] = []

    def search(command: SearchCommand) -> list[SearchHit]:
        commands.append(command)
        if command.query == "handbook":
            return [_hit("doc-other", asset_id="doc-handbook", title="doc-handbook")]
        return []

    results = run_eval(
        top_k=3,
        cases_path=path,
        search_fn=search,
        collection="team",
        principal="alice",
        metadata_filter={"department": "research"},
    )

    assert results[0] == EvalResult(
        query_id="q1",
        query="handbook",
        qrels={"doc-handbook": 2},
        actual_document_ids=["doc-other"],
        hit=False,
        rank=None,
        group="en",
    )
    assert commands[0] == SearchCommand(
        query="handbook",
        mode=SearchMode.HYBRID,
        top_k=3,
        collection="team",
        principal="alice",
        metadata_filter={"department": "research"},
    )


def test_run_eval_matches_exact_document_id_at_first_rank(tmp_path: Path) -> None:
    path = _write_cases(tmp_path / "cases.json", qrels={"q1": {"doc-handbook": 2}, "q2": {}})

    results = run_eval(
        cases_path=path,
        collection="team",
        principal="alice",
        search_fn=lambda command: (
            [_hit("miss"), _hit("doc-handbook")] if command.query == "handbook" else []
        ),
    )

    assert results[0].hit is True
    assert results[0].rank == 2


def test_scenarios_use_qrels_to_separate_positive_and_negative() -> None:
    results = [
        EvalResult("q1", "one", {"doc": 1}, ["doc"], True, 1, "en"),
        EvalResult("q2", "two", {}, ["noise"], False, None, "negative"),
    ]

    scenarios = aggregate_retrieval_scenarios(results)

    assert scenarios["positive"]["metrics"]["recall"][1] == 1.0
    assert scenarios["negative"]["false_retrieval_rate"] == 1.0


def test_write_eval_report_includes_required_metrics(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    results = [EvalResult("q1", "one", {"doc": 3}, ["doc"], True, 1, "en")]

    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert set(payload["metrics"]["all"]) == {"recall", "mrr", "map", "ndcg"}
    assert payload["metrics"]["all"]["recall"]["1"] == 1.0
    assert payload["per_query"][0]["qrels"] == {"doc": 3}
    assert payload["per_query"][0]["actual_document_ids"] == ["doc"]


def test_run_eval_scores_user_named_groups(tmp_path: Path) -> None:
    """User-supplied case files name their own groups; run_eval must score
    every group in the file, not a hardcoded menu (silently skipping a
    group leaves it unscored and the report shows an empty eval)."""
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "version": "v1",
                "groups": {
                    "口播音频": [{"query_id": "k1", "query": "三伏天"}],
                    "健康养生": [{"query_id": "j1", "query": "空腹"}],
                },
                "qrels": {"k1": {"三伏天养生": 1}, "j1": {"空腹力_36": 1}},
            }
        ),
        encoding="utf-8",
    )

    results = run_eval(
        top_k=5,
        cases_path=path,
        collection="team",
        principal="alice",
        search_fn=lambda _command: [_hit("三伏天养生")],
    )

    assert {r.query_id for r in results} == {"k1", "j1"}
    by_id = {r.query_id: r for r in results}
    assert by_id["k1"].group == "口播音频" and by_id["k1"].hit
    assert by_id["j1"].group == "健康养生" and not by_id["j1"].hit
