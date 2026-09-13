"""Tests for the qrels-only v2 evaluation runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from mm_asset_rag.evaluation_v2 import (
    V2Result,
    load_cases,
    run_eval_v2,
    run_image_to_image_eval_v2,
    run_text_to_image_eval_v2,
    run_text_to_text_eval_v2,
    write_eval_report_v2,
)
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.search_service import SearchCommand, SearchMode


def _hit(document_id: str, *, asset_id: str = "physical", title: str = "handbook") -> SearchHit:
    return SearchHit(
        route="text",
        score=1.0,
        asset_id=asset_id,
        title=title,
        source_type="pdf",
        source_path="doc.pdf",
        metadata={"document_id": document_id},
    )


def _write_cases(path: Path, *, image_path: Path | None = None) -> Path:
    groups: dict[str, list[dict[str, str]]] = {
        "zh_on_en": [{"query_id": "q1", "query": "handbook"}],
        "negative": [{"query_id": "q2", "query": "unknown"}],
        "text_to_image": [{"query_id": "q3", "query": "diagram"}],
    }
    qrels: dict[str, dict[str, int]] = {
        "q1": {"doc-handbook": 3},
        "q2": {},
        "q3": {"doc-diagram": 2},
    }
    if image_path is not None:
        groups["image_to_image"] = [{"query_id": "q4", "image_path": str(image_path)}]
        qrels["q4"] = {"doc-image": 1}
    path.write_text(
        json.dumps({"version": "v2", "groups": groups, "qrels": qrels}),
        encoding="utf-8",
    )
    return path


def test_load_cases_preserves_graded_qrels(tmp_path: Path) -> None:
    groups = load_cases(_write_cases(tmp_path / "cases.json"), version="v2")

    assert groups["zh_on_en"][0]["qrels"] == {"doc-handbook": 3}
    assert groups["negative"][0]["qrels"] == {}


def test_bundled_v2_default_runs_with_qrels_only() -> None:
    results = run_eval_v2(search_fn=lambda _command: [], collection="team", principal="alice")

    assert len(results) == 9
    assert len({result.query_id for result in results}) == 9
    assert sum(not result.qrels for result in results) == 2
    assert all(result.actual_document_ids == [] for result in results)


def test_api_v2_default_uses_bundled_qrels(monkeypatch) -> None:
    from types import SimpleNamespace

    from mm_asset_rag.api import app

    monkeypatch.setattr(
        "mm_asset_rag.evaluation_v2.get_search_service",
        lambda: SimpleNamespace(execute=lambda _command: []),
    )

    response = TestClient(app, base_url="http://127.0.0.1").post(
        "/eval", json={"v2": True, "collection": "team", "principal": "alice"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "v2"
    assert len(payload["results"]) == 9
    assert sum(not row["qrels"] for row in payload["results"]) == 2


def test_text_runner_uses_exact_document_qrels(tmp_path: Path) -> None:
    results = run_text_to_text_eval_v2(
        cases_path=_write_cases(tmp_path / "cases.json"),
        collection="team",
        principal="alice",
        search_fn=lambda command: (
            [_hit("doc-other", asset_id="doc-handbook", title="doc-handbook")]
            if command.query == "handbook"
            else []
        ),
    )

    assert results[0] == V2Result(
        query_id="q1",
        query="handbook",
        qrels={"doc-handbook": 3},
        actual_document_ids=["doc-other"],
        hit=False,
        rank=None,
        group="zh_on_en",
    )
    assert results[1].qrels == {}


def test_run_eval_v2_wraps_text_to_text() -> None:
    with patch("mm_asset_rag.evaluation_v2.run_text_to_text_eval_v2", return_value=[]) as stub:
        out = run_eval_v2(
            top_k=7,
            cases_path="cases.json",
            collection="team",
            principal="alice",
        )

    stub.assert_called_once_with(
        top_k=7,
        cases_path="cases.json",
        collection="team",
        principal="alice",
        metadata_filter=None,
    )
    assert out == []


def test_text_and_image_runners_pass_typed_commands(tmp_path: Path) -> None:
    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"fake image")
    cases_path = _write_cases(tmp_path / "cases.json", image_path=image_path)
    commands: list[SearchCommand] = []

    def search(command: SearchCommand) -> list[SearchHit]:
        commands.append(command)
        if command.mode is SearchMode.TEXT_TO_IMAGE:
            return [_hit("doc-diagram")]
        if command.mode is SearchMode.IMAGE_TO_IMAGE:
            return [_hit("doc-image")]
        return []

    text_results = run_text_to_image_eval_v2(
        search_fn=search, cases_path=cases_path, collection="team", principal="alice"
    )
    image_results = run_image_to_image_eval_v2(
        search_fn=search, cases_path=cases_path, collection="team", principal="alice"
    )

    assert [command.mode for command in commands] == [
        SearchMode.TEXT_TO_IMAGE,
        SearchMode.IMAGE_TO_IMAGE,
    ]
    assert commands[1].image_path == image_path
    assert text_results[0].actual_document_ids == ["doc-diagram"]
    assert text_results[0].hit is True
    assert image_results[0].actual_document_ids == ["doc-image"]
    assert image_results[0].hit is True


def test_missing_image_case_keeps_its_qrels(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    results = run_image_to_image_eval_v2(
        cases_path=_write_cases(tmp_path / "cases.json", image_path=missing),
        collection="team",
        principal="alice",
    )

    assert results == [
        V2Result(
            query_id="q4",
            query="missing.png",
            qrels={"doc-image": 1},
            actual_document_ids=[],
            hit=False,
            rank=None,
            group="image_to_image",
        )
    ]


def test_write_v2_report_includes_required_metrics(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    result = V2Result("q1", "handbook", {"doc": 2}, ["doc"], True, 1, "zh_on_en")

    write_eval_report_v2({"text_to_text": [result]}, path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "evaluation.v1"
    assert payload["kind"] == "retrieval"
    assert set(payload["groups"]["text_to_text"]["metrics"]) == {
        "recall",
        "mrr",
        "map",
        "ndcg",
    }
    assert payload["per_query"][0]["qrels"] == {"doc": 2}


@dataclass
class _FakeV2Result:
    query_id: str
    query: str
    qrels: dict[str, int]
    actual_document_ids: list[str]
    hit: bool
    rank: int | None
    group: str


def test_eval_endpoint_v2_returns_document_qrels_shape() -> None:
    from mm_asset_rag.api import app

    fake = [_FakeV2Result("q1", "handbook", {"doc": 2}, ["doc"], True, 1, "zh_on_en")]
    with patch("mm_asset_rag.evaluation_v2.run_eval_v2", return_value=fake):
        response = TestClient(app, base_url="http://127.0.0.1").post(
            "/eval",
            json={
                "v2": True,
                "top_k": 5,
                "collection": "team",
                "principal": "alice",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "kind": "retrieval",
        "version": "v2",
        "results": [
            {
                "query_id": "q1",
                "query": "handbook",
                "qrels": {"doc": 2},
                "actual_document_ids": ["doc"],
                "hit": True,
                "rank": 1,
                "group": "zh_on_en",
            }
        ],
    }
