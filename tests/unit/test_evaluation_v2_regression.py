"""Offline regression gate for exact document-qrels evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.evaluation_v2 import run_text_to_text_eval_v2, write_eval_report_v2
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.search_service import SearchCommand


def _hit(document_id: str) -> SearchHit:
    return SearchHit(
        route="mock",
        score=1.0,
        asset_id=f"physical-{document_id}",
        title=document_id,
        source_type="pdf",
        source_path="",
        metadata={"document_id": document_id},
    )


def test_document_qrels_regression_metrics(tmp_path: Path) -> None:
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            {
                "version": "v2",
                "groups": {
                    "zh_on_en": [
                        {"query_id": "q1", "query": "first"},
                        {"query_id": "q2", "query": "second"},
                    ]
                },
                "qrels": {
                    "q1": {"doc-a": 3, "doc-b": 1},
                    "q2": {"doc-c": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    def search(command: SearchCommand) -> list[SearchHit]:
        if command.query == "first":
            return [_hit("doc-b"), _hit("doc-a")]
        return [_hit("miss"), _hit("doc-c")]

    results = run_text_to_text_eval_v2(
        cases_path=cases,
        search_fn=search,
        collection="team",
        principal="alice",
    )
    report = tmp_path / "report.json"
    write_eval_report_v2({"text_to_text": results}, path=report)
    metrics = json.loads(report.read_text(encoding="utf-8"))["groups"]["text_to_text"]["metrics"]

    assert metrics["recall"]["1"] == pytest.approx(0.25)
    assert metrics["recall"]["3"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["map"] == pytest.approx(0.75)
    assert metrics["ndcg"]["3"] < 1.0
