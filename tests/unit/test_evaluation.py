"""Tests for mm_asset_rag.evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mm_asset_rag.evaluation import EVAL_CASES, run_eval, write_eval_report
from mm_asset_rag.schema import SearchHit


def _fake_hybrid(query: str, **kwargs):
    """Return a single hit whose asset_id derives from the query string."""
    if "retrieval augmented generation" in query:
        asset_id = "pdf_rag"
    elif "CLIP" in query:
        asset_id = "pdf_clip"
    elif "OCR" in query or "版面" in query:
        asset_id = "pdf_layoutlm"
    else:
        asset_id = "other"
    return [
        SearchHit(
            route="text",
            score=0.9,
            asset_id=asset_id,
            title=asset_id,
            source_type="pdf",
            source_path=f"{asset_id}.pdf",
        )
    ]


def test_eval_cases_count() -> None:
    assert len(EVAL_CASES) == 3


def test_run_eval_hits_expected_assets() -> None:
    with patch("mm_asset_rag.evaluation.hybrid_search", side_effect=_fake_hybrid):
        results = run_eval()
    assert len(results) == 3
    assert all(result.hit for result in results), [result for result in results if not result.hit]


def test_run_eval_misses_when_assets_wrong() -> None:
    with patch(
        "mm_asset_rag.evaluation.hybrid_search",
        return_value=[
            SearchHit(
                route="text",
                score=0.9,
                asset_id="unrelated",
                title="x",
                source_type="pdf",
                source_path="x.pdf",
            )
        ],
    ):
        results = run_eval()
    assert not any(result.hit for result in results)


def test_write_eval_report(tmp_path: Path) -> None:
    from mm_asset_rag.evaluation import EvalResult

    results = [EvalResult(query="q", expected_asset_ids=["a"], actual_asset_ids=["a"], hit=True)]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload[0]["query"] == "q"
