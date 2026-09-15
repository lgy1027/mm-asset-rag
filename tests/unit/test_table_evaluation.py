"""Regression coverage for row-level table retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from mm_asset_rag.cli import build_parser
from mm_asset_rag.evaluation_v2 import run_eval_v2, write_eval_report_v2
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.table_evaluation import build_csv_cases


def _hit(*, evidence: str) -> SearchHit:
    return SearchHit(
        route="text",
        score=1.0,
        asset_id="physical-csv",
        title="citizen-qa",
        source_type="document",
        source_path="citizen-qa.csv",
        evidence=evidence,
        metadata={"document_id": "citizen-qa"},
    )


def _write_table_cases(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "v2",
                "groups": {
                    "table": [
                        {
                            "query_id": "csv-pension-proof",
                            "query": "养老保险缴费证明在哪里打印",
                            "evidence_contains": ["社保缴费证明查询打印"],
                        }
                    ],
                    "negative": [{"query_id": "csv-unrelated", "query": "火星探测器表面照片"}],
                },
                "qrels": {"csv-pension-proof": {"citizen-qa": 1}, "csv-unrelated": {}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_table_cases_score_top_evidence_and_report_the_metric(tmp_path: Path) -> None:
    cases_path = _write_table_cases(tmp_path / "table.json")

    results = run_eval_v2(
        cases_path=cases_path,
        collection="csv-eval",
        principal="evaluator",
        search_fn=lambda command: (
            [_hit(evidence="回答：社保缴费证明查询打印可在大厅办理")]
            if command.query.startswith("养老保险")
            else []
        ),
    )

    assert len(results) == 2
    assert results[0].group == "table"
    assert results[0].hit is True
    assert results[0].evidence_expected == ["社保缴费证明查询打印"]
    assert results[0].evidence_hit is True
    assert results[0].evidence_rank == 1

    report_path = tmp_path / "report.json"
    write_eval_report_v2({"table": results}, path=report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["groups"]["table"]["evidence"] == {
        "total": 1,
        "hit_count": 1,
        "hit_rate": 1.0,
    }


def test_build_csv_cases_creates_qrels_and_row_evidence(tmp_path: Path) -> None:
    source = tmp_path / "citizen-qa.csv"
    source.write_text(
        "问题,回答\n养老保险缴费证明在哪里打印,社保缴费证明查询打印\n,无效行\n",
        encoding="utf-8",
    )

    cases = build_csv_cases(
        source,
        document_id="citizen-qa",
        question_column="问题",
        evidence_column="回答",
    )

    assert cases == {
        "version": "v2",
        "groups": {
            "table": [
                {
                    "query_id": "citizen-qa-row-0001",
                    "query": "养老保险缴费证明在哪里打印",
                    "evidence_contains": ["社保缴费证明查询打印"],
                }
            ]
        },
        "qrels": {"citizen-qa-row-0001": {"citizen-qa": 1}},
    }


def test_cli_make_table_cases_writes_an_eval_cases_file(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import mm_asset_rag.cli as cli_mod

    source = tmp_path / "citizen-qa.csv"
    source.write_text(
        "问题,回答\n养老保险缴费证明在哪里打印,社保缴费证明查询打印\n", encoding="utf-8"
    )
    eval_cases = tmp_path / "eval_cases"
    monkeypatch.setattr(cli_mod, "get_eval_cases_dir", lambda: eval_cases)

    args = build_parser().parse_args(
        [
            "make-table-cases",
            str(source),
            "--document-id",
            "citizen-qa",
            "--question-column",
            "问题",
            "--evidence-column",
            "回答",
            "--output",
            "citizen-qa.json",
        ]
    )
    args.func(args)

    target = eval_cases / "citizen-qa.json"
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["groups"]["table"][0]["query"] == (
        "养老保险缴费证明在哪里打印"
    )
    assert str(target) in capsys.readouterr().out


def test_cli_v2_writes_table_results_under_their_own_group(tmp_path: Path, monkeypatch) -> None:
    import mm_asset_rag.cli as cli_mod
    import mm_asset_rag.evaluation_v2 as evaluation_v2

    result = run_eval_v2(
        cases_path=_write_table_cases(tmp_path / "table.json"),
        collection="csv-eval",
        principal="evaluator",
        search_fn=lambda _command: [_hit(evidence="社保缴费证明查询打印")],
    )[0]
    written: dict[str, object] = {}
    monkeypatch.setattr(cli_mod, "_resolve_cli_cases_path", lambda _value: "table.json")
    monkeypatch.setattr(evaluation_v2, "run_eval_v2", lambda **_kwargs: [result])
    monkeypatch.setattr(
        evaluation_v2, "write_eval_report_v2", lambda groups: written.update(groups)
    )

    build_parser().parse_args(
        [
            "eval",
            "--v2",
            "--cases",
            "table.json",
            "--collection",
            "csv-eval",
            "--principal",
            "evaluator",
        ]
    ).func(
        build_parser().parse_args(
            [
                "eval",
                "--v2",
                "--cases",
                "table.json",
                "--collection",
                "csv-eval",
                "--principal",
                "evaluator",
            ]
        )
    )

    assert written == {"table": [result]}
