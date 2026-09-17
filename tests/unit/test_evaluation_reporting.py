from mm_asset_rag.eval.evaluation_reporting import build_report


def test_build_report_has_one_stable_envelope() -> None:
    report = build_report(
        kind="retrieval",
        summary={"total": 1},
        groups={"en": {"total": 1}},
        metrics={"all": {"mrr": 1.0}},
        per_query=[{"query_id": "q1"}],
    )

    assert report == {
        "schema_version": "evaluation.v1",
        "kind": "retrieval",
        "summary": {"total": 1},
        "groups": {"en": {"total": 1}},
        "metrics": {"all": {"mrr": 1.0}},
        "per_query": [{"query_id": "q1"}],
    }
