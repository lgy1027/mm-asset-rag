from mm_asset_rag.core.observability import RuntimeMetrics


def test_runtime_metrics_aggregates_retrieval_and_refusal_events() -> None:
    metrics = RuntimeMetrics()
    metrics.record_retrieval(route="hybrid", elapsed_ms=42, candidates=8, returned=2, reason="none")
    metrics.record_retrieval(
        route="hybrid", elapsed_ms=180, candidates=0, returned=0, reason="no_candidates"
    )
    metrics.record_refusal(reason="low_lexical_coverage", candidates=1)

    snapshot = metrics.snapshot()

    assert snapshot["retrieval"]["hybrid"] == {
        "count": 2,
        "avg_elapsed_ms": 111,
        "max_elapsed_ms": 180,
        "candidates": 8,
        "returned": 2,
        "reasons": {"none": 1, "no_candidates": 1},
    }
    assert snapshot["refusals"] == {"low_lexical_coverage": {"count": 1, "candidates": 1}}


def test_metrics_endpoint_returns_runtime_snapshot(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import mm_asset_rag.api.api as api_mod
    from mm_asset_rag.api.api import app

    metrics = RuntimeMetrics()
    metrics.record_refusal(reason="no_candidates", candidates=0)
    monkeypatch.setattr(api_mod, "runtime_metrics", metrics)

    response = TestClient(app, base_url="http://127.0.0.1").get("/metrics")

    assert response.status_code == 200
    assert response.json()["refusals"] == {"no_candidates": {"count": 1, "candidates": 0}}
