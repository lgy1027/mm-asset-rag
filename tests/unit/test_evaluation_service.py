from dataclasses import dataclass

from mm_asset_rag.evaluation_service import EvaluationCommand, EvaluationService


@dataclass
class _Result:
    query_id: str = "q1"


def test_service_runs_v2_and_writes_one_group_report() -> None:
    calls: dict[str, object] = {}

    def run_v2(**kwargs):
        calls["run"] = kwargs
        return [_Result()]

    def write_v2(groups):
        calls["groups"] = groups

    service = EvaluationService(run_v2=run_v2, write_v2=write_v2)
    response = service.execute(EvaluationCommand(collection="team", principal="alice", v2=True))

    assert response == {"kind": "retrieval", "version": "v2", "results": [{"query_id": "q1"}]}
    assert calls["groups"] == {"text_to_text": [_Result()]}
