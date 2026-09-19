from dataclasses import dataclass

from mm_asset_rag.eval.evaluation_service import EvaluationCommand, EvaluationService


@dataclass
class _Result:
    query_id: str = "q1"


def test_service_runs_v2_and_writes_one_group_report() -> None:
    calls: dict[str, object] = {}

    def run_v2(**kwargs):
        calls["run"] = kwargs
        return [_Result()]

    def write_v2(groups, **kwargs):
        calls["groups"] = groups

    service = EvaluationService(run_v2=run_v2, write_v2=write_v2)
    response = service.execute(EvaluationCommand(collection="team", principal="alice", v2=True))

    assert response == {"kind": "retrieval", "version": "v2", "results": [{"query_id": "q1"}]}
    assert calls["groups"] == {"text_to_text": [_Result()]}


def test_service_runs_image_eval_and_groups_report() -> None:
    from mm_asset_rag.eval.evaluation_v2 import V2Result

    calls: dict[str, object] = {}
    rows = [
        V2Result("q1", "猫", {"cat": 1}, ["cat"], True, 1, "text_to_image_zh"),
        V2Result("q2", "cat-image", {"cat": 1}, ["cat"], True, 1, "image_to_image"),
    ]

    def run_image(**kwargs):
        calls["run"] = kwargs
        return rows

    def write_v2(groups, **kwargs):
        calls["groups"] = groups
        calls["write"] = kwargs

    service = EvaluationService(run_image=run_image, write_v2=write_v2)
    response = service.execute(
        EvaluationCommand(
            collection="team",
            principal="alice",
            image=True,
            cases_path="image-cases.json",
        )
    )

    assert response["kind"] == "retrieval"
    assert response["version"] == "v2"
    assert response["results"][0]["query_id"] == "q1"
    assert calls["run"]["cases_path"] == "image-cases.json"
    assert calls["groups"] == {
        "text_to_image_zh": [rows[0]],
        "image_to_image": [rows[1]],
    }
    assert calls["write"]["collection"] == "team"
    assert calls["write"]["run_context"]["retrieval_gate"] == "primitive_image_routes"
