from mm_asset_rag.core.knowledge_models import (
    AccessPolicy,
    Asset,
    Chunk,
    Document,
    Source,
)
from mm_asset_rag.eval.image_eval import build_tag_cases, build_tag_qrels


def test_build_tag_qrels_uses_image_document_ids() -> None:
    document = Document("poster", "Poster", Source("source"), AccessPolicy("team", ("alice",)))
    chunk = Chunk.create(
        document=document,
        asset=Asset("a" * 64, "image", "images/poster.jpg"),
        ordinal=0,
        text="poster",
        source=document.source,
        access_policy=document.access_policy,
        metadata={"tags": ["活动"]},
    )
    assert build_tag_qrels([chunk], {"q1": "活动", "q2": "历史"}) == {
        "q1": {"poster": 1},
        "q2": {},
    }


def test_build_tag_cases_is_valid_v2_shape() -> None:
    assert build_tag_cases([], {"q1": "活动"}) == {
        "version": "v2",
        "groups": {"text_to_image": [{"query_id": "q1", "query": "活动"}]},
        "qrels": {"q1": {}},
    }


def test_build_tag_cases_includes_unanswerable_image_queries() -> None:
    cases = build_tag_cases(
        [],
        {"q1": "活动"},
        negative_queries={"negative-1": "火星探测器表面照片"},
    )

    assert cases["groups"]["negative"] == [
        {"query_id": "negative-1", "query": "火星探测器表面照片"}
    ]
    assert cases["qrels"]["negative-1"] == {}
