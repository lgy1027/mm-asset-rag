"""Tests for deterministic answer evidence assessment."""

from __future__ import annotations

from mm_asset_rag.evidence_policy import assess_answer_evidence
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.settings import Settings


def _hit(*, evidence: str, score: float = 1.0, metadata: dict | None = None) -> SearchHit:
    return SearchHit(
        route="text",
        score=score,
        asset_id="a",
        title="Docker guide",
        source_type="pdf",
        source_path="a.pdf",
        evidence=evidence,
        metadata=metadata or {},
    )


def test_rejects_normalized_top_score_without_query_coverage() -> None:
    result = assess_answer_evidence(
        "美国联邦基金利率", [_hit(evidence="Docker Compose 启动服务", score=1.0)], Settings()
    )
    assert result.sufficient is False
    assert result.reason == "weak_lexical_coverage"


def test_accepts_relevant_raw_rerank_hit() -> None:
    result = assess_answer_evidence(
        "如何启动 Docker Compose",
        [_hit(evidence="Docker Compose 启动服务", metadata={"rerank_score": 8.0})],
        Settings(),
    )
    assert result.sufficient is True
