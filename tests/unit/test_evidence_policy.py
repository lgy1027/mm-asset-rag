"""Tests for deterministic answer evidence assessment."""

from __future__ import annotations

from mm_asset_rag.core.schema import SearchHit
from mm_asset_rag.core.settings import Settings
from mm_asset_rag.query.evidence_policy import assess_answer_evidence


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


def test_rerank_score_does_not_bypass_lexical_gate_when_threshold_disabled() -> None:
    result = assess_answer_evidence(
        "美国联邦基金利率",
        [_hit(evidence="Docker Compose 启动服务", metadata={"rerank_score": 0.1})],
        Settings(answer_min_rerank_score=0.0),
    )

    assert result.sufficient is False
    assert result.reason == "weak_lexical_coverage"


def test_evidence_with_conflicting_structured_claims_is_rejected() -> None:
    result = assess_answer_evidence(
        "what is the policy status",
        [
            _hit(evidence="policy is active", metadata={"claims": {"policy_status": "active"}}),
            _hit(evidence="policy is inactive", metadata={"claims": {"policy_status": "inactive"}}),
        ],
        Settings(answer_min_lexical_coverage=0.1),
    )

    assert result.sufficient is False
    assert result.reason == "conflicting_evidence"


def test_evidence_with_near_match_but_wrong_year_is_rejected() -> None:
    result = assess_answer_evidence(
        "2025 tax rate",
        [_hit(evidence="2024 tax rate is 10 percent")],
        Settings(answer_min_lexical_coverage=0.2),
    )

    assert result.sufficient is False
    assert result.reason == "missing_critical_terms"
