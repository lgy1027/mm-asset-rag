"""Local, explainable sufficiency checks before generating an answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.schema import SearchHit
from ..core.settings import Settings


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str | None = None


def _terms(text: str) -> set[str]:
    lowered = text.casefold()
    words = set(re.findall(r"[a-z0-9][a-z0-9_.-]{1,}", lowered))
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        words.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return words


def _searchable_text(hit: SearchHit) -> str:
    meta = hit.metadata.get("metadata")
    extra = (
        " ".join(f"{key} {value}" for key, value in meta.items()) if isinstance(meta, dict) else ""
    )
    return f"{hit.title} {hit.evidence} {extra}"


def lexical_coverage(question: str, hits: list[SearchHit]) -> float:
    """Return the fraction of query terms supported by retrieved evidence."""
    query_terms = _terms(question)
    if not query_terms:
        return 0.0
    matched = set().union(*(_terms(_searchable_text(hit)) for hit in hits)) & query_terms
    return len(matched) / len(query_terms)


def _has_conflicting_claims(hits: list[SearchHit]) -> bool:
    values_by_claim: dict[str, set[str]] = {}
    for hit in hits:
        claims = hit.metadata.get("claims")
        if not isinstance(claims, dict):
            continue
        for key, value in claims.items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, (str, int, float))
            ):
                continue
            normalized = str(value).strip().casefold()
            if normalized:
                values_by_claim.setdefault(key, set()).add(normalized)
    return any(len(values) > 1 for values in values_by_claim.values())


def _missing_critical_terms(question: str, hits: list[SearchHit]) -> bool:
    """Require explicit numeric/date/version tokens from a query in evidence."""
    critical = set(re.findall(r"(?<![\w.])\d+(?:[.-]\d+)*(?![\w.])", question.casefold()))
    if not critical:
        return False
    evidence = " ".join(_searchable_text(hit).casefold() for hit in hits)
    return any(term not in evidence for term in critical)


def assess_answer_evidence(
    question: str, hits: list[SearchHit], settings: Settings
) -> EvidenceAssessment:
    """Assess evidence without relying on normalized final retrieval scores."""
    if not hits:
        return EvidenceAssessment(False, "no_evidence")
    usable = [hit for hit in hits if hit.evidence.strip()]
    if not usable:
        return EvidenceAssessment(False, "empty_evidence")
    text_hits = [hit for hit in usable if hit.source_type != "image"]
    if not text_hits:
        return EvidenceAssessment(False, "insufficient_candidates")
    if _has_conflicting_claims(text_hits):
        return EvidenceAssessment(False, "conflicting_evidence")
    if _missing_critical_terms(question, text_hits):
        return EvidenceAssessment(False, "missing_critical_terms")
    rerank_scores = [
        float(hit.metadata["rerank_score"]) for hit in text_hits if "rerank_score" in hit.metadata
    ]
    if (
        settings.answer_min_rerank_score > 0.0
        and rerank_scores
        and max(rerank_scores) >= settings.answer_min_rerank_score
    ):
        return EvidenceAssessment(True)
    if lexical_coverage(question, text_hits) >= settings.answer_min_lexical_coverage:
        return EvidenceAssessment(True)
    return EvidenceAssessment(False, "weak_lexical_coverage")
