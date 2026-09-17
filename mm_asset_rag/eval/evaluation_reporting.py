"""Stable report envelope shared by every evaluation runner."""

from __future__ import annotations


def build_report(
    *,
    kind: str,
    summary: dict[str, object],
    groups: dict[str, dict[str, object]],
    metrics: dict[str, object],
    per_query: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": "evaluation.v1",
        "kind": kind,
        "summary": summary,
        "groups": groups,
        "metrics": metrics,
        "per_query": per_query,
    }
