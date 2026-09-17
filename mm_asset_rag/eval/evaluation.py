"""Qrels-only v1 retrieval evaluation over exact logical document IDs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.metrics import aggregate_metrics
from ..core.observability import get_tracer
from ..core.paths import get_eval_report
from ..core.schema import SearchHit
from ..query.search_service import SearchCommand, SearchMode, get_search_service
from .evaluation_reporting import build_report
from .evaluation_v2 import (
    _document_ids,
    _first_relevant_rank,
    aggregate_retrieval_scenarios,
)
from .evaluation_v2 import (
    load_cases as _load_cases,
)
from .evaluation_v2 import (
    load_qrels as _load_qrels,
)


def load_cases(path: str | Path | None = None) -> dict[str, list[dict]]:
    return _load_cases(path, version="v1")


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    return _load_qrels(path)


@dataclass
class EvalResult:
    query_id: str
    query: str
    qrels: dict[str, int]
    actual_document_ids: list[str]
    hit: bool
    rank: int | None
    group: str


def run_eval(
    top_k: int = 5,
    *,
    collection: str,
    principal: str,
    metadata_filter: dict[str, object] | None = None,
    cases_path: str | Path | None = None,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
) -> list[EvalResult]:
    """Run all v1 text retrieval cases using exact document qrels."""
    search = search_fn or get_search_service().execute
    groups = load_cases(cases_path)
    results: list[EvalResult] = []
    for group in ("en", "zh", "zh_doc", "legacy", "negative"):
        for case in groups.get(group, ()):
            query = str(case["query"])
            # Wrap each case so retrieval scores land on the query's trace
            # (search.dispatch nests underneath as a child observation).
            with get_tracer().start_span(
                "eval.case",
                attributes={"query_id": str(case["query_id"]), "group": group},
                input=query,
            ) as span:
                hits = search(
                    SearchCommand(
                        query=query,
                        mode=SearchMode.HYBRID,
                        top_k=top_k,
                        collection=collection,
                        metadata_filter=metadata_filter,
                        principal=principal,
                    )
                )
                actual = _document_ids(hits)
                qrels = dict(case["qrels"])
                rank = _first_relevant_rank(actual, qrels)
                span.score(
                    name="retrieval_hit",
                    value=1.0 if rank is not None else 0.0,
                    comment=f"rank={rank}",
                )
                if group == "negative":
                    span.score(
                        name="negative_reject",
                        value=1.0 if rank is None else 0.0,
                        comment=f"rank={rank}",
                    )
            results.append(
                EvalResult(
                    query_id=str(case["query_id"]),
                    query=query,
                    qrels=qrels,
                    actual_document_ids=actual,
                    hit=rank is not None,
                    rank=rank,
                    group=group,
                )
            )
    return results


def _metric_rows(results: list[EvalResult]) -> list[dict[str, object]]:
    return [
        {"actual_document_ids": list(result.actual_document_ids), "qrels": dict(result.qrels)}
        for result in results
    ]


def write_eval_report(results: list[EvalResult], path=None) -> None:
    """Write qrels, exact document results, and required retrieval metrics."""
    target = path or get_eval_report()
    by_group: dict[str, list[EvalResult]] = {"all": list(results)}
    for result in results:
        by_group.setdefault(result.group, []).append(result)
    groups = {
        group: {
            "total": len(group_results),
            "hits": sum(result.hit for result in group_results),
            "hit_rate": sum(result.hit for result in group_results) / max(len(group_results), 1),
        }
        for group, group_results in by_group.items()
        if group != "all"
    }
    metrics = {
        group: aggregate_metrics(_metric_rows(group_results)) if group_results else {}
        for group, group_results in by_group.items()
    }
    payload = build_report(
        kind="retrieval",
        summary={
            "total": len(results),
            "hits": sum(result.hit for result in results),
            "hit_rate": sum(result.hit for result in results) / max(len(results), 1),
            "scenarios": aggregate_retrieval_scenarios(results),
        },
        groups=groups,
        metrics=metrics,
        per_query=[asdict(result) for result in results],
    )
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    from ..core.config import load_env

    load_env()
    evaluated = run_eval()
    write_eval_report(evaluated)
