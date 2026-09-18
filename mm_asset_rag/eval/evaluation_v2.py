"""Qrels-only retrieval evaluation over exact logical document IDs.

Case files contain grouped queries plus one top-level graded-qrels mapping::

    {
      "version": "v2",
      "groups": {"en_on_en": [{"query_id": "q1", "query": "..."}]},
      "qrels": {"q1": {"document-a": 3, "document-b": 1}}
    }

``expected_asset_ids`` and filename/title matching are intentionally not
supported.  A search result is scoreable only when its metadata contains a
non-empty ``document_id``; relevance is exact string equality.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

from ..core.metrics import _is_relevant, aggregate_metrics
from ..core.paths import get_assets_dir, get_eval_report
from ..core.schema import SearchHit
from ..query.search_service import SearchCommand, SearchMode, get_search_service
from .evaluation_reporting import build_report


def _default_cases_path(version: str):
    return files("mm_asset_rag").joinpath("eval", "eval_data", f"{version}_cases.json")


def _parse_qrels(raw_qrels: object) -> dict[str, dict[str, int]]:
    """Validate and retain positive integer grades, including empty queries."""
    if not isinstance(raw_qrels, Mapping):
        return {}
    parsed: dict[str, dict[str, int]] = {}
    for query_id, labels in raw_qrels.items():
        if not isinstance(query_id, str) or not query_id or not isinstance(labels, Mapping):
            continue
        query_labels: dict[str, int] = {}
        for document_id, relevance in labels.items():
            if (
                isinstance(document_id, str)
                and document_id
                and isinstance(relevance, int)
                and not isinstance(relevance, bool)
                and relevance > 0
            ):
                query_labels[document_id] = relevance
        parsed[query_id] = query_labels
    return parsed


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load the top-level graded qrels mapping from a JSON case file."""
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return _parse_qrels(payload.get("qrels"))


def load_cases(path: str | Path | None = None, *, version: str) -> dict[str, list[dict]]:
    """Load and validate qrels-only grouped evaluation cases.

    Each case must have a unique evaluation ``query_id`` and that ID must
    appear explicitly in the top-level qrels mapping.  ``{}`` is a valid,
    intentional negative judgment; an absent qrel is an input error.
    """
    if path is None:
        env_path = None
        try:
            from ..core.settings import get_settings

            env_path = get_settings().eval_cases_path
        except Exception:  # pragma: no cover - settings infrastructure failure
            pass
        path = env_path

    if path is None:
        source = _default_cases_path(version)
        data = json.loads(source.read_text(encoding="utf-8"))
        src = str(source)
    else:
        source = Path(path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"eval cases file not found: {source}")
        data = json.loads(source.read_text(encoding="utf-8"))
        src = str(source)

    if not isinstance(data, Mapping):
        raise ValueError(f"eval cases file {src}: expected a top-level JSON object.")
    file_version = data.get("version")
    if file_version is not None and file_version != version:
        raise ValueError(
            f"eval cases file {src}: version mismatch (expected {version!r}, got {file_version!r})."
        )
    groups = data.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError(f"eval cases file {src}: expected a non-empty top-level 'groups' object.")
    raw_qrels = data.get("qrels")
    if not isinstance(raw_qrels, Mapping):
        raise ValueError(
            f"eval cases file {src}: qrels are required; expected "
            "{query_id: {document_id: relevance}} and not expected_asset_ids."
        )
    qrels = _parse_qrels(raw_qrels)

    loaded: dict[str, list[dict]] = {}
    seen_query_ids: set[str] = set()
    for group, cases in groups.items():
        if not isinstance(group, str) or not isinstance(cases, list):
            raise ValueError(f"eval cases file {src}: every group must map to a case list.")
        loaded_cases: list[dict] = []
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError(f"eval cases file {src}: every case must be an object.")
            if "expected_asset_ids" in case:
                raise ValueError(
                    f"eval cases file {src}: expected_asset_ids is obsolete; use graded qrels."
                )
            query_id = case.get("query_id")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"eval cases file {src}: every case requires a query_id.")
            if query_id in seen_query_ids:
                raise ValueError(f"eval cases file {src}: duplicate query_id {query_id!r}.")
            seen_query_ids.add(query_id)
            if query_id not in raw_qrels or query_id not in qrels:
                raise ValueError(f"eval cases file {src}: missing qrels for query_id {query_id!r}.")
            if "query" not in case and "image_path" not in case:
                raise ValueError(
                    f"eval cases file {src}: query_id {query_id!r} needs query or image_path."
                )
            loaded_cases.append({**dict(case), "qrels": dict(qrels[query_id])})
        loaded[group] = loaded_cases
    return loaded


@dataclass
class V2Result:
    query_id: str
    query: str
    qrels: dict[str, int]
    actual_document_ids: list[str]
    hit: bool
    rank: int | None
    group: str
    evidence_expected: list[str] | None = None
    evidence_hit: bool | None = None
    evidence_rank: int | None = None


def _document_ids(hits: list[SearchHit]) -> list[str]:
    """Extract scoreable logical identities; never fall back to asset/title."""
    document_ids: list[str] = []
    for hit in hits:
        document_id = hit.metadata.get("document_id")
        if isinstance(document_id, str) and document_id.strip():
            document_ids.append(document_id)
    return document_ids


def _first_relevant_rank(actual_document_ids: list[str], qrels: Mapping[str, int]) -> int | None:
    for rank, document_id in enumerate(actual_document_ids, start=1):
        if _is_relevant(document_id, qrels):
            return rank
    return None


def _metric_rows(results: list[V2Result]) -> list[dict[str, object]]:
    return [
        {"actual_document_ids": list(result.actual_document_ids), "qrels": dict(result.qrels)}
        for result in results
    ]


def aggregate_retrieval_scenarios(results: list[V2Result]) -> dict[str, dict[str, object]]:
    """Separate judged retrieval quality from explicit negative behavior."""
    positive = [result for result in results if result.qrels]
    negative = [result for result in results if not result.qrels]
    empty_count = sum(not result.actual_document_ids for result in negative)
    false_count = sum(bool(result.actual_document_ids) for result in negative)
    return {
        "positive": {
            "total": len(positive),
            "hit_count": sum(result.hit for result in positive),
            "hit_rate": sum(result.hit for result in positive) / max(len(positive), 1),
            "metrics": aggregate_metrics(_metric_rows(positive)) if positive else {},
        },
        "negative": {
            "total": len(negative),
            "empty_result_count": empty_count,
            "empty_result_rate": empty_count / max(len(negative), 1),
            "false_retrieval_count": false_count,
            "false_retrieval_rate": false_count / max(len(negative), 1),
        },
    }


def _make_result(
    *, case: Mapping[str, object], hits: list[SearchHit], group: str, query: str
) -> V2Result:
    actual = _document_ids(hits)
    qrels = dict(case["qrels"])  # validated by load_cases
    rank = _first_relevant_rank(actual, qrels)
    evidence_expected = _evidence_terms(case)
    evidence_rank = _first_evidence_rank(hits, evidence_expected)
    return V2Result(
        query_id=str(case["query_id"]),
        query=query,
        qrels=qrels,
        actual_document_ids=actual,
        hit=rank is not None,
        rank=rank,
        group=group,
        evidence_expected=evidence_expected,
        evidence_hit=evidence_rank is not None if evidence_expected is not None else None,
        evidence_rank=evidence_rank,
    )


def _evidence_terms(case: Mapping[str, object]) -> list[str] | None:
    raw_terms = case.get("evidence_contains")
    if raw_terms is None:
        return None
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("evidence_contains must be a non-empty list of non-empty strings.")
    terms = [term.strip() for term in raw_terms if isinstance(term, str) and term.strip()]
    if len(terms) != len(raw_terms):
        raise ValueError("evidence_contains must be a non-empty list of non-empty strings.")
    return terms


def _first_evidence_rank(hits: list[SearchHit], terms: list[str] | None) -> int | None:
    if terms is None:
        return None
    for rank, hit in enumerate(hits, start=1):
        if all(term in hit.evidence for term in terms):
            return rank
    return None


def run_eval_v2(
    top_k: int = 5,
    *,
    collection: str,
    principal: str,
    metadata_filter: dict[str, object] | None = None,
    cases_path: str | Path | None = None,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
) -> list[V2Result]:
    if search_fn is None:
        return run_text_to_text_eval_v2(
            top_k=top_k,
            cases_path=cases_path,
            collection=collection,
            principal=principal,
            metadata_filter=metadata_filter,
        )
    return run_text_to_text_eval_v2(
        top_k=top_k,
        cases_path=cases_path,
        search_fn=search_fn,
        collection=collection,
        principal=principal,
        metadata_filter=metadata_filter,
    )


def run_text_to_text_eval_v2(
    top_k: int = 5,
    *,
    collection: str,
    principal: str,
    metadata_filter: dict[str, object] | None = None,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
    cases_path: str | Path | None = None,
) -> list[V2Result]:
    search = search_fn or get_search_service().execute
    groups = load_cases(cases_path, version="v2")
    results: list[V2Result] = []
    for group, cases in groups.items():
        if group in {"text_to_image", "image_to_image"}:
            continue
        for case in cases:
            query = str(case["query"])
            hits = search(
                SearchCommand(
                    query=query,
                    mode=SearchMode.HYBRID,
                    top_k=top_k,
                    collection=collection,
                    metadata_filter=_case_metadata_filter(case, metadata_filter),
                    principal=_case_principal(case, principal),
                )
            )
            results.append(_make_result(case=case, hits=hits, group=group, query=query))
    return results


def _case_principal(case: Mapping[str, object], default: str) -> str:
    value = case.get("principal", default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case principal must be a non-empty string.")
    return value


def _case_metadata_filter(
    case: Mapping[str, object], default: dict[str, object] | None
) -> dict[str, object] | None:
    value = case.get("metadata_filter", default)
    if value is not None and not isinstance(value, dict):
        raise ValueError("case metadata_filter must be a JSON object.")
    return value


def run_text_to_image_eval_v2(
    top_k: int = 5,
    *,
    collection: str,
    principal: str,
    mode: SearchMode = SearchMode.TEXT_TO_IMAGE,
    metadata_filter: dict[str, object] | None = None,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
    cases_path: str | Path | None = None,
) -> list[V2Result]:
    search = search_fn or get_search_service().execute
    results: list[V2Result] = []
    for case in load_cases(cases_path, version="v2").get("text_to_image", []):
        query = str(case["query"])
        hits = search(
            SearchCommand(
                query=query,
                mode=mode,
                top_k=top_k,
                collection=collection,
                metadata_filter=metadata_filter,
                principal=principal,
            )
        )
        results.append(_make_result(case=case, hits=hits, group="text_to_image", query=query))
    return results


def run_auto_image_eval_v2(
    top_k: int = 5,
    *,
    collection: str,
    principal: str,
    metadata_filter: dict[str, object] | None = None,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
    cases_path: str | Path | None = None,
) -> list[V2Result]:
    """Evaluate the same automatic route used by the image-oriented UI."""
    search = search_fn or get_search_service().execute
    groups = load_cases(cases_path, version="v2")
    results: list[V2Result] = []
    for group in ("text_to_image", "negative"):
        for case in groups.get(group, ()):
            query = str(case["query"])
            hits = search(
                SearchCommand(
                    query=query,
                    mode=SearchMode.AUTO,
                    top_k=top_k,
                    collection=collection,
                    metadata_filter=metadata_filter,
                    principal=principal,
                )
            )
            results.append(_make_result(case=case, hits=hits, group=group, query=query))
    return results


def run_image_to_image_eval_v2(
    top_k: int = 5,
    *,
    collection: str,
    principal: str,
    metadata_filter: dict[str, object] | None = None,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
    cases_path: str | Path | None = None,
) -> list[V2Result]:
    search = search_fn or get_search_service().execute
    results: list[V2Result] = []
    for case in load_cases(cases_path, version="v2").get("image_to_image", []):
        image_path = Path(str(case["image_path"]))
        if image_path.exists():
            command = SearchCommand(
                query=image_path.name,
                mode=SearchMode.IMAGE_TO_IMAGE,
                image_path=image_path,
                top_k=top_k,
                collection=collection,
                metadata_filter=metadata_filter,
                principal=principal,
            )
            hits = _execute_image_eval_search(
                command,
                image_path=image_path,
                search=search,
                trusted_case=search_fn is None,
            )
        else:
            hits = []
        results.append(
            _make_result(case=case, hits=hits, group="image_to_image", query=image_path.name)
        )
    return results


def _execute_image_eval_search(
    command: SearchCommand,
    *,
    image_path: Path,
    search: Callable[[SearchCommand], list[SearchHit]],
    trusted_case: bool,
) -> list[SearchHit]:
    """Preserve the trusted external image-fixture evaluator adapter."""
    if not trusted_case:
        return search(command)
    try:
        relative_path = image_path.resolve().relative_to(get_assets_dir().resolve())
    except ValueError:
        from ..core.registry import get_active_backend

        return get_active_backend().search_image(image_path=image_path, top_k=command.top_k)
    return search(
        SearchCommand(
            query=command.query,
            mode=command.mode,
            image_path=relative_path,
            top_k=command.top_k,
            min_score=command.min_score,
            collection=command.collection,
            metadata_filter=command.metadata_filter,
            principal=command.principal,
        )
    )


def write_eval_report_v2(
    results_by_group: dict[str, list[V2Result]], path=None, *, collection: str | None = None
) -> None:
    """Write per-query qrels and required document-level aggregate metrics."""
    if path is None:
        slug_path = get_eval_report(collection)
        target = slug_path.with_name(
            slug_path.name.replace("eval_report", "eval_report_v2", 1)
        )
    else:
        target = path
    all_results = [result for results in results_by_group.values() for result in results]
    groups = {
        group: {
            "total": len(results),
            "hits": sum(result.hit for result in results),
            "hit_rate": sum(result.hit for result in results) / max(len(results), 1),
            "metrics": aggregate_metrics(_metric_rows(results)) if results else {},
            "evidence": _aggregate_evidence(results),
        }
        for group, results in results_by_group.items()
    }
    payload = build_report(
        kind="retrieval",
        summary={
            "total": len(all_results),
            "scenarios": aggregate_retrieval_scenarios(all_results),
        },
        groups=groups,
        metrics={"all": aggregate_metrics(_metric_rows(all_results)) if all_results else {}},
        per_query=[asdict(result) for result in all_results],
    )
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _aggregate_evidence(results: list[V2Result]) -> dict[str, int | float]:
    judged = [result for result in results if result.evidence_hit is not None]
    hit_count = sum(result.evidence_hit is True for result in judged)
    return {
        "total": len(judged),
        "hit_count": hit_count,
        "hit_rate": hit_count / max(len(judged), 1),
    }


if __name__ == "__main__":  # pragma: no cover
    from ..core.config import load_env

    load_env()
    by_group = {
        "text_to_text": run_text_to_text_eval_v2(),
        "text_to_image": run_text_to_image_eval_v2(),
        "image_to_image": run_image_to_image_eval_v2(),
    }
    write_eval_report_v2(by_group)
