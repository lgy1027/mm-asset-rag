"""Retrieval quality eval (v1): cases loaded from JSON.

**Cases are parameterized** — loaded from a JSON file via
:func:`load_cases` (``{"version", "groups": {group: [case, ...]}}``).
The bundled default (``mm_asset_rag/eval_data/v1_cases.json``) is a
small text→text generic sample over well-known arxiv papers; point
``EVAL_CASES_PATH`` / ``--cases`` at your own file to score a custom
corpus. A larger internal baseline ships at
``examples/eval_cases_chapter11_v1.json`` for reproducibility (see
that file's README — it needs its own corpus ingested).

Every case pairs a free-text ``query`` with one or more
``expected_asset_ids``. Matching is prefix-tolerant: a case "hits" if
any expected id is a substring of any actual id, or vice versa, so
bare model names like ``clip`` still match
``Learning Transferable Visual Models From Natural Language Supervision_79e328a2``
once the search returns the full asset id.

Use :func:`run_eval` to compute the raw results and
:func:`write_eval_report` to persist them to
``$MM_ASSET_RAG_HOME/eval_report.json``. The full per-query details
plus aggregate metrics (hit_rate, MRR, NDCG@k, MAP, ...) are dumped
as JSON.

The shared pure helpers — :func:`strip_trailing_hash`, :func:`_match`,
:func:`_normalize_id_list`, and :func:`_expand` — live in
:mod:`mm_asset_rag.evaluation_v2` and are re-exported here so v1 keeps
its public surface (``load_cases``, ``run_eval``, ``EvalResult``) while
de-duplicating the matcher / normalisation logic. v2 never imports v1,
so there is no circular dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

# Shared pure helpers live in evaluation_v2 (the new-generation module).
# v2 never imports v1, so importing here is cycle-free.
from .evaluation_v2 import (
    _match,
    _normalize_id_list,
    strip_trailing_hash,
)
from .evaluation_v2 import (
    load_cases as _load_cases_v2,
)
from .metrics import aggregate_metrics
from .paths import get_asset_index_path, get_eval_report
from .retrieval import hybrid_search


def load_cases(path: str | Path | None = None) -> dict[str, list[dict]]:
    """Load v1 eval cases as a ``{group: [case, ...]}`` dict.

    Thin wrapper over :func:`mm_asset_rag.evaluation_v2.load_cases` pinned
    to ``version="v1"``. Resolution: explicit ``path`` →
    ``Settings.eval_cases_path`` → the bundled ``v1_cases.json``.
    """
    return _load_cases_v2(path, version="v1")


def _load_bare_to_all_fulls() -> dict[str, list[str]]:
    """Build a ``bare`` → ``[full, ...]`` map covering every hash variant.

    Preserves all duplicates so the matcher accepts the retriever returning
    *any* hash of the same source — relevant when re-parsing produces a new
    SHA but the document content is unchanged. Keys are
    :func:`strip_trailing_hash`-normalised (hash stripped + casefolded) so a
    case-different expected id still resolves to the full-variant set.
    """
    index_path = get_asset_index_path()
    if not index_path.exists():
        return {}
    bare_to_all: dict[str, list[str]] = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("deleted"):
                continue
            full = row.get("asset_id", "")
            if not full:
                continue
            bare = strip_trailing_hash(full)
            if not bare:
                continue
            bare_to_all.setdefault(bare, [])
            if full not in bare_to_all[bare]:
                bare_to_all[bare].append(full)
    return bare_to_all


@dataclass
class EvalResult:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool
    rank: int | None  # 1-based rank of the first hit, None if missed
    group: str  # "en", "zh", or "legacy"
    # Parallel to actual_asset_ids; the LLM-derived paper title per hit (empty
    # when no title). Fed to metrics alongside the asset_id so a paper-title
    # expected id matches the title, mirroring _match's contract.
    actual_titles: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.actual_titles is None:
            self.actual_titles = []


def run_eval(top_k: int = 5, *, cases_path: str | Path | None = None) -> list[EvalResult]:
    """Run the full text→text regression set against the live index.

    Returns a list of :class:`EvalResult` — one per case, in declared
    order. Iterates the text→text groups present in the loaded case file
    (``en`` / ``zh`` / ``zh_doc`` / ``legacy`` — whichever the file
    defines; absent groups are skipped) so the report can break down
    cross-language accuracy.

    ``cases_path`` overrides the case file for this run (default:
    ``Settings.eval_cases_path`` → the bundled ``v1_cases.json``).
    """
    bare_to_all_fulls = _load_bare_to_all_fulls()
    groups = load_cases(cases_path)
    results: list[EvalResult] = []
    for group in ("en", "zh", "zh_doc", "legacy"):
        for case in groups.get(group, ()):
            hits = hybrid_search(str(case["query"]), top_k=top_k)
            # Resolve expected ids to the set of full ids the index
            # actually returns. Accepts both bare titles and full
            # ``<title>_<hash>`` ids, and expands to all hash variants
            # of the same bare document so duplicate parses don't
            # count as misses.
            expected: list[str] = []
            for item in case["expected_asset_ids"]:
                bare = strip_trailing_hash(str(item))
                if bare in bare_to_all_fulls:
                    for f in bare_to_all_fulls[bare]:
                        if f not in expected:
                            expected.append(f)
                elif str(item) not in expected:
                    # User gave a bare or full id that didn't match any
                    # known asset — keep it as a literal expected so
                    # the strict match still has something to compare.
                    expected.append(str(item))
            rank = _match([(hit.asset_id, hit.title or "") for hit in hits], expected)
            results.append(
                EvalResult(
                    query=str(case["query"]),
                    expected_asset_ids=expected,
                    actual_asset_ids=[hit.asset_id for hit in hits],
                    hit=rank is not None,
                    rank=rank,
                    group=group,
                    actual_titles=[hit.title or "" for hit in hits],
                )
            )
    return results


def write_eval_report(results: list[EvalResult], path=None) -> None:
    """Write per-query results + aggregate metrics to ``eval_report.json``.

    The aggregate block uses :func:`mm_asset_rag.metrics.aggregate_metrics`
    which computes hit_rate / precision / recall / f1 / ndcg at k=1,3,5,10
    plus MRR and MAP. Metrics are reported for the full set and per
    language group. Ids are normalised via :func:`strip_trailing_hash`
    before being handed to metrics so re-parses with a different content
    hash don't dilute the aggregate scores.
    """
    target = path or get_eval_report()

    def _to_dict(r: EvalResult) -> dict:
        d = asdict(r)
        return d

    per_query = [_to_dict(r) for r in results]

    def _agg(rs: list[EvalResult]) -> dict:
        if not rs:
            return {}
        return aggregate_metrics(
            [
                {
                    # Pair each actual asset_id with its title so metrics can
                    # match on the LLM-derived paper title too (mirrors _match's
                    # contract → aggregate stays self-consistent with per_query).
                    # zip_longest (not zip) guards against a future code path where
                    # actual_titles and actual_asset_ids diverge in length — a
                    # silent truncation there would drop trailing asset_ids and
                    # skew the metrics. When no titles are present (image route /
                    # old reports), fall back to the bare asset_id.
                    "actual_ids": (
                        [
                            (aid, t) if t else aid
                            for aid, t in zip_longest(
                                r.actual_asset_ids, r.actual_titles, fillvalue=""
                            )
                        ]
                        if r.actual_titles
                        else list(r.actual_asset_ids)
                    ),
                    "expected_ids": _normalize_id_list(r.expected_asset_ids),
                }
                for r in rs
            ]
        )

    # Per-group aggregates: ``all`` + every group present in ``results``
    # (en / zh / zh_doc / legacy / negative / … — whichever the loaded
    # case file defines). Hard-coding only ``en``/``zh`` here used to
    # silently drop the ``zh_doc`` aggregate when running the chapter11
    # opt-in set, so its 10 cross-doc cases counted in ``all`` but had
    # no breakdown of their own.
    by_group: dict[str, list[EvalResult]] = {"all": list(results)}
    for r in results:
        by_group.setdefault(r.group, []).append(r)

    payload = {
        "total": len(results),
        "hit_count": sum(1 for r in results if r.hit),
        "hit_rate": (sum(1 for r in results if r.hit) / max(len(results), 1)),
        "per_query": per_query,
        "metrics": {g: _agg(rs) for g, rs in by_group.items()},
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    from .config import load_env

    load_env()
    res = run_eval(top_k=5)
    write_eval_report(res)
    print(
        json.dumps(
            {
                k: v
                for k, v in [("hit_rate", sum(r.hit for r in res) / len(res)), ("total", len(res))]
            },
            ensure_ascii=False,
        )
    )
