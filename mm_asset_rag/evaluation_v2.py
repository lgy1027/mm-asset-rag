"""v2 retrieval eval: cases loaded from JSON, multi-dimensional groups.

v2 is the new-generation retrieval eval set — opt-in via ``--v2`` on
``mmrag eval`` or ``v2: true`` on ``POST /eval``; the default is still
v1 so existing scripts / dashboards keep their numbers.

**Cases are parameterized** — loaded from a JSON file via
:func:`load_cases` (``{"version", "groups": {group: [case, ...]}}``).
The bundled default (``mm_asset_rag/eval_data/v2_cases.json``) is a
small text→text generic sample; point ``EVAL_CASES_PATH`` / ``--cases``
at your own file to score a custom corpus. A larger internal baseline
ships at ``examples/eval_cases_chapter11_v2.json`` for reproducibility
(see that file's README — it needs its own corpus ingested).

Every case pairs a free-text ``query`` with one or more
``expected_asset_ids``. The ``_match`` helper uses prefix-tolerant
matching so a case "hits" if any expected id is a substring of any
actual id, or vice versa, so bare model names like ``clip`` still
match ``Learning Transferable Visual Models From Natural Language
Supervision_79e328a2`` once the search returns the full asset id.

Use :func:`run_eval_v2` to run text→text, :func:`run_text_to_image_eval_v2`
to run the text→image cases, and :func:`run_image_to_image_eval_v2`
to run the image→image cases. The full per-query results plus
aggregate metrics (hit_rate / precision / recall / f1 / ndcg + MRR + MAP)
are dumped to ``$MM_ASSET_RAG_HOME/eval_report_v2.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.resources import files
from itertools import zip_longest
from pathlib import Path

from .metrics import _is_relevant, aggregate_metrics
from .paths import get_asset_index_path, get_eval_report


def _default_cases_path(version: str) -> Path:
    """Resolve the bundled default case file for ``version`` (``v1``/``v2``).

    Uses :mod:`importlib.resources` so the JSON ships inside the wheel and
    resolves correctly both installed and editable. The default files live
    under ``mm_asset_rag/eval_data/``.
    """
    return files("mm_asset_rag").joinpath("eval_data", f"{version}_cases.json")


def load_cases(path: str | Path | None = None, *, version: str) -> dict[str, list[dict]]:
    """Load eval cases as a ``{group: [case, ...]}`` dict.

    Resolution order: explicit ``path`` arg → ``Settings.eval_cases_path``
    → the bundled default at ``mm_asset_rag/eval_data/<version>_cases.json``.
    The bundled default is a small text→text-only generic sample; point
    ``path`` / ``EVAL_CASES_PATH`` at your own file to score a custom corpus.

    Raises :class:`FileNotFoundError` / :class:`ValueError` with a clear
    message on a missing or malformed file rather than failing silently.
    """
    if path is None:
        env_path = None
        try:
            from .settings import get_settings

            env_path = get_settings().eval_cases_path
        except Exception:  # pragma: no cover - settings infra failure
            env_path = None
        if env_path:
            path = env_path

    if path is None:
        traversable = _default_cases_path(version)
        data = json.loads(traversable.read_text(encoding="utf-8"))
        src = str(traversable)
    else:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"eval cases file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        src = str(p)

    if not isinstance(data, dict):
        raise ValueError(f"eval cases file {src}: expected a top-level JSON object.")
    # Guard against silently scoring 0 cases: the v1 runner iterates
    # ``en/zh/zh_doc/legacy`` and v2 ``zh_on_en/en_on_en/zh_on_zh/negative``,
    # so a cross-version mix-up (v2 file fed to ``mmrag eval`` / v1 file
    # fed to ``--v2``) would leave every group absent and return an empty
    # result with exit 0 — no signal that the wrong file was loaded.
    file_version = data.get("version")
    if file_version is not None and file_version != version:
        raise ValueError(
            f"eval cases file {src}: version mismatch "
            f"(expected {version!r}, got {file_version!r}). "
            f"Use the matching --cases file for the --v2 flag."
        )
    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            f"eval cases file {src}: expected a top-level 'groups' object mapping "
            "group name → list of {{query, expected_asset_ids}} cases."
        )
    return groups


# ── Runner / report writers (parallel to v1 helpers) ──────────────────


@dataclass
class V2Result:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool
    rank: int | None
    group: str
    # Parallel to actual_asset_ids; the LLM-derived paper title per hit (empty
    # when no title, e.g. image route or pre-AUTO_META index). Kept separate
    # so the reported actual_asset_ids stay plain ids while metrics still get
    # the title to match paper-title expected ids.
    actual_titles: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.actual_titles is None:
            self.actual_titles = []


def _load_full_ids() -> set[str]:
    """Return the set of full asset_ids for the active rows.

    Multiple ``_NNN_hash`` variants of the same content can coexist
    in the index (re-ingestion of a slightly different file revision
    keeps the SHA256 stable but bumps a per-write hash). We dedupe by
    ``asset_id`` instead of ``sha256`` so ``_expand`` can return every
    hash variant — ``_match`` then accepts any of them as a valid hit.
    """
    seen: set[str] = set()
    out: set[str] = set()
    index_path = get_asset_index_path()
    if not index_path.exists():
        return out
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
            aid = row.get("asset_id", "")
            if aid in seen:
                continue
            seen.add(aid)
            out.add(aid)
    return out


def _expand(prefix: str, full_ids: set[str]) -> list[str]:
    """Expand a bare prefix (e.g. ``Caltech Airplanes``) to all full
    asset_ids that start with it. Returns ``[prefix]`` if no match
    so the strict match still works when the caller passed a full id.

    Multiple ``_NNN_hash`` variants of the same title are common (each
    parse run + content edit produces a new SHA). A single hash should
    *not* be treated as the canonical answer — the matcher below also
    accepts any actual id whose title is prefixed by the bare term.
    """
    matches = sorted(f for f in full_ids if f.startswith(prefix))
    return matches if matches else [prefix]


def _title_of(asset_id: str) -> str:
    """Strip the trailing ``_<8-hex-hash>`` from an asset id to get the
    bare title used for prefix-tolerant matching.

    Asset ids look like ``<title>_<8-hex>``. If the id has no ``_`` we
    return the whole id (e.g. for synthetic or user-supplied ids).
    """
    if "_" not in asset_id:
        return asset_id
    # The hash is the last ``_``-segment, exactly 8 lowercase hex chars.
    head, _, tail = asset_id.rpartition("_")
    if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
        return head
    return asset_id


def strip_trailing_hash(asset_id: str) -> str:
    """Normalise an asset id for eval matching.

    Drops a trailing ``_<8-hex>`` content-hash suffix and casefolds the
    remainder so ``Rich feature hierarchies`` matches
    ``Rich Feature Hierarchies for Accurate Object Detection And Semantic Segmentation_b857cf69``.
    Mirrors the v1 helper so both eval harnesses apply the same
    normalisation.
    """
    if not asset_id:
        return ""
    if "_" in asset_id:
        head, _, tail = asset_id.rpartition("_")
        if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
            return head.casefold()
    return asset_id.casefold()


def _actual_candidates(act: str | tuple[str, str]) -> list[str]:
    """Normalise an actual entry to the list of id-strings to compare.

    Accepts either a bare ``asset_id`` (str, for backwards compatibility with
    image-route evals and unit tests) or an ``(asset_id, title)`` pair. For the
    pair, both the asset_id and the title are candidates — a hit on either
    counts as relevant. This is what lets the CLIP case resolve when the
    retriever returns ``("clip_b14b418e", "Learning Transferable Visual Models
    From Natural Language Supervision")`` and the expected id is the paper's
    canonical title: the asset_id alone never matches, but the LLM-derived
    title does.
    """
    if isinstance(act, tuple):
        return [v for v in act if v]
    return [act] if act else []


def _match(actual: list[str | tuple[str, str]], expected: list[str]) -> int | None:
    # Relevance is the normalised bidirectional-substring check shared with
    # ``aggregate_metrics`` (via ``metrics._is_relevant``), so per-query
    # ``hit``/``rank`` and the aggregate ``hit_rate``/``MRR``/``NDCG`` agree
    # by construction — a bare short title matches a longer returned id both
    # here and in the reported metrics. Each actual may carry a title too
    # (see :func:`_actual_candidates`); the asset_id or the title matching any
    # expected id is a hit at that rank.
    for rank, act in enumerate(actual, start=1):
        if any(_is_relevant(c, expected) for c in _actual_candidates(act)):
            return rank
    return None


def run_eval_v2(top_k: int = 5, *, cases_path: str | Path | None = None) -> list[V2Result]:
    """Run the v2 text→text regression set (CLI / API entry point).

    Thin alias for :func:`run_text_to_text_eval_v2` so the CLI ``--v2``
    flag and the API ``v2: true`` field can treat v1 and v2 symmetrically
    (``run_eval`` vs ``run_eval_v2``). Returns the same :class:`V2Result`
    shape as the other v2 runners, which parallels v1's ``EvalResult`` so
    callers can swap the two without adapting the response shape.

    ``cases_path`` overrides the case file for this run (default:
    ``Settings.eval_cases_path`` → the bundled ``v2_cases.json``).

    The text→image / image→image groups have their own runners; this
    convenience only covers the text→text set because that is what v1's
    ``run_eval`` covers and what the default ``mmrag eval`` output
    compares against.
    """
    return run_text_to_text_eval_v2(top_k=top_k, cases_path=cases_path)


def run_text_to_text_eval_v2(
    top_k: int = 5,
    *,
    search_fn: Callable[[str, int], list] | None = None,
    full_ids: set[str] | None = None,
    cases_path: str | Path | None = None,
) -> list[V2Result]:
    """Run all v2 text→text cases against the live hybrid index.

    ``search_fn`` defaults to ``retrieval.hybrid_search`` (production
    path). Tests pass a stub that returns canned ``SearchHit`` lists
    so the eval can run offline against a mock corpus — this is the
    ``auto-eval`` integration used by CI. The stub signature is
    ``(query: str, top_k: int) -> list[SearchHit]``; the query
    preprocessor / RRF / min_score are bypassed because the test
    ships its own pre-computed results.

    ``full_ids`` is the set of known asset_ids used by ``_expand`` to
    resolve bare expected ids. Tests inject a synthetic set so the
    eval runs without a real ``asset_index.jsonl``.

    ``cases_path`` overrides the case file (default: the bundled
    ``v2_cases.json`` or ``Settings.eval_cases_path``). Only the
    text→text groups are iterated; image groups belong to the other
    runners. A group absent from the file is skipped.
    """
    from .retrieval import hybrid_search

    search = search_fn or hybrid_search
    ids = full_ids if full_ids is not None else _load_full_ids()

    groups = load_cases(cases_path, version="v2")
    out: list[V2Result] = []
    for group in ("zh_on_en", "en_on_en", "zh_on_zh", "negative"):
        for case in groups.get(group, ()):
            hits = search(str(case["query"]), top_k=top_k)
            # Carry (asset_id, title) pairs for _match so it can hit on either.
            # The asset_id is a filename stem (e.g. clip_b14b418e) which never
            # matches a paper-title expected id; with AUTO_META on, hit.title is
            # the LLM-derived canonical paper title, which does. The reported
            # actual_asset_ids stay plain asset_ids for readability.
            actual_pairs: list[tuple[str, str]] = [(hit.asset_id, hit.title or "") for hit in hits]
            expected: list[str] = []
            for item in case["expected_asset_ids"]:
                expected.extend(_expand(str(item), ids))
            rank = _match(actual_pairs, expected) if expected else None
            out.append(
                V2Result(
                    query=str(case["query"]),
                    expected_asset_ids=expected,
                    actual_asset_ids=[hit.asset_id for hit in hits],
                    hit=rank is not None,
                    rank=rank,
                    group=group,
                    actual_titles=[hit.title or "" for hit in hits],
                )
            )
    return out


def run_text_to_image_eval_v2(
    top_k: int = 5,
    *,
    search_fn: Callable[[str, int], list] | None = None,
    full_ids: set[str] | None = None,
    cases_path: str | Path | None = None,
) -> list[V2Result]:
    """Run the v2 text→image cases against the Qdrant image collection.

    ``search_fn`` is the dependency-injection hook for tests; default
    is the live ``qdrant_text_to_image_search`` call. See
    :func:`run_text_to_text_eval_v2` for the same pattern.
    """
    from .backends.qdrant_backend import qdrant_text_to_image_search

    search = search_fn or qdrant_text_to_image_search
    ids = full_ids if full_ids is not None else _load_full_ids()

    cases = load_cases(cases_path, version="v2").get("text_to_image", [])
    out: list[V2Result] = []
    for case in cases:
        hits = search(str(case["query"]), top_k)
        actual = [hit.asset_id for hit in hits]
        expected: list[str] = []
        for item in case["expected_asset_ids"]:
            expected.extend(_expand(str(item), ids))
        rank = _match(actual, expected) if expected else None
        out.append(
            V2Result(
                query=str(case["query"]),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=rank is not None,
                rank=rank,
                group="text_to_image",
            )
        )
    return out


def run_image_to_image_eval_v2(
    top_k: int = 5, *, cases_path: str | Path | None = None
) -> list[V2Result]:
    """Run the v2 image→image cases."""
    from .backends.qdrant_backend import qdrant_image_to_image_search

    full_ids = _load_full_ids()
    out: list[V2Result] = []
    for case in load_cases(cases_path, version="v2").get("image_to_image", []):
        image_path = Path(case["image_path"])
        if not image_path.exists():
            out.append(
                V2Result(
                    query=str(image_path.name),
                    expected_asset_ids=list(case["expected_asset_ids"]),
                    actual_asset_ids=[],
                    hit=False,
                    rank=None,
                    group="image_to_image",
                )
            )
            continue
        hits = qdrant_image_to_image_search(image_path, top_k=top_k)
        actual = [hit.asset_id for hit in hits]
        expected: list[str] = []
        for item in case["expected_asset_ids"]:
            expected.extend(_expand(str(item), full_ids))
        rank = _match(actual, expected) if expected else None
        out.append(
            V2Result(
                query=str(image_path.name),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=rank is not None,
                rank=rank,
                group="image_to_image",
            )
        )
    return out


def _normalize_id_list(ids: list[str]) -> list[str]:
    """Strip ``_<8-hex>`` hash suffixes + casefold, dedup preserving order.

    Used to collapse duplicate hash variants of the same document before
    writing them into the eval report's ``expected_ids``/``actual_ids``
    fields (so a re-parse producing a new hash doesn't bloat the stored
    lists). :mod:`mm_asset_rag.metrics` applies its own normalisation on
    top when computing relevance, so this pre-pass is purely cosmetic for
    the report payload — it does not change any metric value.
    """
    out: list[str] = []
    for aid in ids:
        norm = strip_trailing_hash(aid)
        if norm and norm not in out:
            out.append(norm)
    return out


def write_eval_report_v2(results_by_group: dict[str, list[V2Result]], path=None) -> None:
    """Write per-query results + per-group aggregate metrics to JSON."""
    target = path or get_eval_report().with_name("eval_report_v2.json")

    def _agg(rs: list[V2Result]) -> dict:
        if not rs:
            return {}
        return aggregate_metrics(
            [
                {
                    # Pair each actual asset_id with its title so metrics can
                    # match on the LLM-derived paper title too (matches the
                    # per-query _match contract → aggregate stays self-consistent
                    # with per_query hit/rank). Fall back to bare asset_id when
                    # no titles (image route / old reports). zip_longest (not
                    # zip) guards against a future code path where the two lists
                    # diverge in length — silent truncation would drop trailing
                    # asset_ids and skew the metrics.
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

    payload = {
        "version": "v2",
        "per_group": {
            g: {
                "total": len(rs),
                "hits": sum(1 for r in rs if r.hit),
                "hit_rate": sum(1 for r in rs if r.hit) / max(len(rs), 1),
                "metrics": _agg(rs),
                "per_query": [asdict(r) for r in rs],
            }
            for g, rs in results_by_group.items()
        },
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    from .config import load_env

    load_env()
    t2t = run_text_to_text_eval_v2(top_k=5)
    t2i = run_text_to_image_eval_v2(top_k=5)
    i2i = run_image_to_image_eval_v2(top_k=5)
    by_group: dict[str, list[V2Result]] = {
        "text_to_text": t2t,
        "text_to_image": t2i,
        "image_to_image": i2i,
    }
    write_eval_report_v2(by_group)
    for g, rs in by_group.items():
        hits = sum(1 for r in rs if r.hit)
        print(f"{g}: {hits}/{len(rs)} hit_rate={hits / max(len(rs), 1):.3f}")
