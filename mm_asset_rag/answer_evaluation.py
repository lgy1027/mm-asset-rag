"""``/answer`` quality evaluation — coverage + citation + LLM-judge faithfulness.

Why this module exists
----------------------

``mm_asset_rag.evaluation`` / ``evaluation_v2`` measure **retrieval** quality
(hit_rate / precision / recall / f1 / ndcg / MRR / MAP at k=1,3,5,10). They
say nothing about whether the **generated answer** is faithful to the
retrieved evidence, whether it cites the documents it should, or whether it
covers the keywords a domain expert would expect. Without those signals,
hallucination regressions, prompt changes, and LLM swaps are all invisible.

This module adds the answer side:

- ``coverage`` — cheap substring match (NFC + casefold + ZW-char strip) of
  ``expected_answer_keywords`` against the model output. No LLM. Always runs.
- ``citation precision / recall`` — regex-extracts ``[N]`` markers from the
  answer (matching the ``_build_evidence_context`` formatting in
  ``answer.py``), looks each marker up in the top-k evidence to get an
  ``asset_id``, and compares against ``expected_answer_assets`` (which falls
  back to ``expected_asset_ids`` when unset). No LLM. Always runs.
- ``faithfulness`` — single-shot LLM-as-judge: prompts the LLM with the
  question + evidence block + the model's answer and asks for
  ``{"faithfulness": float, "unsupported_claims": [...]}`` (with bare-float
  fallback for older / smaller judges that refuse JSON). Skipped when no
  judge creds are configured, when the call times out, when the case is over
  ``EVAL_JUDGE_MAX_CASES``, or on any exception — one bad case never aborts
  the run.

CLI / API surface
-----------------

- ``mmrag eval --answer-quality`` — runs the runner with default
  ``SearchService`` / ``llm_answer`` and writes
  ``$MM_ASSET_RAG_HOME/eval_report_answer.json``.
- ``POST /eval {"answer_quality": true}`` — same, via the API.

DI hooks (mirroring ``evaluation_v2.run_text_to_text_eval_v2``)
---------------------------------------------------------------

``run_answer_eval`` accepts ``search_fn`` / ``answer_fn`` / ``judge_fn`` /
``full_ids`` / ``max_judge_cases`` so unit tests can mock every layer
without touching the real retriever / LLM. Default arguments resolve at
call time, not at import time, so monkeypatched modules (see
``tests/conftest._isolate_env_file``) are picked up correctly.

v0 scope
--------

**Text→text only.** Cases with an ``image_path`` field, or groups named
``text_to_image`` / ``image_to_image``, raise a clear ``ValueError``:
``llm_answer(question, hits)`` doesn't accept an image input today and the
runner would silently score 0 if we didn't refuse. Image-route support is
tracked separately and will land when the answer module grows an
``image_query`` parameter.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from .paths import get_answer_eval_report
from .schema import SearchHit
from .search_service import SearchCommand, SearchMode, get_search_service
from .settings import get_settings

# ─── Dataclass ────────────────────────────────────────────────────────────


@dataclass
class AnswerEvalResult:
    """One row of the answer-quality report. All score fields are 0.0-1.0."""

    query: str
    expected_asset_ids: list[str]
    expected_answer_assets: list[str]  # resolved: falls back to expected_asset_ids
    actual_asset_ids: list[str]  # from answer sources
    answer_text: str
    answer_source: str  # "llm" | "fallback"
    coverage: float
    citation_precision: float
    citation_recall: float
    citation_present: bool
    faithfulness: float | None
    faithfulness_skipped: bool
    faithfulness_error: str | None
    group: str


# ─── Public runner ────────────────────────────────────────────────────────


def run_answer_eval(
    cases_path: str | Path | None = None,
    *,
    top_k: int = 5,
    collection: str,
    principal: str,
    metadata_filter: dict[str, object] | None = None,
    search_fn: Callable[[SearchCommand], list[SearchHit]] | None = None,
    answer_fn: Callable[[str, list], dict] | None = None,
    judge_fn: Callable[[str, list, str], float] | None = None,
    full_ids: set[str] | None = None,
    max_judge_cases: int | None = None,
) -> list[AnswerEvalResult]:
    """Run the answer-quality eval over a case JSON.

    Defaults resolve at call time:
    - ``search_fn`` → ``get_search_service().execute``
    - ``answer_fn`` → ``mm_asset_rag.answer.llm_answer``
    - ``judge_fn``  → :func:`faithfulness_judge`
    - ``full_ids``  → loaded from ``asset_index.jsonl`` (empty set on miss)
    - ``max_judge_cases`` → ``settings.eval_judge_max_cases``

    Returns one ``AnswerEvalResult`` per case, in case-file order. The
    runner never raises on a per-case failure — judge errors land as
    ``faithfulness_skipped=True`` rows. Image-route cases raise ``ValueError``
    up front (see module docstring).
    """
    # Default-fn resolution deferred so tests can monkeypatch modules
    # (see ``tests/conftest._isolate_env_file``).
    from .answer import llm_answer
    from .evaluation_v2 import _load_full_ids as _load_full_ids_v2

    search = search_fn or get_search_service().execute
    answer_call = answer_fn or llm_answer
    judge_call = judge_fn or faithfulness_judge

    if max_judge_cases is None:
        max_judge_cases = get_settings().eval_judge_max_cases

    groups, _version = _load_cases(cases_path)
    _assert_no_image_cases(groups)

    if full_ids is None:
        full_ids = _load_full_ids_v2()

    judge_count = 0  # for max_judge_cases cap

    results: list[AnswerEvalResult] = []
    for group_name, cases in groups.items():
        for case in cases:
            query = case["query"]
            expected_asset_ids = list(case.get("expected_asset_ids") or [])
            expected_answer_assets = list(case.get("expected_answer_assets") or expected_asset_ids)

            # Retrieval is command-level DI so answer quality uses the same
            # production search policy as public answer and retrieval eval.
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

            # Passing the pre-computed hits keeps answer scoring aligned with
            # the retrieval command evaluated above.
            answer_result = answer_call(query, hits)
            answer_text = str(answer_result.get("answer", ""))
            sources = answer_result.get("sources") or []
            actual_asset_ids = [s.get("asset_id") for s in sources if s.get("asset_id")]
            answer_source = "fallback" if answer_result.get("_fallback") else "llm"

            # Cheap scorers (no LLM)
            coverage = _coverage(answer_text, list(case.get("expected_answer_keywords") or []))
            cit_prec, cit_rec, cit_present = _citation_metrics(
                answer_text, hits, expected_answer_assets, full_ids
            )

            # Expensive scorer (LLM judge) — capped + isolated
            faithfulness, fs_skipped, fs_error = _judge_one(
                query, hits, answer_text, judge_call, judge_count, max_judge_cases
            )
            if not fs_skipped and answer_source == "llm":
                judge_count += 1

            results.append(
                AnswerEvalResult(
                    query=query,
                    expected_asset_ids=expected_asset_ids,
                    expected_answer_assets=expected_answer_assets,
                    actual_asset_ids=actual_asset_ids,
                    answer_text=answer_text,
                    answer_source=answer_source,
                    coverage=coverage,
                    citation_precision=cit_prec,
                    citation_recall=cit_rec,
                    citation_present=cit_present,
                    faithfulness=faithfulness,
                    faithfulness_skipped=fs_skipped,
                    faithfulness_error=fs_error,
                    group=group_name,
                )
            )

    return results


def write_answer_eval_report(
    results: list[AnswerEvalResult],
    path: Path | None = None,
) -> None:
    """Write the answer-quality report (payload version ``answer_v1``)."""
    payload = _aggregate_answer_metrics(results)
    out = path or get_answer_eval_report()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Faithfulness judge (public so tests can override) ────────────────────


def faithfulness_judge(question: str, hits: list[SearchHit], answer: str) -> float:
    """LLM-as-judge: score 0.0-1.0 for whether the answer is grounded in evidence.

    Skips with a logged reason when no creds are configured (the runner
    records this as ``faithfulness_skipped=True``). On any HTTP / parse error
    we re-raise — the runner's per-case try/except records ``skipped`` and
    continues.

    The judge LLM is requested to return JSON
    ``{"faithfulness": float, "unsupported_claims": [str, ...]}`` for
    diagnostic value; the parser falls back to a bare float (first 0.0-1.0
    match) when the model refuses JSON mode.
    """
    settings = get_settings()
    creds = settings.llm_creds
    if not all(creds):
        raise _JudgeUnavailable("no LLM creds configured (set OPENAI_* or VLM_*)")

    base_url, api_key, model = creds
    judge_model = settings.eval_judge_model or model
    timeout = float(settings.eval_judge_timeout)

    evidence_block = _format_evidence_for_judge(hits)
    messages = [
        {
            "role": "system",
            "content": (
                "你是严格的检索增强生成 (RAG) 评估员。给定 (问题, 证据块, "
                "模型答案),判断模型答案中的事实性陈述是否有证据支持。\n"
                "输出严格的 JSON:\n"
                '{"faithfulness": 0.85, "unsupported_claims": ["..."]}\n'
                "faithfulness 范围 0.0(完全无中生有) 到 1.0(全部有据可查)。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题: {question}\n\n"
                f"证据:\n{evidence_block}\n\n"
                f"模型答案: {answer}\n\n"
                "按系统指令输出 JSON。"
            ),
        },
    ]

    raw = _judge_chat_json(base_url, api_key, judge_model, messages, timeout=timeout)
    return _parse_faithfulness(raw)


class _JudgeUnavailable(Exception):
    """Raised internally when the judge can't run; runner records as skipped."""


# ─── Internal helpers ─────────────────────────────────────────────────────


def _normalize_for_coverage(s: str) -> str:
    """NFC + casefold + zero-width / ideographic-space strip + whitespace collapse.

    Handles three real-world noise sources:
    - one tool writes ``é`` as two code points (``e`` + combining acute),
      another as one → NFC normalizes to the same form
    - copy/paste through chat apps introduces zero-width spaces / BOM / full-
      width spaces → stripped
    - whitespace runs (``"双碳  目标"``) → collapsed to single space
    """
    s = unicodedata.normalize("NFC", s)
    # Strip zero-width + BOM + full-width + ideographic space.
    s = re.sub(r"[​‌‍⁠﻿　]", "", s)
    s = s.casefold()
    s = re.sub(r"[\s　]+", " ", s).strip()
    return s


def _coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of keywords that appear (substring) in the answer after normalize."""
    if not keywords:
        return 0.0
    norm_answer = _normalize_for_coverage(answer)
    matched = 0
    for kw in keywords:
        if not kw:
            continue
        if _normalize_for_coverage(kw) in norm_answer:
            matched += 1
    return matched / len(keywords)


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _citation_metrics(
    answer: str,
    hits: list[SearchHit],
    expected: list[str],
    full_ids: set[str],
) -> tuple[float, float, bool]:
    """Extract ``[N]`` markers → look up hits → compare to expected.

    Returns ``(precision, recall, citation_present)``.
    """
    # Extract unique sorted rank numbers from the answer text.
    ranks = sorted({int(m.group(1)) for m in _CITATION_RE.finditer(answer)})
    # Map each rank to the hit's asset_id (1-based indexing, matches the
    # evidence block format). Out-of-range ranks are silently dropped —
    # `citation_present=False` will surface that to the report.
    cited: list[str] = []
    for rank in ranks:
        idx = rank - 1
        if 0 <= idx < len(hits):
            aid = hits[idx].asset_id
            if aid:
                cited.append(aid)
    cited = list(dict.fromkeys(cited))  # dedup, preserve order

    if not cited:
        return 0.0, 0.0, False

    # Expand bare expected ids (e.g. "Bert" → all hashed variants) the
    # same way v2 retrieval does, then normalise both sides via the
    # canonical metrics._normalize_id slug so trailing _hash + casefold +
    # separator differences don't cost a point.
    from .evaluation_v2 import _expand
    from .metrics import _normalize_id

    expanded: list[str] = []
    for exp in expected:
        expanded.extend(_expand(exp, full_ids))
    expected_norms = {_normalize_id(e) for e in expanded if e}

    cited_norms = {_normalize_id(c) for c in cited if c}

    # Use bidirectional substring (mirrors metrics._is_relevant semantics)
    # so a bare expected like "Bert" matches the cited "Bert_ec793c5d".
    hits_set = cited_norms & expected_norms
    # Bidirectional match for the residual cases
    for c in cited_norms:
        for e in expected_norms:
            if c != e and (c in e or e in c):
                hits_set.add(c)
                break

    matched = len(hits_set)
    precision = matched / len(cited) if cited else 0.0
    recall = matched / len(expected_norms) if expected_norms else 0.0
    return precision, recall, True


def _judge_one(
    query: str,
    hits: list[SearchHit],
    answer: str,
    judge_call: Callable[[str, list, str], float],
    judge_count: int,
    max_judge_cases: int | None,
) -> tuple[float | None, bool, str | None]:
    """Run one faithfulness judgement with skip-on-failure semantics.

    Returns ``(faithfulness, skipped, error)``. ``faithfulness`` is non-None
    only when ``skipped`` is False.
    """
    if max_judge_cases is not None and judge_count >= max_judge_cases:
        return None, True, "max cases reached"
    try:
        score = float(judge_call(query, hits, answer))
    except _JudgeUnavailable as exc:
        return None, True, str(exc)
    except (requests.Timeout, requests.HTTPError, ValueError, KeyError) as exc:
        return None, True, f"{type(exc).__name__}: {exc}"
    # Clamp into [0, 1] — judge may overshoot.
    score = max(0.0, min(1.0, score))
    return score, False, None


def _judge_chat_json(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    *,
    timeout: float,
) -> str:
    """One OpenAI-compatible chat completion POST requesting JSON output.

    Returns the raw response text. Raises ``requests.HTTPError`` on non-2xx,
    ``requests.Timeout`` on timeout, ``KeyError`` if the response shape is
    unexpected. Caller (:func:`_judge_one`) catches these and records them
    as ``faithfulness_skipped=True``.
    """
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.0,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


_FLOAT_RE = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b")


def _parse_faithfulness(raw: str) -> float:
    """Parse JSON ``{"faithfulness": float}`` or fall back to first float in text."""
    text = (raw or "").strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "faithfulness" in data:
                return float(data["faithfulness"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    match = _FLOAT_RE.search(text)
    if match:
        return float(match.group(1))
    raise ValueError(f"could not parse faithfulness from judge response: {text!r}")


def _format_evidence_for_judge(hits: list[SearchHit]) -> str:
    """Mirror ``answer._build_evidence_context`` so the judge sees the same
    evidence block the model saw (capped per-hit for token budget)."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        header = (
            f"[{i}] asset_id={hit.asset_id} title={hit.title} "
            f"source={hit.source_path} page={hit.metadata.get('page')}"
        )
        body = (hit.evidence or "")[:1200]
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks) if blocks else "(no evidence)"


# ─── Case loader + image-route guard ──────────────────────────────────────


def _load_cases(cases_path: str | Path | None) -> tuple[dict[str, list[dict]], str]:
    """Load + minimally validate the eval case JSON.

    Reuses the permissive v2 loader shape (``{version, groups: {group: [...]}}``)
    so existing case files work unchanged. Unknown per-case fields
    (``expected_answer_keywords`` / ``expected_answer_assets``) pass through.

    Resolution: explicit ``cases_path`` → ``Settings.eval_cases_path`` →
    bundled default at ``mm_asset_rag/eval_data/answer_v1_cases.json``
    (resolved via ``importlib.resources`` so it ships inside the wheel).
    """
    from .evaluation_v2 import _default_cases_path

    if cases_path is None:
        try:
            env_path = get_settings().eval_cases_path
        except Exception:  # pragma: no cover - settings infra failure
            env_path = None
        cases_path = env_path or _default_cases_path("answer_v1")

    if hasattr(cases_path, "read_text"):  # importlib Traversable (bundled)
        src = str(cases_path)
        data = json.loads(cases_path.read_text(encoding="utf-8"))
    else:
        p = Path(cases_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"eval cases file not found: {p}")
        src = str(p)
        data = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"eval cases JSON must have a 'groups' field: {src}")
    groups = data["groups"]
    if not isinstance(groups, dict):
        raise ValueError(f"eval cases JSON 'groups' must be a dict: {src}")
    return groups, str(data.get("version", "v1"))


def _assert_no_image_cases(groups: dict[str, list[dict]]) -> None:
    """v0: refuse any case that needs image input (citation pattern doesn't
    transfer, and ``llm_answer(question, hits)`` doesn't take an image_path)."""
    for group_name, cases in groups.items():
        if group_name.startswith(("text_to_image", "image_to_image")):
            raise ValueError(
                f"answer-quality eval is text→text only in v0; group "
                f"{group_name!r} is image-route. Image-route support will "
                "land with the answer module's image_query parameter."
            )
        for case in cases:
            if case.get("image_path"):
                raise ValueError(
                    "answer-quality eval is text→text only in v0; case "
                    f"{case.get('query', '?')!r} has image_path."
                )


# ─── Aggregation + report writer ──────────────────────────────────────────


def _aggregate_answer_metrics(results: list[AnswerEvalResult]) -> dict:
    """Group results, compute per-group means + answer_source breakdown."""
    per_group: dict[str, dict] = {}
    for r in results:
        g = per_group.setdefault(
            r.group,
            {
                "total": 0,
                "coverage_sum": 0.0,
                "cit_prec_sum": 0.0,
                "cit_rec_sum": 0.0,
                "cit_present_count": 0,
                "faith_sum": 0.0,
                "faith_count": 0,
                "faith_skipped": 0,
                "per_query": [],
            },
        )
        g["total"] += 1
        g["coverage_sum"] += r.coverage
        g["cit_prec_sum"] += r.citation_precision
        g["cit_rec_sum"] += r.citation_recall
        if r.citation_present:
            g["cit_present_count"] += 1
        if not r.faithfulness_skipped and r.faithfulness is not None:
            g["faith_sum"] += r.faithfulness
            g["faith_count"] += 1
        else:
            g["faith_skipped"] += 1
        g["per_query"].append(
            {
                "query": r.query,
                "answer_source": r.answer_source,
                "coverage": r.coverage,
                "citation_precision": r.citation_precision,
                "citation_recall": r.citation_recall,
                "citation_present": r.citation_present,
                "faithfulness": r.faithfulness,
                "faithfulness_skipped": r.faithfulness_skipped,
                "faithfulness_error": r.faithfulness_error,
            }
        )

    payload_groups: dict[str, dict] = {}
    for name, g in per_group.items():
        n = max(g["total"], 1)
        payload_groups[name] = {
            "total": g["total"],
            "coverage_mean": round(g["coverage_sum"] / n, 4),
            "citation_precision_mean": round(g["cit_prec_sum"] / n, 4),
            "citation_recall_mean": round(g["cit_rec_sum"] / n, 4),
            "citation_present_rate": round(g["cit_present_count"] / n, 4),
            "faithfulness_mean": (
                round(g["faith_sum"] / g["faith_count"], 4) if g["faith_count"] else None
            ),
            "faithfulness_skipped": g["faith_skipped"],
            "per_query": g["per_query"],
        }

    answer_source_breakdown: dict[str, int] = {}
    for r in results:
        answer_source_breakdown[r.answer_source] = (
            answer_source_breakdown.get(r.answer_source, 0) + 1
        )

    # Aggregate over all groups (simple mean of group means, weighted by
    # group size — keeps per-group skew visible while exposing one headline).
    all_n = max(len(results), 1)
    payload = {
        "version": "answer_v1",
        "total": len(results),
        "answer_source_breakdown": answer_source_breakdown,
        "per_group": payload_groups,
        "metrics": {
            "all": {
                "coverage_mean": round(sum(r.coverage for r in results) / all_n, 4),
                "citation_precision_mean": round(
                    sum(r.citation_precision for r in results) / all_n, 4
                ),
                "citation_recall_mean": round(sum(r.citation_recall for r in results) / all_n, 4),
                "citation_present_rate": round(
                    sum(1 for r in results if r.citation_present) / all_n, 4
                ),
                "faithfulness_mean": _safe_mean(
                    [r.faithfulness for r in results if not r.faithfulness_skipped]
                ),
                "faithfulness_skipped": sum(1 for r in results if r.faithfulness_skipped),
            }
        },
    }
    return payload


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)
