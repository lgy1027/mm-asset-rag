"""Full evaluation runner: text→text + text→image + image→image + answer-quality.

调用现成的 evaluation_v2 / answer_evaluation 模块,聚合 4 份 JSON 报告,
输出 Markdown 表格。所有 runner 沿用 DI 接口 (*, search_fn / answer_fn /
judge_fn),这里传 None 走真实 hybrid/单路由,跟 mmrag CLI 行为一致。

Usage:
    uv run python scripts/run_full_eval.py
    uv run python scripts/run_full_eval.py --cases chapter11_v2.json --top-k 5
    uv run python scripts/run_full_eval.py --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

from mm_asset_rag.answer_evaluation import (
    _aggregate_answer_metrics,
    run_answer_eval,
    write_answer_eval_report,
)
from mm_asset_rag.evaluation_v2 import (
    V2Result,
    run_image_to_image_eval_v2,
    run_text_to_image_eval_v2,
    run_text_to_text_eval_v2,
    write_eval_report_v2,
)
from mm_asset_rag.search_service import SearchCommand, get_search_service

# ── latency instrumentation ──────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _latency_block(timings_ms: list[float]) -> dict:
    return {
        "count": len(timings_ms),
        "p50_ms": round(_percentile(timings_ms, 0.5), 1),
        "p95_ms": round(_percentile(timings_ms, 0.95), 1),
        "p99_ms": round(_percentile(timings_ms, 0.99), 1),
        "mean_ms": round(statistics.mean(timings_ms), 1) if timings_ms else 0.0,
    }


def _wrap_search(latencies: list[float]):
    """Time one retrieval request through the application service."""
    service = get_search_service()

    def wrapped(command: SearchCommand):
        t0 = time.perf_counter()
        hits = service.execute(command)
        latencies.append((time.perf_counter() - t0) * 1000)
        return hits

    return wrapped


# ── runners ──────────────────────────────────────────────────────────────


def run_text_to_text(
    cases_path: str, top_k: int, lat_text: list[float], collection: str, principal: str
) -> list:
    """Run v2 text→text eval (zh_on_en / en_on_en / zh_on_zh / negative)."""
    return run_text_to_text_eval_v2(
        top_k=top_k,
        cases_path=cases_path,
        collection=collection,
        principal=principal,
        search_fn=_wrap_search(lat_text),
    )


def run_text_to_image(
    cases_path: str, top_k: int, lat_t2i: list[float], collection: str, principal: str
) -> list:
    """Run v2 text→image eval (text→image group)."""
    return run_text_to_image_eval_v2(
        top_k=top_k,
        cases_path=cases_path,
        collection=collection,
        principal=principal,
        search_fn=_wrap_search(lat_t2i),
    )


def run_image_to_image(cases_path: str, top_k: int, collection: str, principal: str) -> list:
    """Run image→image through the evaluator's external-fixture adapter."""
    return run_image_to_image_eval_v2(
        top_k=top_k,
        cases_path=cases_path,
        collection=collection,
        principal=principal,
    )


def run_answer(
    cases_path: str, top_k: int, max_judge_cases: int | None, collection: str, principal: str
) -> list:
    """Run answer-quality eval (text→text only, image-route raises ValueError)."""
    return run_answer_eval(
        top_k=top_k,
        cases_path=cases_path,
        max_judge_cases=max_judge_cases,
        collection=collection,
        principal=principal,
    )


# ── aggregation ──────────────────────────────────────────────────────────


def _v2_to_grouped(results: list[V2Result]) -> dict[str, list]:
    by: dict[str, list] = {}
    for r in results:
        by.setdefault(r.group, []).append(r)
    return by


def _v2_metrics(results: list[V2Result]) -> dict:
    """Compute hit_rate / MRR / NDCG / MAP @k=1,3,5,10 等标准指标。

    使用 evaluation_v2._compute_metrics_at_k 同一逻辑,这里抄一份以避免
    依赖 v2 内部函数可能的位置变动。
    """
    if not results:
        return {}
    n = len(results)
    out: dict = {"total": n, "hit_rate": float(sum(1 for r in results if r.hit) / n)}
    # MRR
    mrr_sum = 0.0
    for r in results:
        if r.rank is not None and r.rank >= 1:
            mrr_sum += 1.0 / r.rank
    out["mrr"] = round(mrr_sum / n, 4)
    return out


def _ndcg_at_k(results: list, k: int) -> float:
    import math

    if not results:
        return 0.0
    s = 0.0
    for r in results:
        discount = 1.0 / math.log2(r.rank + 1) if r.rank and 1 <= r.rank <= k else 0.0
        s += discount
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(results)) + 1))
    return s / ideal if ideal > 0 else 0.0


def _agg_v2_text_to_text(results: list) -> dict:
    """Aggregate v2 text→text results: hit_rate / MRR / NDCG / MAP per group."""
    grouped = _v2_to_grouped(results)
    out: dict = {}
    for g, rs in grouped.items():
        m = _v2_metrics(rs)
        m["ndcg@5"] = round(_ndcg_at_k(rs, 5), 4)
        m["ndcg@10"] = round(_ndcg_at_k(rs, 10), 4)
        out[g] = {**m, "total": len(rs)}
    return out


def _flat_payload(
    *,
    corpus_summary: dict,
    t2t_results: list,
    t2i_results: list,
    i2i_results: list,
    answer_results: list,
    lat_text: list[float],
    lat_t2i: list[float],
    lat_i2i: list[float],
    top_k: int,
    cases_path: str,
) -> dict:
    """Build the consolidated JSON report payload."""
    payload = {
        "version": "full_eval_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases_path": cases_path,
        "top_k": top_k,
        "corpus": corpus_summary,
        "retrieval": {
            "text_to_text": {
                "groups": _agg_v2_text_to_text(t2t_results),
                "per_query": [
                    {
                        "query": r.query,
                        "expected": r.qrels,
                        "actual": r.actual_document_ids,
                        "hit": r.hit,
                        "rank": r.rank,
                        "group": r.group,
                    }
                    for r in t2t_results
                ],
            },
            "text_to_image": {
                "total": len(t2i_results),
                "hit_rate": float(sum(1 for r in t2i_results if r.hit) / max(len(t2i_results), 1)),
                "per_query": [
                    {
                        "query": r.query,
                        "expected": r.qrels,
                        "actual": r.actual_document_ids,
                        "hit": r.hit,
                        "rank": r.rank,
                        "group": r.group,
                    }
                    for r in t2i_results
                ],
            },
            "image_to_image": {
                "total": len(i2i_results),
                "hit_rate": float(sum(1 for r in i2i_results if r.hit) / max(len(i2i_results), 1)),
                "per_query": [
                    {
                        "query": r.query,
                        "expected": r.qrels,
                        "actual": r.actual_document_ids,
                        "hit": r.hit,
                        "rank": r.rank,
                        "group": r.group,
                    }
                    for r in i2i_results
                ],
            },
        },
        "answer_quality": _aggregate_answer_metrics(answer_results),
        "latency_ms": {
            "text_to_text": _latency_block(lat_text),
            "text_to_image": _latency_block(lat_t2i),
            "image_to_image": _latency_block(lat_i2i),
        },
    }
    return payload


def _markdown_report(payload: dict) -> str:
    """Render the consolidated payload as a Markdown summary."""
    lines: list[str] = []
    lines.append("# mm-asset-rag 全面评测报告")
    lines.append("")
    lines.append(f"- 生成时间: `{payload['generated_at']}`")
    lines.append(f"- 评测 cases: `{payload['cases_path']}`")
    lines.append(f"- top_k: `{payload['top_k']}`")
    lines.append("")
    lines.append("## 1. 语料")
    corpus = payload["corpus"]
    lines.append(f"- 总文件: **{corpus['total_files']}**")
    lines.append(f"- 总 chunks (text): **{corpus['text_chunks']}**")
    lines.append(f"- 总 image vectors: **{corpus['image_vectors']}**")
    if corpus.get("by_type"):
        lines.append("- 按类型:")
        for t, n in sorted(corpus["by_type"].items(), key=lambda x: -x[1]):
            lines.append(f"  - `{t}`: {n}")
    lines.append("")

    lines.append("## 2. 检索指标 (业内标准)")
    lines.append("")
    lines.append("### 2.1 text→text (v2 全 group)")
    t2t = payload["retrieval"]["text_to_text"]["groups"]
    if t2t:
        lines.append("| group | total | hit_rate | MRR | NDCG@5 | NDCG@10 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for g, m in sorted(t2t.items()):
            lines.append(
                f"| {g} | {m['total']} | {m.get('hit_rate', 0):.3f} | "
                f"{m.get('mrr', 0):.3f} | {m.get('ndcg@5', 0):.3f} | "
                f"{m.get('ndcg@10', 0):.3f} |"
            )
    else:
        lines.append("(无 text→text 结果)")
    lines.append("")

    lines.append("### 2.2 text→image")
    t2i = payload["retrieval"]["text_to_image"]
    lines.append(f"- total: {t2i['total']}")
    lines.append(f"- hit_rate@5: **{t2i['hit_rate']:.3f}**")
    lines.append("")

    lines.append("### 2.3 image→image")
    i2i = payload["retrieval"]["image_to_image"]
    lines.append(f"- total: {i2i['total']}")
    lines.append(f"- hit_rate@5: **{i2i['hit_rate']:.3f}**")
    lines.append("")

    lines.append("## 3. answer-quality (LLM-judge)")
    aq = payload["answer_quality"]
    lines.append("")
    lines.append("### 3.1 整体")
    lines.append(f"- answer_source breakdown: `{aq.get('summary', {}).get('answer_sources')}`")
    metrics_all = aq.get("metrics", {}).get("all", {})
    if metrics_all:
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        for k, v in metrics_all.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### 3.2 per-group")
    per_group = aq.get("groups", {})
    if per_group:
        lines.append(
            "| group | total | coverage | citation_p | citation_r | citation_present | faithfulness | skipped |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for g, m in sorted(per_group.items()):
            lines.append(
                f"| {g} | {m.get('total', 0)} | "
                f"{m.get('coverage_mean', 0):.3f} | "
                f"{m.get('citation_precision_mean', 0):.3f} | "
                f"{m.get('citation_recall_mean', 0):.3f} | "
                f"{m.get('citation_present_rate', 0):.3f} | "
                f"{m.get('faithfulness_mean', 0) if m.get('faithfulness_mean') is not None else 'n/a'} | "
                f"{m.get('faithfulness_skipped', 0)} |"
            )
    lines.append("")

    lines.append("## 4. 检索延迟 (毫秒)")
    lat = payload["latency_ms"]
    lines.append("")
    lines.append("| route | count | mean | p50 | p95 | p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for route, k in [
        ("text_to_text", "text_to_text"),
        ("text_to_image", "text_to_image"),
        ("image_to_image", "image_to_image"),
    ]:
        b = lat[k]
        lines.append(
            f"| {route} | {b['count']} | {b['mean_ms']} | {b['p50_ms']} | {b['p95_ms']} | {b['p99_ms']} |"
        )
    lines.append("")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────


def _corpus_summary() -> dict:
    """Snapshot what's currently in the local mm_asset_rag home."""
    home = Path(os.environ.get("MM_ASSET_RAG_HOME", Path.home() / ".mm_asset_rag"))
    asset_index = home / "asset_index.jsonl"
    documents = home / "documents.jsonl"
    total = 0
    by_type: dict[str, int] = {}
    if asset_index.exists():
        for line in asset_index.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("deleted"):
                continue
            total += 1
            t = rec.get("source_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
    text_chunks = 0
    image_vectors = 0
    if documents.exists():
        for line in documents.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source_type") == "image":
                image_vectors += 1
            else:
                text_chunks += 1
    return {
        "home": str(home),
        "total_files": total,
        "text_chunks": text_chunks,
        "image_vectors": image_vectors,
        "by_type": by_type,
    }


def main() -> None:
    for name in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "https_proxy",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(name, None)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="examples/eval_cases_chapter11_v2.json",
        help="Path to v2 case file (must contain text→text + text→image + image→image groups).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default="default")
    parser.add_argument("--principal", default="eval")
    parser.add_argument(
        "--max-judge-cases",
        type=int,
        default=12,
        help="Cap LLM-judge calls per run (cost guard).",
    )
    parser.add_argument(
        "--skip-answer",
        action="store_true",
        help="Skip the answer-quality eval (saves LLM cost).",
    )
    parser.add_argument(
        "--skip-i2i",
        action="store_true",
        help="Skip image→image eval (faster).",
    )
    args = parser.parse_args()

    cases_path = args.cases
    top_k = args.top_k

    print(f"[full_eval] cases={cases_path} top_k={top_k}")
    corpus = _corpus_summary()
    print(
        f"[full_eval] corpus: {corpus['total_files']} files / "
        f"{corpus['text_chunks']} text chunks / {corpus['image_vectors']} image vectors"
    )

    # 1. text→text
    print("[full_eval] running text→text ...")
    lat_text: list[float] = []
    t2t = run_text_to_text(cases_path, top_k, lat_text, args.collection, args.principal)
    write_eval_report_v2({"text_to_text": t2t})
    print(f"[full_eval] text→text: {len(t2t)} cases, p50={_percentile(lat_text, 0.5):.1f}ms")

    # 2. text→image
    print("[full_eval] running text→image ...")
    lat_t2i: list[float] = []
    t2i = run_text_to_image(cases_path, top_k, lat_t2i, args.collection, args.principal)
    # write_eval_report_v2 expects {group: [V2Result]} — text_to_image goes
    # under its own key. Write a separate report merge.
    write_eval_report_v2({"text_to_text": t2t, "text_to_image": t2i})
    print(f"[full_eval] text→image: {len(t2i)} cases, p50={_percentile(lat_t2i, 0.5):.1f}ms")

    # 3. image→image
    i2i = []
    if not args.skip_i2i:
        print("[full_eval] running image→image ...")
        lat_i2i: list[float] = []
        i2i = run_image_to_image(cases_path, top_k, args.collection, args.principal)
        write_eval_report_v2({"text_to_text": t2t, "text_to_image": t2i, "image_to_image": i2i})
        print(f"[full_eval] image→image: {len(i2i)} cases, p50={_percentile(lat_i2i, 0.5):.1f}ms")
    else:
        lat_i2i = []

    # 4. answer-quality
    aq_results = []
    if not args.skip_answer:
        print("[full_eval] running answer-quality ...")
        aq_results = run_answer(
            cases_path, top_k, args.max_judge_cases, args.collection, args.principal
        )
        write_answer_eval_report(aq_results)
        print(f"[full_eval] answer-quality: {len(aq_results)} cases")

    # 5. consolidated report
    payload = _flat_payload(
        corpus_summary=corpus,
        t2t_results=t2t,
        t2i_results=t2i,
        i2i_results=i2i,
        answer_results=aq_results,
        lat_text=lat_text,
        lat_t2i=lat_t2i,
        lat_i2i=lat_i2i,
        top_k=top_k,
        cases_path=cases_path,
    )
    home = Path(os.environ.get("MM_ASSET_RAG_HOME", Path.home() / ".mm_asset_rag"))
    json_path = home / "eval_report_full.json"
    md_path = home / "eval_report_full.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    print(f"[full_eval] wrote {json_path}")
    print(f"[full_eval] wrote {md_path}")
    print()
    print(_markdown_report(payload))


if __name__ == "__main__":
    main()
