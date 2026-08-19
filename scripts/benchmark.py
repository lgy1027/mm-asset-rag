"""End-to-end performance + resource benchmark.

Puts numbers on the retrieval-latency / index-throughput claims so the next
person tuning ``Settings`` has data to act on. Talks only to the public API
(``hybrid_search``, ``build_default_text_embedder``, ``bm25_zh``,
``read_documents``) — no private ``_`` helpers — so it survives internal
refactors.

Phases:

1. **Retrieval latency** — ``hybrid_search`` round-trip, measured ``N_RUNS``
   times on a warm cache; reports mean / p50 / p95 / p99.
2. **Embedding throughput** — one batch per channel against the live
   ``documents.jsonl``; chunks/sec + a full-rebuild projection.
3. **Sequential QPS** — single-worker requests against ``hybrid_search``
   (Qdrant local-file mode is single-process; set ``QDRANT_URL`` for real
   concurrency), verifying every request still returns ``top_k`` hits.

Writes ``$MM_ASSET_RAG_HOME/benchmark_report.json`` + a stdout table.
Resource use (peak RSS, Qdrant index size) via ``resource.getrusage`` / ``du``.

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --top-k 5 --n-runs 50 --n-requests 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def _pct(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = max(0, min(len(s) - 1, int(q / 100 * len(s)) - 1))
    return s[idx]


def _stat_block(samples: list[float]) -> dict:
    if not samples:
        return {
            "n": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "n": len(samples),
        "mean_ms": statistics.mean(samples),
        "p50_ms": _pct(samples, 50),
        "p95_ms": _pct(samples, 95),
        "p99_ms": _pct(samples, 99),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _peak_rss_mb() -> float:
    import resource

    # ru_maxrss is KB on Linux, bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10 * 1024 * 1024:  # bytes → MB
        return rss / (1024 * 1024)
    return rss / 1024  # KB → MB


def _index_size_mb(indexes_dir: Path) -> float:
    """Approximate Qdrant local-storage footprint via ``du``."""
    try:
        out = subprocess.run(
            ["du", "-sm", str(indexes_dir / "qdrant")],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.split()[0])
    except Exception:
        return 0.0


def _bench(name: str, fn, n_runs: int) -> dict:
    """Time ``fn()`` ``n_runs`` times on a warm cache."""
    samples_ms: list[float] = []
    for _ in range(min(3, n_runs)):
        fn()
    for _ in range(n_runs):
        t = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t) * 1000)
    block = _stat_block(samples_ms)
    block["name"] = name
    return block


_QUERIES = [
    "BERT",
    "Mona Lisa painting",
    "深度学习",
    "transformer attention",
    "stable diffusion",
    "fish",
    "butterfly",
]


def phase1_latency(n_runs: int, top_k: int) -> dict:
    """``hybrid_search`` end-to-end latency distribution on a warm cache."""
    from mm_asset_rag.retrieval import hybrid_search

    q = _QUERIES[0]
    for _ in range(3):  # warm imports / caches
        hybrid_search(q, top_k=top_k)
    block = _bench(
        "hybrid_search (dense+BM25+BM25-zh, RRF, rerank)",
        lambda: hybrid_search(q, top_k=top_k),
        n_runs,
    )
    return block


def phase2_throughput(batch_size: int = 16) -> dict:
    """Per-channel embedding throughput against the live ``documents.jsonl``."""
    from mm_asset_rag.bm25_zh import build_bm25_zh_index
    from mm_asset_rag.document_store import read_documents
    from mm_asset_rag.embedders.text_embedder import build_default_text_embedder

    docs = read_documents()
    if not docs:
        return {"name": "embed_throughput", "skipped": True, "reason": "no documents.jsonl"}

    sample = [d.text for d in docs[:batch_size]]
    total_chunks = len(docs)
    embedder = build_default_text_embedder()

    t = time.perf_counter()
    embedder.embed_batch(sample)
    dense_elapsed = time.perf_counter() - t
    dense_cps = batch_size / dense_elapsed if dense_elapsed > 0 else 0.0

    subset = docs[: min(len(docs), 256)]
    t = time.perf_counter()
    build_bm25_zh_index(subset)
    bzh_elapsed = time.perf_counter() - t
    bzh_cps = len(subset) / bzh_elapsed if bzh_elapsed > 0 else 0.0

    avg_embed = (dense_cps + bzh_cps) / 2
    projection = total_chunks / avg_embed if avg_embed > 0 else 0.0

    return {
        "name": "embed_throughput (one batch per channel)",
        "batch_size": batch_size,
        "total_chunks": total_chunks,
        "channels": {
            "dense_embed": {
                "ms_per_batch": round(dense_elapsed * 1000, 1),
                "chunks_per_sec": round(dense_cps, 1),
            },
            "bm25_zh (jieba + Okapi)": {
                "ms_per_batch": round(bzh_elapsed * 1000, 1),
                "chunks_per_sec": round(bzh_cps, 1),
            },
        },
        "projected_full_rebuild_seconds": round(projection, 1),
    }


def phase3_qps(n_concurrent: int, top_k: int, n_requests: int) -> dict:
    """Sequential QPS — the honest local-mode ceiling."""
    from mm_asset_rag.retrieval import hybrid_search

    bench_queries = [_QUERIES[i % len(_QUERIES)] for i in range(n_requests)]

    samples: list[float] = []
    result_lengths: list[int] = []
    wall_start = time.perf_counter()
    for q in bench_queries:
        t = time.perf_counter()
        hits = hybrid_search(q, top_k=top_k)
        samples.append((time.perf_counter() - t) * 1000)
        result_lengths.append(len(hits))
    wall_total = time.perf_counter() - wall_start

    return {
        "name": "sequential QPS (local Qdrant baseline)",
        "n_concurrent_requested": n_concurrent,
        "n_concurrent_actual": 1,
        "n_requests": n_requests,
        "wall_seconds": round(wall_total, 3),
        "qps": round(n_requests / wall_total, 2) if wall_total > 0 else 0.0,
        "latency": _stat_block(samples),
        "result_lengths": {
            "min": min(result_lengths) if result_lengths else 0,
            "max": max(result_lengths) if result_lengths else 0,
            "all_top_k": all(n == top_k for n in result_lengths),
        },
        "concurrency_note": (
            "Qdrant local-file mode is single-process (a single .lock per "
            "indexes/qdrant directory). Set QDRANT_URL to a Qdrant server to "
            "run multi-worker concurrent QPS — the numbers above are the "
            "per-request ceiling, not the cluster ceiling."
        ),
    }


def render_report(report: dict) -> str:
    lines = ["=" * 78, "mm-asset-rag benchmark", "=" * 78, ""]

    p1 = report["phase1_latency"]
    lines.append("Retrieval latency (warm cache, hybrid_search end-to-end):")
    lines.append(f"  {p1['name']}")
    lines.append(
        f"    mean={p1['mean_ms']:.1f}ms p50={p1['p50_ms']:.1f}ms "
        f"p95={p1['p95_ms']:.1f}ms p99={p1['p99_ms']:.1f}ms n={p1['n']}"
    )
    lines.append("")

    lines.append("Embedding throughput (one batch per channel):")
    p2 = report["phase2_throughput"]
    if p2.get("skipped"):
        lines.append(f"  skipped: {p2.get('reason')}")
    else:
        lines.append(f"  total_chunks={p2['total_chunks']} batch_size={p2['batch_size']}")
        for ch, m in p2["channels"].items():
            lines.append(
                f"    {ch:28s} {m['ms_per_batch']:>7.1f}ms/batch  "
                f"{m['chunks_per_sec']:>6.1f} chunks/s"
            )
        lines.append(f"  projected full rebuild: ~{p2['projected_full_rebuild_seconds']:.1f}s")
    lines.append("")

    lines.append("QPS (sequential, Qdrant local-mode baseline):")
    p3 = report["phase3_qps"]
    lines.append(
        f"  workers={p3['n_concurrent_actual']} requests={p3['n_requests']} "
        f"wall={p3['wall_seconds']}s qps={p3['qps']}"
    )
    lines.append(
        f"  latency: mean={p3['latency']['mean_ms']:.1f}ms "
        f"p50={p3['latency']['p50_ms']:.1f}ms "
        f"p95={p3['latency']['p95_ms']:.1f}ms "
        f"p99={p3['latency']['p99_ms']:.1f}ms"
    )
    lines.append(
        f"  result_lengths: min={p3['result_lengths']['min']} "
        f"max={p3['result_lengths']['max']} "
        f"all_top_k={p3['result_lengths']['all_top_k']}"
    )
    lines.append("")

    lines.append("Resource use:")
    res = report["resources"]
    lines.append(f"  peak RSS: {res['peak_rss_mb']:.1f} MB")
    lines.append(f"  Qdrant index size: {res['qdrant_index_mb']:.1f} MB")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-runs", type=int, default=20, help="Repetitions for phase 1 latency (default 20)."
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=4,
        help="Requested workers for phase 3 (default 4; local Qdrant uses 1).",
    )
    parser.add_argument(
        "--n-requests", type=int, default=10, help="Total requests fired in phase 3 (default 10)."
    )
    parser.add_argument("--top-k", type=int, default=5, help="top_k for hybrid_search (default 5).")
    parser.add_argument(
        "--skip-phase2", action="store_true", help="Skip the embed-throughput pass."
    )
    args = parser.parse_args()

    home = Path(os.environ.get("MM_ASSET_RAG_HOME", str(Path.home() / ".mm_asset_rag")))
    print(
        f"Running benchmark (n_runs={args.n_runs}, n_concurrent={args.n_concurrent}, "
        f"n_requests={args.n_requests}, top_k={args.top_k})..."
    )

    print("\n[phase 1] retrieval latency...")
    phase1 = phase1_latency(args.n_runs, args.top_k)

    phase2: dict = {"name": "embed_throughput", "skipped": True}
    if not args.skip_phase2:
        print("\n[phase 2] embed throughput...")
        phase2 = phase2_throughput()

    print("\n[phase 3] sequential QPS...")
    phase3 = phase3_qps(args.n_concurrent, args.top_k, args.n_requests)

    print("\n[resources] collecting...")
    resources = {
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "qdrant_index_mb": round(_index_size_mb(home / "indexes"), 1),
    }

    report = {
        "config": {
            "n_runs": args.n_runs,
            "n_concurrent": args.n_concurrent,
            "n_requests": args.n_requests,
            "top_k": args.top_k,
        },
        "phase1_latency": phase1,
        "phase2_throughput": phase2,
        "phase3_qps": phase3,
        "resources": resources,
    }

    out = home / "benchmark_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(render_report(report))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
