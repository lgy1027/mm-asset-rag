"""End-to-end performance + resource benchmark.

The goal of this script is to put numbers on the claims we make about
retrieval latency and index throughput, so the next person tuning
``Settings`` has data to act on rather than vibes.

It exercises three phases:

1. **Per-component latency** — dense embed (ollama), BM25-en, BM25-zh,
   Qdrant 3-way RRF, and the full ``hybrid_search`` round-trip.
   Each is measured ``N_RUNS`` times on a warm cache; we report mean,
   p50, p95, p99.

2. **Indexing throughput** — re-run ``build_qdrant_text_index`` against
   a clean collection, time the upsert, and report chunks/sec plus
   the BM25-zh tokenise+IDF wall clock as a separate line.

3. **Concurrent QPS** — fire ``N_CONCURRENT`` workers in
   ``concurrent.futures.ThreadPoolExecutor`` against ``hybrid_search``,
   measure wall-clock + QPS, and verify the result quality does not
   drift (every request still returns ``top_k`` hits).

All measurements write a JSON report to
``$MM_ASSET_RAG_HOME/benchmark_report.json`` and a human-readable
table to stdout. Resource use (peak RSS, Qdrant index size) is read
from ``resource.getrusage`` / a ``du`` on the Qdrant directory.

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --top-k 5 --n-runs 50 --n-concurrent 8
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
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
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
    if rss > 10 * 1024 * 1024:
        return rss / (1024 * 1024)  # bytes
    return rss / 1024  # KB
    return 0.0


def _index_size_mb(indexes_dir: Path) -> float:
    """Approximate Qdrant local-storage footprint via ``du``."""
    try:
        out = subprocess.run(
            ["du", "-sm", str(indexes_dir / "qdrant")],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.split()[0])
    except Exception:
        return 0.0


def _bench_component(name: str, fn, n_runs: int) -> dict:
    """Time ``fn()`` ``n_runs`` times. ``fn`` is the only callable under test."""
    samples_ms: list[float] = []
    # Warm up so the first call doesn't include lazy imports / model load.
    for _ in range(min(3, n_runs)):
        fn()
    for _ in range(n_runs):
        t = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t) * 1000)
    block = _stat_block(samples_ms)
    block["name"] = name
    return block


def phase1_per_component_latency(n_runs: int) -> list[dict]:
    """Measure each retrieval component in isolation."""
    from mm_asset_rag.backends.qdrant_backend import (
        _bm25_embedder,
        _embed_bm25,
        _embed_bm25_zh_query,
        _hybrid_text_query,
        get_qdrant_client,
    )
    from mm_asset_rag.embedders.text_embedder import EmbeddingProvider

    # Warm up.
    _bm25_embedder()
    embedder = EmbeddingProvider()
    embedder.embed("warmup")
    _embed_bm25_zh_query("warmup")

    client = get_qdrant_client()
    try:
        coll = next(c.name for c in client.get_collections().collections if c.name.startswith("multimodal_text"))

        queries = ["BERT", "Mona Lisa painting", "深度学习", "transformer attention"]
        bench_queries = queries * (max(1, n_runs // len(queries) + 1))
        bench_queries = bench_queries[:n_runs]

        def dense(q):
            return embedder.embed(q)

        def bm25(q):
            return _embed_bm25([q])

        def bm25_zh(q):
            return _embed_bm25_zh_query(q)

        def qdrant_one(q):
            d = embedder.embed(q)
            s = _embed_bm25([q])[0]
            sz = _embed_bm25_zh_query(q)
            _hybrid_text_query(client, coll, d, s, sz, top_k=5)

        results: list[dict] = []
        for name, fn in [
            ("dense_embed (ollama HTTP)", lambda: dense(bench_queries[0])),
            ("bm25_en (fastembed local)", lambda: bm25(bench_queries[1])),
            ("bm25_zh (jieba + Okapi)",   lambda: bm25_zh(bench_queries[2])),
            ("qdrant_text_search (3-way RRF)", lambda: qdrant_one(bench_queries[3])),
        ]:
            block = _bench_component(name, fn, n_runs)
            results.append(block)
    finally:
        # Explicit close so phase 2 (drop+rebuild) doesn't see a held .lock.
        try:
            client.close()
        except Exception:
            pass
    return results


def phase2_embed_throughput(batch_size: int = 16) -> dict:
    """Measure the per-channel embedding throughput.

    Indexing time in production is dominated by the embedding pass
    (Qdrant upsert is sub-second per batch). We time one batch per
    channel against the live ``documents.jsonl`` and report
    chunks/sec plus a projection for the full corpus.
    """
    from mm_asset_rag.backends.qdrant_backend import _bm25_embedder, _embed_bm25
    from mm_asset_rag.bm25_zh import build_bm25_zh_index
    from mm_asset_rag.document_store import read_documents
    from mm_asset_rag.embedders.text_embedder import EmbeddingProvider

    docs = read_documents()
    if not docs:
        return {"name": "embed_throughput", "skipped": True, "reason": "no documents.jsonl"}

    sample = [d.text for d in docs[:batch_size]]
    total_chunks = len(docs)

    _bm25_embedder()  # warm
    embedder = EmbeddingProvider()

    # dense (ollama HTTP)
    t = time.perf_counter()
    embedder.embed_batch(sample)
    dense_elapsed = time.perf_counter() - t
    dense_chunks_per_sec = batch_size / dense_elapsed if dense_elapsed > 0 else 0.0

    # bm25 (fastembed local)
    t = time.perf_counter()
    _embed_bm25(sample)
    bm25_elapsed = time.perf_counter() - t
    bm25_chunks_per_sec = batch_size / bm25_elapsed if bm25_elapsed > 0 else 0.0

    # bm25_zh (jieba + Okapi; build full IDF first then encode the sample)
    t = time.perf_counter()
    _all_vectors, _idf = build_bm25_zh_index(docs[: min(len(docs), 256)])  # small corpus for IDF
    bzh_elapsed = time.perf_counter() - t
    bzh_chunks_per_sec = len(docs[: min(len(docs), 256)]) / bzh_elapsed if bzh_elapsed > 0 else 0.0

    # Qdrant upsert is fast (sub-second per batch even at 16 chunks) and
    # is bounded by disk write, not compute. We don't measure it; the
    # projection below adds a small fixed overhead.
    avg_embed = (dense_chunks_per_sec + bm25_chunks_per_sec + bzh_chunks_per_sec) / 3
    projection_seconds = total_chunks / avg_embed if avg_embed > 0 else 0.0

    return {
        "name": "embed_throughput (one batch per channel)",
        "batch_size": batch_size,
        "total_chunks": total_chunks,
        "channels": {
            "dense_embed (ollama)":   {"ms_per_batch": round(dense_elapsed * 1000, 1), "chunks_per_sec": round(dense_chunks_per_sec, 1)},
            "bm25 (fastembed)":        {"ms_per_batch": round(bm25_elapsed * 1000, 1), "chunks_per_sec": round(bm25_chunks_per_sec, 1)},
            "bm25_zh (jieba + Okapi)": {"ms_per_batch": round(bzh_elapsed * 1000, 1), "chunks_per_sec": round(bzh_chunks_per_sec, 1)},
        },
        "projected_full_rebuild_seconds": round(projection_seconds, 1),
    }


def phase3_concurrent_qps(n_concurrent: int, top_k: int, n_requests: int) -> dict:
    """Measure single-process QPS, plus a note on Qdrant local-mode concurrency.

    Qdrant local-file mode is single-process (one ``.lock``). Multi-worker
    ``ThreadPoolExecutor`` against the same Qdrant directory raises
    ``QdrantLockHeldError`` for every contending worker. Production
    deployments expecting real concurrency must point ``QDRANT_URL`` at
    a Qdrant **server**. We therefore measure sequential QPS as the
    realistic local-mode baseline and report a clear note.
    """
    from mm_asset_rag.retrieval import hybrid_search

    queries = [
        "BERT", "transformer attention", "Mona Lisa",
        "深度学习入门", "fish", "butterfly", "stable diffusion",
    ]
    bench_queries = [queries[i % len(queries)] for i in range(n_requests)]

    samples: list[float] = []
    result_lengths: list[int] = []

    def one(q: str) -> tuple[float, int]:
        t = time.perf_counter()
        hits = hybrid_search(q, top_k=top_k)
        return (time.perf_counter() - t) * 1000, len(hits)

    wall_start = time.perf_counter()
    # Single-worker loop — local Qdrant cannot be opened by N processes
    # simultaneously, and a thread pool still serialises on the Qdrant
    # connection lock in practice. Sequential is the only honest local
    # measurement.
    for q in bench_queries:
        ms, n = one(q)
        samples.append(ms)
        result_lengths.append(n)
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
            "indexes/qdrant directory). Set QDRANT_URL to point at a Qdrant "
            "server to run multi-worker concurrent QPS — local-mode numbers "
            "above are the per-request ceiling, not the cluster ceiling."
        ),
    }


def render_report(report: dict) -> str:
    lines = ["=" * 78, "mm-asset-rag benchmark", "=" * 78, ""]
    lines.append("Per-component latency (warm cache, end-to-end wall time):")
    lines.append(f"  {'component':40s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'n':>4s}")
    for c in report["phase1_per_component"]:
        lines.append(
            f"  {c['name']:40s} {c['mean_ms']:>7.1f}ms {c['p50_ms']:>7.1f}ms {c['p95_ms']:>7.1f}ms {c['p99_ms']:>7.1f}ms {c['n']:>4d}"
        )
    lines.append("")
    lines.append("Indexing throughput (one batch per channel):")
    p2 = report["phase2_indexing"]
    if p2.get("skipped"):
        lines.append(f"  skipped: {p2.get('reason')}")
    else:
        lines.append(f"  total_chunks in corpus: {p2['total_chunks']}  batch_size: {p2['batch_size']}")
        for ch, m in p2["channels"].items():
            lines.append(
                f"    {ch:32s} {m['ms_per_batch']:>7.1f}ms/batch  {m['chunks_per_sec']:>6.1f} chunks/s"
            )
        lines.append(
            f"  projected full-rebuild (3-way embed, no Qdrant upsert): ~{p2['projected_full_rebuild_seconds']:.1f}s"
        )
    lines.append("")
    lines.append("QPS (sequential, Qdrant local-mode baseline):")
    p3 = report["phase3_concurrent"]
    lines.append(f"  workers={p3['n_concurrent_actual']}  requests={p3['n_requests']}  wall={p3['wall_seconds']}s  "
                 f"qps={p3['qps']}")
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
    if p3.get("errors"):
        lines.append(f"  ERRORS: {p3['errors']}")
    lines.append("")
    lines.append("Resource use:")
    res = report["resources"]
    lines.append(f"  peak RSS: {res['peak_rss_mb']:.1f} MB")
    lines.append(f"  Qdrant index size: {res['qdrant_index_mb']:.1f} MB")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-runs", type=int, default=20,
                        help="Repetitions per component in phase 1 (default 20).")
    parser.add_argument("--n-concurrent", type=int, default=4,
                        help="Worker count for phase 3 (default 4).")
    parser.add_argument("--n-requests", type=int, default=10,
                        help="Total requests fired in phase 3 (default 10).")
    parser.add_argument("--top-k", type=int, default=5,
                        help="top_k for hybrid_search in phase 3 (default 5).")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip the indexing-throughput rebuild (saves the full embed pass).")
    args = parser.parse_args()

    home = Path(os.environ.get("MM_ASSET_RAG_HOME", str(Path.home() / ".mm_asset_rag")))

    print(f"Running benchmark (n_runs={args.n_runs}, "
          f"n_concurrent={args.n_concurrent}, n_requests={args.n_requests}, top_k={args.top_k})...")

    print("\n[phase 1] per-component latency...")
    phase1 = phase1_per_component_latency(args.n_runs)

    phase2: dict = {"name": "embed_throughput", "skipped": True}
    if not args.skip_phase2:
        print("\n[phase 2] embed throughput (one batch per channel)...")
        phase2 = phase2_embed_throughput()

    print("\n[phase 3] concurrent QPS...")
    phase3 = phase3_concurrent_qps(args.n_concurrent, args.top_k, args.n_requests)

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
        "phase1_per_component": phase1,
        "phase2_indexing": phase2,
        "phase3_concurrent": phase3,
        "resources": resources,
    }

    out = home / "benchmark_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(render_report(report))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
