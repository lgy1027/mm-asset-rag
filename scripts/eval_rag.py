"""Standard RAG evaluation: hit_rate@k, MRR, per-category breakdown.

Run from the repo root after a successful ``mmrag index``:

    python scripts/eval_rag.py --home $MM_ASSET_RAG_HOME

Eval cases are written as a flat list of dicts with these fields:

- ``query``              str, the user-supplied query
- ``expected_ids``       list[str], asset_ids that should appear in the
                         top-k retrieval results (any of them counts)
- ``category``           one of "keyword", "phrase", "semantic_zh",
                         "semantic_en", "image_search", "mixed"
- ``mode``               which ``mmrag search --mode`` to use
                         ("text", "text-to-image", "image-to-image",
                         "hybrid"); default "hybrid"

Outputs:

1. Per-case table (printed to stdout)
2. Per-category aggregate: hit_rate@k, MRR
3. Overall hit_rate@k and MRR
4. Writes JSON to ``$MM_ASSET_RAG_HOME/eval_report_full.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Eval set: covers 5 query categories over the 30-PDF + 184-photo set.
EVAL_CASES: list[dict] = [
    # ── Exact keyword (BM25 strength) ─────────────────────────────────
    {"query": "BERT",                     "expected_ids": ["bert"],                   "category": "keyword",      "mode": "text"},
    {"query": "LoRA",                     "expected_ids": ["lora"],                   "category": "keyword",      "mode": "text"},
    {"query": "ViT",                      "expected_ids": ["vit"],                    "category": "keyword",      "mode": "text"},
    {"query": "DETR",                     "expected_ids": ["detr"],                   "category": "keyword",      "mode": "text"},
    {"query": "LayoutLM",                 "expected_ids": ["layoutlm"],               "category": "keyword",      "mode": "text"},
    {"query": "AlexNet",                  "expected_ids": ["alexnet"],                "category": "keyword",      "mode": "text"},
    {"query": "ResNet",                   "expected_ids": ["resnet"],                 "category": "keyword",      "mode": "text"},
    {"query": "DDPM",                     "expected_ids": ["ddpm"],                   "category": "keyword",      "mode": "text"},
    {"query": "Word2Vec",                 "expected_ids": ["word2vec"],               "category": "keyword",      "mode": "text"},
    {"query": "U-Net",                    "expected_ids": ["unet"],                   "category": "keyword",      "mode": "text"},

    # ── Exact phrase (BM25 + dense) ────────────────────────────────────
    {"query": "retrieval augmented generation", "expected_ids": ["retrieval_augmented_generation"], "category": "phrase", "mode": "text"},
    {"query": "stable diffusion",          "expected_ids": ["stable_diffusion"],       "category": "phrase",       "mode": "text"},
    {"query": "segment anything",          "expected_ids": ["segment_anything"],       "category": "phrase",       "mode": "text"},
    {"query": "generative adversarial",    "expected_ids": ["gan"],                     "category": "phrase",       "mode": "text"},

    # ── Semantic Chinese query (dense strength) ───────────────────────
    {"query": "讲 transformer 的论文",            "expected_ids": ["attention_is_all_you_need"], "category": "semantic_zh", "mode": "text"},
    {"query": "图像生成扩散模型",                  "expected_ids": ["ddpm", "stable_diffusion", "pix2pix"], "category": "semantic_zh", "mode": "text"},
    {"query": "小模型移动端推理",                  "expected_ids": ["mobilenet", "mobilenetv2", "efficientnet"], "category": "semantic_zh", "mode": "text"},
    {"query": "CLIP 图文对齐",                     "expected_ids": ["clip"],                   "category": "semantic_zh", "mode": "text"},
    {"query": "文档版面理解 OCR",                  "expected_ids": ["layoutlm"],               "category": "semantic_zh", "mode": "text"},

    # ── Semantic English query (dense) ─────────────────────────────────
    {"query": "text to image generative model",    "expected_ids": ["stable_diffusion", "pix2pix"], "category": "semantic_en", "mode": "text"},
    {"query": "few-shot language model",          "expected_ids": ["gpt3", "llama"],       "category": "semantic_en", "mode": "text"},
    {"query": "real time object detection",       "expected_ids": ["yolo", "ssd"],          "category": "semantic_en", "mode": "text"},

    # ── Image search (CLIP) ────────────────────────────────────────────
    {"query": "fish",          "expected_ids": ["picsum_1018", "img_03_opencv_sample_data_happyfish_jpg"], "category": "image_search", "mode": "text-to-image"},
    {"query": "logo",          "expected_ids": ["img_04_opencv_sample_data_linuxlogo_jpg", "img_05_opencv_sample_data_windowslogo_jpg"], "category": "image_search", "mode": "text-to-image"},
    {"query": "butterfly",     "expected_ids": ["img_20_opencv_sample_data_butterfly_jpg"], "category": "image_search", "mode": "text-to-image"},

    # ── Mixed (text + image in same query) ──────────────────────────────
    {"query": "butterfly", "expected_ids": ["img_20_opencv_sample_data_butterfly_jpg"], "category": "mixed", "mode": "hybrid"},
]


def _prefix_hit(expected: list[str], actual_ids: list[str]) -> bool:
    """Prefix-tolerant substring match against any expected id."""
    return any(exp in act or act in exp for exp in expected for act in actual_ids)


def _reciprocal_rank(expected: list[str], actual_ids: list[str]) -> float:
    """1 / rank of the first matching asset; 0 if none match."""
    for rank, act in enumerate(actual_ids, start=1):
        if any(exp in act or act in exp for exp in expected):
            return 1.0 / rank
    return 0.0


def evaluate_case(case: dict, top_k: int) -> dict:
    """Run a single eval case and return its result dict."""
    query = case["query"]
    expected = case["expected_ids"]
    mode = case.get("mode", "hybrid")

    if mode == "text-to-image":
        hits = _text_to_image_search(query, top_k)
    elif mode == "image-to-image":
        hits = []  # needs an actual image; not covered in this batch
    else:  # "text" or "hybrid" both go through hybrid_search (hybrid is a superset)
        hits = _hybrid_search(query, top_k)

    actual_ids = [hit.asset_id for hit in hits]
    rr = _reciprocal_rank(expected, actual_ids)
    hit = _prefix_hit(expected, actual_ids) and rr > 0
    return {
        "query": query,
        "category": case["category"],
        "mode": mode,
        "expected_ids": expected,
        "actual_ids": actual_ids,
        "hit_at_k": hit,
        "reciprocal_rank": rr,
    }


# Local imports kept lazy so running ``--help`` doesn't require a live
# Qdrant collection.
def _hybrid_search(query: str, top_k: int):
    from mm_asset_rag.retrieval import hybrid_search as _hs
    return _hs(query, top_k=top_k)


def _text_to_image_search(query: str, top_k: int):
    from mm_asset_rag.qdrant_store import qdrant_text_to_image_search
    return qdrant_text_to_image_search(query, top_k=top_k)


def aggregate(results: list[dict], top_k: int) -> dict:
    """Compute overall + per-category hit rate @ k and MRR."""
    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    def _metrics(group: list[dict]) -> dict:
        n = len(group)
        if n == 0:
            return {"hit_rate": 0.0, "mrr": 0.0, "n": 0}
        hr = sum(1 for r in group if r["hit_at_k"]) / n
        mrr = sum(r["reciprocal_rank"] for r in group) / n
        return {"hit_rate": round(hr, 3), "mrr": round(mrr, 3), "n": n}

    per_category = {cat: _metrics(group) for cat, group in sorted(by_category.items())}
    overall = _metrics(results)
    return {"top_k": top_k, "overall": overall, "per_category": per_category}


def render_report(report: dict, results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"RAG evaluation  ·  top_k={report['top_k']}  ·  cases={sum(c['n'] for c in report['per_category'].values())}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"  overall      hit_rate={report['overall']['hit_rate']:.3f}   MRR={report['overall']['mrr']:.3f}   n={report['overall']['n']}")
    lines.append("")
    lines.append("  per category:")
    for cat, m in report["per_category"].items():
        lines.append(f"    {cat:18s}  hit_rate={m['hit_rate']:.3f}   MRR={m['mrr']:.3f}   n={m['n']}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("per-case details:")
    for r in results:
        mark = "✓" if r["hit_at_k"] else "✗"
        lines.append(f"  {mark} [{r['category']:13s}] {r['mode']:15s}  q={r['query']!r}")
        lines.append(f"      expected: {r['expected_ids']}")
        lines.append(f"      actual  : {r['actual_ids'][:5]}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--home", default=None, help="MM_ASSET_RAG_HOME override")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path; defaults to $MM_ASSET_RAG_HOME/eval_report_full.json",
    )
    args = parser.parse_args()

    if args.home:
        import os
        os.environ["MM_ASSET_RAG_HOME"] = args.home

    results = [evaluate_case(case, args.top_k) for case in EVAL_CASES]
    report = aggregate(results, args.top_k)
    print(render_report(report, results))

    output_path = Path(
        args.output
        or f"{__import__('os').environ['MM_ASSET_RAG_HOME']}/eval_report_full.json"
    )
    output_path.write_text(
        json.dumps({"summary": report, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
