"""Cross-scenario evaluation over the expanded sample set.

Companion to ``scripts/eval_rag.py``. Where eval_rag.py tests the
bundled 30-PDF ML benchmark (6 categories, 26 cases), this script
tests the 22 cross-scenario PDFs added by ``scripts/expand_corpus.py``
(Wikipedia EN/ZH, arXiv short papers, IRS forms, scan variants) with
2-3 ground-truth queries per PDF across 5 new categories.

Run after ``mmrag reindex --text-only``:

    export MM_ASSET_RAG_HOME=$HOME/.mm_asset_rag
    python scripts/eval_extended.py --top-k 5

Output: ``$MM_ASSET_RAG_HOME/eval_report_extended.json`` plus a
per-case + per-category table to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EVAL_CASES: list[dict] = [
    # ── Wikipedia EN (12 assets) ───────────────────────────────────────────
    # keyword
    {
        "query": "Mona Lisa painting",
        "expected_ids": ["wiki_en_monalisa"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Bicycle history",
        "expected_ids": ["wiki_en_bicycle"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Pizza origin Italy",
        "expected_ids": ["wiki_en_pizza"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Coca-Cola brand",
        "expected_ids": ["wiki_en_cocacola"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Piano musical instrument",
        "expected_ids": ["wiki_en_piano"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Mushroom biology",
        "expected_ids": ["wiki_en_mushroom"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Telescope astronomy",
        "expected_ids": ["wiki_en_telescope"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Photography",
        "expected_ids": ["wiki_en_photography"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "QR code two-dimensional barcode",
        "expected_ids": ["wiki_en_qrcode"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "PDF file format Adobe",
        "expected_ids": ["wiki_en_pdf"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Apollo 11 moon landing",
        "expected_ids": ["wiki_en_apollo11"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    {
        "query": "Sony corporation history",
        "expected_ids": ["wiki_en_sony"],
        "category": "wiki_en_kw",
        "mode": "text",
    },
    # semantic paraphrase — should still hit via dense
    {
        "query": "renaissance portrait with enigmatic smile",
        "expected_ids": ["wiki_en_monalisa"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    {
        "query": "human-powered two-wheeled transport",
        "expected_ids": ["wiki_en_bicycle"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    {
        "query": "Italian flatbread topped with tomato and cheese",
        "expected_ids": ["wiki_en_pizza"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    {
        "query": "Atlanta-based soft drink company",
        "expected_ids": ["wiki_en_cocacola"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    {
        "query": "88-key hammer-string instrument",
        "expected_ids": ["wiki_en_piano"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    {
        "query": "fruiting body of fungus",
        "expected_ids": ["wiki_en_mushroom"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    {
        "query": "NASA's first crewed lunar landing mission",
        "expected_ids": ["wiki_en_apollo11"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    {
        "query": "Japanese electronics conglomerate founded post-war",
        "expected_ids": ["wiki_en_sony"],
        "category": "wiki_en_sm",
        "mode": "text",
    },
    # ── Wikipedia ZH (4 assets) ────────────────────────────────────────────
    {
        "query": "北京 中国首都",
        "expected_ids": ["wiki_zh_beijing"],
        "category": "wiki_zh_kw",
        "mode": "text",
    },
    {
        "query": "肯德基 KFC 快餐",
        "expected_ids": ["wiki_zh_kfc"],
        "category": "wiki_zh_kw",
        "mode": "text",
    },
    {
        "query": "咖啡 饮品",
        "expected_ids": ["wiki_zh_coffee"],
        "category": "wiki_zh_kw",
        "mode": "text",
    },
    {
        "query": "大熊猫 保护动物",
        "expected_ids": ["wiki_zh_panda"],
        "category": "wiki_zh_kw",
        "mode": "text",
    },
    # Chinese semantic
    {
        "query": "中国的政治中心和文化古城",
        "expected_ids": ["wiki_zh_beijing"],
        "category": "wiki_zh_sm",
        "mode": "text",
    },
    {
        "query": "源自美国的炸鸡连锁餐厅",
        "expected_ids": ["wiki_zh_kfc"],
        "category": "wiki_zh_sm",
        "mode": "text",
    },
    {
        "query": "含咖啡因的热带植物种子饮品",
        "expected_ids": ["wiki_zh_coffee"],
        "category": "wiki_zh_sm",
        "mode": "text",
    },
    {
        "query": "中国特有的黑白熊科动物",
        "expected_ids": ["wiki_zh_panda"],
        "category": "wiki_zh_sm",
        "mode": "text",
    },
    # ── arXiv (2 assets) ───────────────────────────────────────────────────
    {
        "query": "bolometer thermodynamics economics",
        "expected_ids": ["arxiv_phys_thermo"],
        "category": "arxiv_kw",
        "mode": "text",
    },
    {
        "query": "spiking neuron soliton model",
        "expected_ids": ["arxiv_qbio_neuro"],
        "category": "arxiv_kw",
        "mode": "text",
    },
    # semantic
    {
        "query": "non-ML cross-disciplinary physics paper about money",
        "expected_ids": ["arxiv_phys_thermo"],
        "category": "arxiv_sm",
        "mode": "text",
    },
    {
        "query": "biological neuron firing dynamics paper",
        "expected_ids": ["arxiv_qbio_neuro"],
        "category": "arxiv_sm",
        "mode": "text",
    },
    # ── IRS forms (2 assets) ───────────────────────────────────────────────
    {
        "query": "W-9 taxpayer identification request",
        "expected_ids": ["irs_w9"],
        "category": "irs_kw",
        "mode": "text",
    },
    {
        "query": "W-4 employee withholding certificate",
        "expected_ids": ["irs_w4"],
        "category": "irs_kw",
        "mode": "text",
    },
    # semantic
    {
        "query": "US tax form a contractor fills out for a client",
        "expected_ids": ["irs_w9"],
        "category": "irs_sm",
        "mode": "text",
    },
    {
        "query": "US form employees submit to control payroll tax withholding",
        "expected_ids": ["irs_w4"],
        "category": "irs_sm",
        "mode": "text",
    },
    # ── Scan variants (2 assets) ───────────────────────────────────────────
    # Scan variants carry OCR-simulated text in this test rig (the test
    # is whether the retrieval pipeline treats image-only-PDF-style
    # content correctly once text exists). In a real deployment with a
    # true OCR backend, the same pipeline consumes OCR output directly.
    {
        "query": "scanned Mona Lisa article",
        "expected_ids": ["wiki_en_monalisa__scan"],
        "category": "scan_kw",
        "mode": "text",
    },
    {
        "query": "scanned pizza article",
        "expected_ids": ["wiki_en_pizza__scan"],
        "category": "scan_kw",
        "mode": "text",
    },
    {
        "query": "OCR-output page about Leonardo da Vinci portrait",
        "expected_ids": ["wiki_en_monalisa__scan"],
        "category": "scan_sm",
        "mode": "text",
    },
    # ── CLIP text-to-image (image_search category) ────────────────────────
    # Picsum + OpenCV sample images. CLIP embeds the image at parse
    # time; the retriever matches the text query against the image
    # collection. Re-runs require ``mmrag reindex --image-only``.
    {
        "query": "fish",
        "expected_ids": ["img_03_opencv_sample_data_happyfish_jpg", "picsum_1018"],
        "category": "image_search",
        "mode": "text-to-image",
    },
    {
        "query": "logo",
        "expected_ids": [
            "img_04_opencv_sample_data_linuxlogo_jpg",
            "img_05_opencv_sample_data_windowslogo_jpg",
        ],
        "category": "image_search",
        "mode": "text-to-image",
    },
    {
        "query": "butterfly",
        "expected_ids": ["img_20_opencv_sample_data_butterfly_jpg"],
        "category": "image_search",
        "mode": "text-to-image",
    },
    {
        "query": "happy fish swimming",
        "expected_ids": ["img_03_opencv_sample_data_happyfish_jpg"],
        "category": "image_search_sm",
        "mode": "text-to-image",
    },
    {
        "query": "open source operating system logo",
        "expected_ids": ["img_04_opencv_sample_data_linuxlogo_jpg"],
        "category": "image_search_sm",
        "mode": "text-to-image",
    },
    # ── VLM caption retrieval (placeholder; needs ENABLE_VLM=true re-parse) ─
    # Captions are written to ``captions/<asset_id>.json`` only when
    # ``ENABLE_VLM=true`` is set during ``mmrag parse``. The current
    # bundled set was parsed without VLM, so the caption file is empty
    # and the asset's text contribution comes from title + tags only.
    # To exercise this path, run:
    #     mmrag parse --pdf-parser pymupdf --enable-vlm=true --vlm-model gemma4:latest
    # then re-add cases here. Skipped for now.
    # {"query": "A Mona Lisa painting on a wall", "expected_ids": ["picsum_278", ...], "category": "vlm_sm", "mode": "text"},
]


def _prefix_hit(expected: list[str], actual_ids: list[str]) -> bool:
    return any(exp in act or act in exp for exp in expected for act in actual_ids)


def _reciprocal_rank(expected: list[str], actual_ids: list[str]) -> float:
    for rank, act in enumerate(actual_ids, start=1):
        if any(exp in act or act in exp for exp in expected):
            return 1.0 / rank
    return 0.0


def evaluate_case(case: dict, top_k: int) -> dict:
    from mm_asset_rag.retrieval import hybrid_search

    hits = hybrid_search(case["query"], top_k=top_k)
    actual_ids = [h.asset_id for h in hits]
    expected = case["expected_ids"]
    rr = _reciprocal_rank(expected, actual_ids)
    hit = _prefix_hit(expected, actual_ids) and rr > 0
    return {
        "query": case["query"],
        "category": case["category"],
        "mode": case.get("mode", "text"),
        "expected_ids": expected,
        "actual_ids": actual_ids,
        "hit_at_k": hit,
        "reciprocal_rank": rr,
    }


def aggregate(results: list[dict], top_k: int) -> dict:
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    def _metrics(group: list[dict]) -> dict:
        n = len(group)
        if n == 0:
            return {"hit_rate": 0.0, "mrr": 0.0, "n": 0}
        return {
            "hit_rate": round(sum(1 for r in group if r["hit_at_k"]) / n, 3),
            "mrr": round(sum(r["reciprocal_rank"] for r in group) / n, 3),
            "n": n,
        }

    per_cat = {c: _metrics(g) for c, g in sorted(by_cat.items())}
    return {"top_k": top_k, "overall": _metrics(results), "per_category": per_cat}


def render_report(report: dict, results: list[dict]) -> str:
    lines = ["=" * 78]
    lines.append(
        f"EXTENDED RAG eval · top_k={report['top_k']} · "
        f"cases={sum(c['n'] for c in report['per_category'].values())}"
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"  overall      hit_rate={report['overall']['hit_rate']:.3f}   "
        f"MRR={report['overall']['mrr']:.3f}   n={report['overall']['n']}"
    )
    lines.append("")
    lines.append("  per category:")
    for cat, m in report["per_category"].items():
        lines.append(
            f"    {cat:18s}  hit_rate={m['hit_rate']:.3f}   MRR={m['mrr']:.3f}   n={m['n']}"
        )
    lines.append("")
    lines.append("-" * 78)
    lines.append("per-case details:")
    for r in results:
        mark = "✓" if r["hit_at_k"] else "✗"
        lines.append(f"  {mark} [{r['category']:13s}] q={r['query']!r}")
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
        help="JSON output path; defaults to $MM_ASSET_RAG_HOME/eval_report_extended.json",
    )
    args = parser.parse_args()

    if args.home:
        os.environ["MM_ASSET_RAG_HOME"] = args.home
    elif "MM_ASSET_RAG_HOME" not in os.environ:
        os.environ["MM_ASSET_RAG_HOME"] = str(Path.home() / ".mm_asset_rag")

    results = [evaluate_case(c, args.top_k) for c in EVAL_CASES]
    report = aggregate(results, args.top_k)
    print(render_report(report, results))

    out = Path(args.output or f"{os.environ['MM_ASSET_RAG_HOME']}/eval_report_extended.json")
    out.write_text(
        json.dumps({"summary": report, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
