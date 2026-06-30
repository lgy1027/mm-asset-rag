"""Multimodal retrieval evaluation.

The bundled sample set is mostly a text-PDF benchmark, but the
project also indexes 171 images via CLIP. This script exercises the
multimodal paths separately from the text benchmark so we get
honest per-mode numbers.

Five categories of ground truth:

- **clip_text_to_image** — natural-language query → relevant image
  (CLIP text encoder + Qdrant image collection)
- **clip_image_to_image** — image query → visually similar image
  (CLIP image encoder + Qdrant image collection)
- **ocr_to_image** — pretend the query is the OCR output of a
  scanned asset, run CLIP text against the image collection
- **cross_modal_hybrid** — text query → mixed PDF + image top-k via
  the full ``hybrid_search`` (covers use cases where the answer can
  be either a text chunk or an image)
- **negative_text** — query that should return *no* relevant image;
  tests that the image index doesn't over-fire. ``expected_ids=[]``
  and the helper scores hit_rate@k = 1.0 when no result is
  returned for that k.

Per-category metrics reported at k ∈ {1, 3, 5, 10}: Hit Rate,
Precision, Recall, F1, NDCG, plus MRR (k-independent) and MAP.

Usage::

    export MM_ASSET_RAG_HOME=$HOME/.mm_asset_rag
    python scripts/eval_multimodal.py --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from mm_asset_rag.metrics import (
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _img_path(asset_id: str) -> Path:
    """Resolve an OpenCV/Picsum asset_id back to its on-disk image path."""
    # Manifest stores paths like ``images/img_01_opencv-sample-data-blender-suzanne1-jpg.jpg``.
    # The on-disk filename uses hyphen-separated form; we read the manifest
    # to find the exact relative path.
    import json

    manifest_path = (
        Path(
            os.environ.get(
                "MM_ASSET_RAG_ASSETS_DIR",
                Path(__file__).resolve().parent.parent / "examples/data/chapter11_assets",
            )
        )
        / "asset_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rec in payload["records"]:
        if rec["id"] == asset_id:
            return manifest_path.parent / rec["path"].replace("\\", "/")
    raise FileNotFoundError(f"asset_id {asset_id!r} not in manifest")


# ─── Ground-truth cases ────────────────────────────────────────────────
# Built from the 20 OpenCV sample images (semantic, varied) plus a
# handful of negative queries (no expected asset) to test precision.
# Picsum photos are random and not labelled, so they're not used for
# multimodal ground truth — that limitation is documented in the report.

EVAL_CASES: list[dict] = [
    # ── clip_text_to_image: natural-language → image ─────────────────────
    {
        "query": "fish",
        "expected_ids": ["img_03_opencv_sample_data_happyfish_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "Linux logo",
        "expected_ids": ["img_04_opencv_sample_data_linuxlogo_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "Windows logo",
        "expected_ids": ["img_05_opencv_sample_data_windowslogo_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "Apple computer",
        "expected_ids": ["img_11_opencv_sample_data_apple_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "butterfly insect",
        "expected_ids": ["img_20_opencv_sample_data_butterfly_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "baboon monkey primate",
        "expected_ids": ["img_12_opencv_sample_data_baboon_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "basketball",
        "expected_ids": [
            "img_13_opencv_sample_data_basketball1_png",
            "img_14_opencv_sample_data_basketball2_png",
        ],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "airplane aeroplane",
        "expected_ids": [
            "img_06_opencv_sample_data_aero1_jpg",
            "img_07_opencv_sample_data_aero3_jpg",
        ],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "box",
        "expected_ids": [
            "img_15_opencv_sample_data_blox_jpg",
            "img_17_opencv_sample_data_box_png",
            "img_18_opencv_sample_data_box_in_scene_png",
        ],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "building architecture",
        "expected_ids": ["img_19_opencv_sample_data_building_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "aloe plant",
        "expected_ids": [
            "img_09_opencv_sample_data_aloel_jpg",
            "img_10_opencv_sample_data_aloer_jpg",
        ],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "blender 3D model",
        "expected_ids": [
            "img_01_opencv_sample_data_blender_suzanne1_jpg",
            "img_02_opencv_sample_data_blender_suzanne2_jpg",
        ],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "wooden board plank",
        "expected_ids": ["img_16_opencv_sample_data_board_jpg"],
        "category": "clip_text_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    # ── clip_image_to_image: image query → similar image ───────────────
    {
        "query": "",
        "expected_ids": ["img_03_opencv_sample_data_happyfish_jpg"],
        "category": "clip_image_to_image",
        "mode": "image-to-image",
        "query_image": "img_03_opencv_sample_data_happyfish_jpg",
    },  # self-match excluded in scoring
    {
        "query": "",
        "expected_ids": ["img_05_opencv_sample_data_windowslogo_jpg"],
        "category": "clip_image_to_image",
        "mode": "image-to-image",
        "query_image": "img_04_opencv_sample_data_linuxlogo_jpg",
    },  # logo → logo
    {
        "query": "",
        "expected_ids": ["img_20_opencv_sample_data_butterfly_jpg"],
        "category": "clip_image_to_image",
        "mode": "image-to-image",
        "query_image": "img_20_opencv_sample_data_butterfly_jpg",
    },
    {
        "query": "",
        "expected_ids": [
            "img_06_opencv_sample_data_aero1_jpg",
            "img_07_opencv_sample_data_aero3_jpg",
        ],
        "category": "clip_image_to_image",
        "mode": "image-to-image",
        "query_image": "img_07_opencv_sample_data_aero3_jpg",
    },  # aero → aero
    # ── ocr_to_image: pretend the query is OCR output from a scan ───────
    {
        "query": "Microsoft Windows",
        "expected_ids": ["img_05_opencv_sample_data_windowslogo_jpg"],
        "category": "ocr_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "Linux Tux",
        "expected_ids": ["img_04_opencv_sample_data_linuxlogo_jpg"],
        "category": "ocr_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "Apple Inc Mac",
        "expected_ids": ["img_11_opencv_sample_data_apple_jpg"],
        "category": "ocr_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "swimming fish underwater",
        "expected_ids": ["img_03_opencv_sample_data_happyfish_jpg"],
        "category": "ocr_to_image",
        "mode": "text-to-image",
        "query_image": None,
    },
    # ── cross_modal_hybrid: text query → mixed PDF + image top-k ───────
    # The full ``hybrid_search`` route. We expect images to surface in
    # the top-k for these queries even though the query is text-shaped.
    {
        "query": "logo of operating system",
        "expected_ids": [
            "img_04_opencv_sample_data_linuxlogo_jpg",
            "img_05_opencv_sample_data_windowslogo_jpg",
        ],
        "category": "cross_modal_hybrid",
        "mode": "hybrid",
        "query_image": None,
    },
    {
        "query": "fruit photograph",
        "expected_ids": ["img_11_opencv_sample_data_apple_jpg"],
        "category": "cross_modal_hybrid",
        "mode": "hybrid",
        "query_image": None,
    },
    {
        "query": "papillon lepidoptera",
        "expected_ids": ["img_20_opencv_sample_data_butterfly_jpg"],
        "category": "cross_modal_hybrid",
        "mode": "hybrid",
        "query_image": None,
    },
    # ── negative_text: query that should NOT find any image ────────────
    # expected_ids=[] means: a clean retrieval should return 0 hits.
    # We only score these on the *image* collection (mode=text-to-image);
    # the text collection has docs that *do* match these terms.
    {
        "query": "Schrödinger equation",
        "expected_ids": [],
        "category": "negative_text",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "vintage automobile",
        "expected_ids": [],
        "category": "negative_text",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "domestic feline",
        "expected_ids": [],
        "category": "negative_text",
        "mode": "text-to-image",
        "query_image": None,
    },
    {
        "query": "Mount Everest summit",
        "expected_ids": [],
        "category": "negative_text",
        "mode": "text-to-image",
        "query_image": None,
    },
    # ── caltech_101: 49 labelled object categories (2 images each) ─────
    # Caltech-101 categories cover vehicles, animals, food, tools,
    # landmarks, etc. Each query maps onto its category name as it
    # appears in the manifest ``tags[0]``.
    *[
        {
            "query": label,
            "expected_ids": [
                f"caltech_{cat.lower()}_01",
                f"caltech_{cat.lower()}_02",
                f"caltech_{cat.lower()}_03",
            ],
            "category": "caltech_101",
            "mode": "text-to-image",
            "query_image": None,
        }
        for label, cat in [
            ("commercial airplane", "airplanes"),
            ("motorbike motorcycle", "Motorbikes"),
            ("side view of a car", "car_side"),
            ("helicopter rotorcraft", "helicopter"),
            ("sailboat ketch", "ketch"),
            ("schooner sailing ship", "schooner"),
            ("ferry passenger boat", "ferry"),
            ("laptop computer", "laptop"),
            ("cellphone mobile phone", "cellphone"),
            ("wristwatch", "watch"),
            ("beaver rodent", "beaver"),
            ("dolphin marine mammal", "dolphin"),
            ("dalmatian dog breed", "dalmatian"),
            ("elephant", "elephant"),
            ("kangaroo marsupial", "kangaroo"),
            ("llama pack animal", "llama"),
            ("giant panda bear", "panda"),
            ("pizza pie", "pizza"),
            ("platypus monotreme", "platypus"),
            ("rhinoceros", "rhino"),
            ("rooster chicken", "rooster"),
            ("scorpion arachnid", "scorpion"),
            ("seahorse", "sea_horse"),
            ("snoopy cartoon dog", "snoopy"),
            ("starfish echinoderm", "starfish"),
            ("accordion musical instrument", "accordion"),
            ("bonsai miniature tree", "bonsai"),
            ("brain anatomical", "brain"),
            ("brontosaurus dinosaur", "brontosaurus"),
            ("buddha statue sculpture", "buddha"),
            ("camera photography", "camera"),
            ("cannon artillery", "cannon"),
            ("chair furniture", "chair"),
            ("cup drinking vessel", "cup"),
            ("electric guitar instrument", "electric_guitar"),
            ("strawberry fruit", "strawberry"),
            ("sunflower bloom", "sunflower"),
            ("water lily flower", "water_lilly"),
            ("lotus flower", "lotus"),
            ("pagoda tower", "pagoda"),
            ("pyramid monument", "pyramid"),
            ("minaret tower", "minaret"),
            ("stop sign traffic", "stop_sign"),
            ("saxophone wind instrument", "saxophone"),
            ("stapler office tool", "stapler"),
            ("wrench tool", "wrench"),
            ("scissors cutting tool", "scissors"),
            ("lamp light fixture", "lamp"),
            ("revolver pistol", "revolver"),
        ]
    ],
]


def _run_text_to_image(query: str, top_k: int) -> list[tuple[str, float]]:
    from mm_asset_rag.backends.qdrant_backend import qdrant_text_to_image_search

    hits = qdrant_text_to_image_search(query, top_k=top_k)
    return [(h.asset_id, h.score) for h in hits]


def _run_image_to_image(query_image: str, top_k: int) -> list[tuple[str, float]]:
    from mm_asset_rag.backends.qdrant_backend import qdrant_image_to_image_search

    hits = qdrant_image_to_image_search(_img_path(query_image), top_k=top_k)
    return [(h.asset_id, h.score) for h in hits]


def _run_hybrid(query: str, top_k: int) -> list[tuple[str, float]]:
    from mm_asset_rag.retrieval import hybrid_search

    hits = hybrid_search(query, top_k=top_k)
    return [(h.asset_id, h.score) for h in hits]


_RUNNERS = {
    "text-to-image": _run_text_to_image,
    "image-to-image": _run_image_to_image,
    "hybrid": _run_hybrid,
}


def evaluate_case(case: dict, top_k: int) -> dict:
    runner = _RUNNERS[case["mode"]]
    if case["mode"] == "image-to-image":
        actual_pairs = runner(case["query_image"], top_k)
    else:
        actual_pairs = runner(case["query"], top_k)

    actual_ids = [aid for aid, _score in actual_pairs]
    actual_scores = [score for _aid, score in actual_pairs]
    expected = case["expected_ids"]
    hit = (
        any(e in a for a in actual_ids[:top_k] for e in expected)
        if expected
        else len(actual_ids) == 0
    )

    return {
        "query": case["query"],
        "query_image": case.get("query_image"),
        "category": case["category"],
        "mode": case["mode"],
        "expected_ids": expected,
        "actual_ids": actual_ids,
        "actual_scores": actual_scores,
        "hit_at_k": hit,
        "reciprocal_rank": reciprocal_rank(actual_ids, expected) if expected else 0.0,
    }


def aggregate_per_category(results: list[dict], top_k: int) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out: dict = {}
    for cat, group in by_cat.items():
        if cat == "negative_text":
            # For negatives, "hit@k" = 1.0 when the top-k returns no image
            # candidates. We override hit_rate with the proportion of
            # cases where no result came back.
            neg_rate = sum(1 for r in group if len(r["actual_ids"]) == 0) / len(group)
            out[cat] = {
                "n": len(group),
                "no_result_rate": round(neg_rate, 3),
                "avg_results_returned": round(
                    sum(len(r["actual_ids"]) for r in group) / len(group), 1
                ),
                "examples": [
                    (r["query"], r["actual_ids"][:3], r.get("actual_scores", [])[:3]) for r in group
                ],
            }
        else:
            n = len(group)

            def avg(metric, k):
                return round(
                    sum(metric(r["actual_ids"], r["expected_ids"], k) for r in group) / n,
                    3,
                )

            out[cat] = {
                "n": n,
                "hit_rate@1": avg(hit_rate_at_k, 1),
                "hit_rate@5": avg(hit_rate_at_k, 5),
                "hit_rate@10": avg(hit_rate_at_k, 10),
                "ndcg@5": avg(ndcg_at_k, 5),
                "ndcg@10": avg(ndcg_at_k, 10),
                "precision@5": avg(precision_at_k, 5),
                "precision@10": avg(precision_at_k, 10),
                "recall@10": avg(recall_at_k, 10),
                "mrr": round(sum(r["reciprocal_rank"] for r in group) / n, 3),
                "map": round(
                    sum(average_precision(r["actual_ids"], r["expected_ids"]) for r in group) / n,
                    3,
                ),
            }
    return out


def render_report(report: dict) -> str:
    lines = ["=" * 78, "mm-asset-rag multimodal evaluation", "=" * 78, ""]
    lines.append(f"Total cases: {sum(c['n'] for c in report['per_category'].values())}")
    lines.append("")
    for cat, m in report["per_category"].items():
        lines.append(f"── {cat} (n={m['n']}) ──")
        if "no_result_rate" in m:
            lines.append(
                f"   no_result_rate (precision proxy):  {m['no_result_rate']:.3f}  (1.0 = always empty)"
            )
            lines.append(f"   avg_results_returned:              {m['avg_results_returned']}")
            lines.append("   examples:")
            for q, ids, scores in m["examples"]:
                lines.append(
                    f"     q={q!r:40s} actual={ids} scores={[round(s, 3) for s in scores]}"
                )
        else:
            for k, v in m.items():
                if k in ("n",):
                    continue
                lines.append(f"   {k:20s} {v}")
        lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path; default $MM_ASSET_RAG_HOME/eval_report_multimodal.json",
    )
    args = parser.parse_args()

    if "MM_ASSET_RAG_HOME" not in os.environ:
        os.environ["MM_ASSET_RAG_HOME"] = str(Path.home() / ".mm_asset_rag")

    results = [evaluate_case(c, args.top_k) for c in EVAL_CASES]
    per_cat = aggregate_per_category(results, args.top_k)
    report = {"top_k": args.top_k, "per_category": per_cat, "results": results}
    print(render_report(report))

    out = Path(args.output or f"{os.environ['MM_ASSET_RAG_HOME']}/eval_report_multimodal.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
