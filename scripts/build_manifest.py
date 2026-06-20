"""Build examples/data/chapter11_assets/asset_manifest.json from the
on-disk file layout. Run from the repo root:

    python scripts/build_manifest.py

Intended for one-off regeneration; the file is checked in.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "examples" / "data" / "chapter11_assets"
MANIFEST = ASSETS / "asset_manifest.json"


# Hand-curated entries where we know the content.
PDF_ENTRIES = {
    "attention-is-all-you-need.pdf": (
        "Attention Is All You Need",
        "https://arxiv.org/pdf/1706.03762",
        ["transformer", "paper", "tables", "figures"],
    ),
    "bert.pdf": (
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "https://arxiv.org/pdf/1810.04805",
        ["transformer", "language-model", "pretraining"],
    ),
    "clip.pdf": (
        "Learning Transferable Visual Models From Natural Language Supervision",
        "https://arxiv.org/pdf/2103.00020",
        ["clip", "vision-language", "contrastive", "multimodal"],
    ),
    "detr.pdf": (
        "End-to-End Object Detection with Transformers",
        "https://arxiv.org/pdf/2005.12872",
        ["object-detection", "transformer", "set-prediction"],
    ),
    "flamingo.pdf": (
        "Flamingo: a Visual Language Model for Few-Shot Learning",
        "https://arxiv.org/pdf/2204.14198",
        ["vision-language", "few-shot", "multimodal"],
    ),
    "gpt3.pdf": (
        "Language Models are Few-Shot Learners",
        "https://arxiv.org/pdf/2005.14165",
        ["language-model", "few-shot", "scaling"],
    ),
    "layoutlm.pdf": (
        "LayoutLM: Pre-training of Text and Layout for Document AI",
        "https://arxiv.org/pdf/1912.13322",
        ["document-ai", "layout", "ocr"],
    ),
    "llama.pdf": (
        "LLaMA: Open and Efficient Foundation Language Models",
        "https://arxiv.org/pdf/2302.13971",
        ["language-model", "foundation-model", "open"],
    ),
    "lora.pdf": (
        "LoRA: Low-Rank Adaptation of Large Language Models",
        "https://arxiv.org/pdf/2106.09685",
        ["parameter-efficient", "fine-tuning", "lora"],
    ),
    "rag.pdf": (
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "https://arxiv.org/pdf/2005.11401",
        ["rag", "retrieval", "generation"],
    ),
    "rcnn.pdf": (
        "Rich feature hierarchies for accurate object detection and semantic segmentation",
        "https://arxiv.org/pdf/1311.2524",
        ["object-detection", "rcnn", "cnn"],
    ),
    "segment-anything.pdf": (
        "Segment Anything",
        "https://arxiv.org/pdf/2304.02643",
        ["segmentation", "foundation-model", "vision"],
    ),
    "stable_diffusion.pdf": (
        "High-Resolution Image Synthesis with Latent Diffusion Models",
        "https://arxiv.org/pdf/2112.10752",
        ["diffusion", "text-to-image", "generative"],
    ),
    "vit.pdf": (
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "https://arxiv.org/pdf/2010.11929",
        ["vision-transformer", "image-classification", "transformer"],
    ),
}


def _id_for(filename: str, kind: str) -> str:
    stem = Path(filename).stem
    return stem.replace("-", "_").replace(".", "_")


def build_records() -> list[dict]:
    records: list[dict] = []

    for pdf_path in sorted((ASSETS / "pdfs").glob("*.pdf")):
        title, source_url, tags = PDF_ENTRIES.get(
            pdf_path.name,
            (pdf_path.stem.replace("_", " ").title(), "", ["paper"]),
        )
        records.append(
            {
                "id": _id_for(pdf_path.name, "pdf"),
                "title": title,
                "type": "pdf",
                "path": f"pdfs\\\\{pdf_path.name}",
                "source_url": source_url,
                "tags": tags,
            }
        )

    for img_path in sorted((ASSETS / "images").glob("*.jpg")) + sorted(
        (ASSETS / "images").glob("*.png")
    ):
        records.append(
            {
                "id": _id_for(img_path.name, "img"),
                "title": img_path.stem.replace("_", " ").title(),
                "type": "image",
                "path": f"images\\\\{img_path.name}",
                "source_url": "",
                "tags": ["photo"],
            }
        )

    return records


def main() -> None:
    records = build_records()
    payload = {
        "name": "mm-asset-rag extended sample set (PDFs + Picsum photos + OpenCV)",
        "total": len(records),
        "records": records,
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {MANIFEST} with {len(records)} records "
          f"({sum(1 for r in records if r['type'] == 'pdf')} pdf, "
          f"{sum(1 for r in records if r['type'] == 'image')} image)")


if __name__ == "__main__":
    main()
