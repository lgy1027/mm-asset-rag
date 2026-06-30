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
    "alexnet.pdf": (
        "ImageNet Classification with Deep Convolutional Neural Networks",
        "https://arxiv.org/pdf/1404.5997",
        ["alexnet", "cnn", "image-classification"],
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
    "ddpm.pdf": (
        "Denoising Diffusion Probabilistic Models",
        "https://arxiv.org/pdf/2010.02502",
        ["diffusion", "generative", "ddpm"],
    ),
    "densenet.pdf": (
        "Densely Connected Convolutional Networks",
        "https://arxiv.org/pdf/1608.06993",
        ["cnn", "image-classification", "densenet"],
    ),
    "detr.pdf": (
        "End-to-End Object Detection with Transformers",
        "https://arxiv.org/pdf/2005.12872",
        ["object-detection", "transformer", "set-prediction"],
    ),
    "efficientnet.pdf": (
        "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        "https://arxiv.org/pdf/1905.11946",
        ["cnn", "efficient-net", "scaling"],
    ),
    "flamingo.pdf": (
        "Flamingo: a Visual Language Model for Few-Shot Learning",
        "https://arxiv.org/pdf/2204.14198",
        ["vision-language", "few-shot", "multimodal"],
    ),
    "gan.pdf": (
        "Generative Adversarial Networks",
        "https://arxiv.org/pdf/1406.2661",
        ["gan", "generative", "adversarial"],
    ),
    "glove.pdf": (
        "GloVe: Global Vectors for Word Representation",
        "https://arxiv.org/pdf/1502.03717",
        ["word-embedding", "nlp", "glove"],
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
    "mobilenet.pdf": (
        "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications",
        "https://arxiv.org/pdf/1704.04861",
        ["mobilenet", "cnn", "efficient"],
    ),
    "mobilenetv2.pdf": (
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
        "https://arxiv.org/pdf/1801.04381",
        ["mobilenet", "cnn", "efficient"],
    ),
    "pix2pix.pdf": (
        "Image-to-Image Translation with Conditional Adversarial Networks",
        "https://arxiv.org/pdf/1611.07004",
        ["gan", "image-to-image", "generative"],
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
    "resnet.pdf": (
        "Deep Residual Learning for Image Recognition",
        "https://arxiv.org/pdf/1512.03385",
        ["cnn", "resnet", "image-classification"],
    ),
    "resnext.pdf": (
        "Aggregated Residual Transformations for Deep Neural Networks",
        "https://arxiv.org/pdf/1611.05431",
        ["cnn", "resnext", "image-classification"],
    ),
    "segment-anything.pdf": (
        "Segment Anything",
        "https://arxiv.org/pdf/2304.02643",
        ["segmentation", "foundation-model", "vision"],
    ),
    "ssd.pdf": (
        "SSD: Single Shot MultiBox Detector",
        "https://arxiv.org/pdf/1512.02325",
        ["object-detection", "ssd", "cnn"],
    ),
    "stable_diffusion.pdf": (
        "High-Resolution Image Synthesis with Latent Diffusion Models",
        "https://arxiv.org/pdf/2112.10752",
        ["diffusion", "text-to-image", "generative"],
    ),
    "unet.pdf": (
        "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "https://arxiv.org/pdf/1505.04597",
        ["unet", "segmentation", "medical-imaging"],
    ),
    "vae.pdf": (
        "Auto-Encoding Variational Bayes",
        "https://arxiv.org/pdf/1312.6114",
        ["vae", "generative", "variational"],
    ),
    "vit.pdf": (
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "https://arxiv.org/pdf/2010.11929",
        ["vision-transformer", "image-classification", "transformer"],
    ),
    "word2vec.pdf": (
        "Efficient Estimation of Word Representations in Vector Space",
        "https://arxiv.org/pdf/1301.3781",
        ["word-embedding", "nlp", "word2vec"],
    ),
    "yolo.pdf": (
        "You Only Look Once: Unified, Real-Time Object Detection",
        "https://arxiv.org/pdf/1506.02640",
        ["yolo", "object-detection", "real-time"],
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
    from mm_asset_rag.assets import safe_write_manifest

    records = build_records()
    payload = {
        "name": "mm-asset-rag extended sample set (PDFs + Picsum photos + OpenCV)",
        "total": len(records),
        "records": records,
    }
    # Atomic temp-file + ``os.replace`` + ``.bak`` rotation. If a previous
    # build of the manifest is on disk the .bak preserves the prior
    # version for one round of recovery.
    safe_write_manifest(MANIFEST, payload, backup=True)
    print(f"Wrote {MANIFEST} with {len(records)} records "
          f"({sum(1 for r in records if r['type'] == 'pdf')} pdf, "
          f"{sum(1 for r in records if r['type'] == 'image')} image)")


if __name__ == "__main__":
    main()
