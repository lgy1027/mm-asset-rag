"""Tests for mm_asset_rag.evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mm_asset_rag.evaluation import (
    EN_PAPER_QUERIES,
    EVAL_CASES,
    EvalResult,
    run_eval,
    write_eval_report,
)
from mm_asset_rag.schema import SearchHit

# asset_id returned by the fake retriever for each query substring.
# Keyed on a unique phrase that appears in the actual EVAL_CASES. Keeps
# the test independent of how the asset_id is built (bare / full / hashed).
_QUERY_TO_ASSET = {
    "retrieval augmented generation": "Retrieval Augmented Generation_caaa534b",
    "ImageNet classification": "Alexnet_caaa534b",
    "attention is all you need": "Attention Is All You Need_caaa534b",
    "BERT pre-training": "Bert_caaa534b",
    "transferable visual models": "Learning Transferable Visual Models From Natural Language Supervision_caaa534b",
    "denoising diffusion": "Ddpm_caaa534b",
    "densely connected": "Densenet_caaa534b",
    "object detection with transformers": "Detr_caaa534b",
    "compound scaling": "EfficientNet_caaa534b",
    "flamingo visual": "Flamingo_caaa534b",
    "generative adversarial": "Gan_caaa534b",
    "GloVe global": "Glove_caaa534b",
    "Language Models are Few-Shot": "Gpt3_caaa534b",
    "LayoutLM": "LayoutLM_caaa534b",
    "LLaMA open": "Llama_caaa534b",
    "LoRA low-rank": "Lora_caaa534b",
    "MobileNets": "Mobilenet_caaa534b",
    "MobileNetV2": "Mobilenetv2_caaa534b",
    "Pix2Pix": "Pix2Pix_caaa534b",
    "rich feature hierarchies": "Rich feature hierarchies_caaa534b",
    "Deep Residual": "Resnet_caaa534b",
    "Aggregated Residual": "Aggregated Residual Transformations_caaa534b",
    "RAG": "Retrieval Augmented Generation_caaa534b",
    "segment anything": "Segment Anything_caaa534b",
    "SSD single shot": "Ssd_caaa534b",
    "latent diffusion": "Stable Diffusion_caaa534b",
    "U-Net convolutional": "U-Net Convolutional Networks for Biomedical Image Segmentation_caaa534b",
    "auto-encoding": "Vae_caaa534b",
    "vision transformer": "Vit_caaa534b",
    "word2vec": "Word2Vec_caaa534b",
    "YOLO you only": "You Only Look Once_caaa534b",
    "2026 年 AI": "2026 年 AI 技术趋势与 Codex 模型发展_caaa534b",
    "Obsidian": "Obsidian 的 10 大 AI Skill_caaa534b",
    "CLIP 图文对齐": "Learning Transferable Visual Models From Natural Language Supervision_caaa534b",
    "文档版面": "LayoutLM_caaa534b",
    "GAN": "Gan_caaa534b",
    "自注意力": "Attention Is All You Need_caaa534b",
    "图像分类 深度卷积": "Alexnet_caaa534b",
}


def _fake_hybrid(query: str, **kwargs):
    """Return a single ``SearchHit`` keyed on the test's query → asset map."""
    q = query.lower()
    asset_id = next(
        (aid for needle, aid in _QUERY_TO_ASSET.items() if needle.lower() in q),
        "other",
    )
    return [
        SearchHit(
            route="text",
            score=0.9,
            asset_id=asset_id,
            title=asset_id,
            source_type="pdf",
            source_path=f"{asset_id}.pdf",
        )
    ]


def test_eval_cases_count() -> None:
    # 32 arxiv papers + 6 ZH queries (3 legacy + 3 fresh).
    assert len(EVAL_CASES) == len(EN_PAPER_QUERIES) + 6
    assert len(EVAL_CASES) >= 30  # guard against accidental pruning


def test_run_eval_hits_expected_assets() -> None:
    # Skip bare→full expansion so the expected ids stay as the model
    # names from EVAL_CASES; the prefix-tolerant _match handles the
    # test's mock asset_id suffix transparently.
    with (
        patch("mm_asset_rag.evaluation.hybrid_search", side_effect=_fake_hybrid),
        patch("mm_asset_rag.evaluation._load_asset_id_index", return_value={}),
    ):
        results = run_eval()
    assert len(results) == len(EVAL_CASES)
    misses = [r for r in results if not r.hit]
    assert not misses, misses


def test_run_eval_misses_when_assets_wrong() -> None:
    with patch(
        "mm_asset_rag.evaluation.hybrid_search",
        return_value=[
            SearchHit(
                route="text",
                score=0.9,
                asset_id="unrelated",
                title="x",
                source_type="pdf",
                source_path="x.pdf",
            )
        ],
    ):
        results = run_eval()
    assert not any(result.hit for result in results)
    # Every miss should record rank=None and the actual asset_id of "unrelated".
    for r in results:
        assert r.rank is None
        assert r.actual_asset_ids == ["unrelated"]


def test_run_eval_records_rank_and_group() -> None:
    with (
        patch("mm_asset_rag.evaluation.hybrid_search", side_effect=_fake_hybrid),
        patch("mm_asset_rag.evaluation._load_asset_id_index", return_value={}),
    ):
        results = run_eval()
    en = [r for r in results if r.group == "en"]
    zh = [r for r in results if r.group == "zh"]
    assert en, "expected en queries"
    assert zh, "expected zh queries"
    for r in en:
        assert r.rank == 1, r
    for r in zh:
        assert r.rank == 1, r


def test_write_eval_report(tmp_path: Path) -> None:
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["a"],
            actual_asset_ids=["a"],
            hit=True,
            rank=1,
            group="en",
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    # New schema: a top-level object (not a bare list) with per_query +
    # aggregate metrics.
    assert payload["total"] == 1
    assert payload["hit_count"] == 1
    assert payload["per_query"][0]["query"] == "q"
    assert payload["per_query"][0]["rank"] == 1
    assert payload["per_query"][0]["group"] == "en"
    assert "metrics" in payload
    assert "all" in payload["metrics"]
    assert "en" in payload["metrics"]
    assert "zh" in payload["metrics"]
