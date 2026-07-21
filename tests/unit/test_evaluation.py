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
    strip_trailing_hash,
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
    # ── ZH_PAPER_QUERIES — the queries not already matched by an EN needle
    # above (e.g. 去噪扩散概率模型 has no English substring in the table).
    "去噪扩散概率模型": "Ddpm_caaa534b",
    "残差网络": "Resnet_caaa534b",
    "端到端 Transformer 候选框": "Detr_caaa534b",
    "反向残差线性瓶颈": "Mobilenetv2_caaa534b",
    "变分自编码器 VAE": "Vae_caaa534b",
    "分割一切 SAM": "Segment Anything_caaa534b",
    "低秩适配 LoRA 大模型微调": "Lora_caaa534b",
    # ── ZH_DOC_QUERIES — 联宝 / AI-tutorial Chinese corpus (10).
    "可发电键盘专利": "创新联宝 会发电的键盘_caaa534b",
    "中试基地 省级备案": "联宝科技中试基地获省级备案_caaa534b",
    "可拉伸屏幕": "CES 2026再绽光芒_caaa534b",
    "合肥新春第一会": "受邀参加合肥_caaa534b",
    "一群机器人": "媒眼看联宝_caaa534b",
    "外贸破万亿": "安徽外贸再创新高_caaa534b",
    "财年启幕": "敢AI敢为_caaa534b",
    "ESG 年度答卷": "ESG年度答卷_caaa534b",
    "Obsidian AI Skill 本地知识库": "Obsidian 的 10 大 AI Skill_caaa534b",
    "Codex 全景指南": "Codex 全景指南_caaa534b",
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
    # EVAL_CASES = EN_PAPER_QUERIES + ZH_PAPER_QUERIES + ZH_DOC_QUERIES.
    from mm_asset_rag.evaluation import ZH_DOC_QUERIES, ZH_PAPER_QUERIES

    assert len(EVAL_CASES) == len(EN_PAPER_QUERIES) + len(ZH_PAPER_QUERIES) + len(ZH_DOC_QUERIES)
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
                title="",
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


# ── strip_trailing_hash + casefold normalisation ─────────────────────────


def test_strip_trailing_hash_drops_8_hex_suffix() -> None:
    assert strip_trailing_hash("Alexnet_0c1c2b23") == "alexnet"
    # casefold applies to the title portion.
    assert strip_trailing_hash("AlexNet_caaa534b") == "alexnet"
    # Non-hex tail (length 8 but not hex) → keep whole, casefold only.
    assert strip_trailing_hash("foo_longtail") == "foo_longtail"
    # No underscore → casefold the whole id.
    assert strip_trailing_hash("CLIP") == "clip"
    assert strip_trailing_hash("") == ""


def test_match_case_insensitive_partial_title() -> None:
    """The R-CNN failure: retriever returns the correct paper with a
    different casing + content hash, expected is a partial bare title.
    After normalisation the matcher must count this as a hit.
    """
    from mm_asset_rag.evaluation import _match

    actual = [
        "Rich Feature Hierarchies for Accurate Object Detection And Semantic Segmentation_b857cf69"
    ]
    expected = ["Rich feature hierarchies"]
    assert _match(actual, expected) == 1


def test_match_casefold_hash_variant() -> None:
    """Same content re-parsed under a new hash + different casing still
    counts as the same document."""
    from mm_asset_rag.evaluation import _match

    assert _match(["ALEXNET_0c1c2b23"], ["Alexnet_caaa534b"]) == 1


def test_match_still_misses_unrelated() -> None:
    from mm_asset_rag.evaluation import _match

    assert _match(["Caltech Panda 01_3443a5d5"], ["Caltech Dolphin"]) is None
    # Empty expected never hits.
    assert _match(["anything"], []) is None


def test_write_eval_report_normalises_ids_for_metrics(tmp_path: Path) -> None:
    """aggregate_metrics gets hash-stripped + casefolded ids so a
    re-parse with a different sha counts in the strict-set metric."""
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["Alexnet_caaa534b"],
            actual_asset_ids=["Alexnet_0c1c2b23"],
            hit=True,
            rank=1,
            group="en",
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    # hit_rate@1 in the aggregate block should be 1.0 once the two
    # hash variants normalise to the same bare title.
    assert payload["metrics"]["all"]["hit_rate"]["1"] == 1.0


def test_write_eval_report_handles_title_list_shorter_than_ids(tmp_path: Path) -> None:
    """If actual_titles is shorter than actual_asset_ids (shouldn't happen in
    production but a future code path might), zip_longest fills with "" so the
    trailing asset_ids are still scored against expected — no silent drop."""
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["Alexnet"],
            actual_asset_ids=["Alexnet_0c1c2b23", "Resnext_69df8de4"],
            hit=True,
            rank=1,
            group="en",
            actual_titles=["Learning Transferable"],  # only 1 title for 2 ids
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    # Both actual_ids still scored (the 2nd falls back to bare asset_id).
    assert payload["metrics"]["all"]["hit_rate"]["5"] == 1.0


def test_write_eval_report_empty_titles_falls_back_to_asset_id(tmp_path: Path) -> None:
    """When actual_titles is empty (image route / old reports), _agg falls back
    to scoring on bare asset_ids — no crash, no empty actual_ids list."""
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["Alexnet_caaa534b"],
            actual_asset_ids=["Alexnet_0c1c2b23"],
            hit=True,
            rank=1,
            group="en",
            actual_titles=[],
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["metrics"]["all"]["hit_rate"]["1"] == 1.0
