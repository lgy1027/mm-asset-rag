"""Regression set for retrieval quality.

The set is split into three groups:

- ``EN_PAPER_QUERIES`` — one to two natural-language topic queries per
  chapter11 arxiv paper. Used to compute hit_rate / MRR / NDCG across
  the 32-paper English corpus.
- ``ZH_PAPER_QUERIES`` — Chinese queries that map to the same papers,
  exercising the cross-language path (Chinese BM25 / dense).
- ``LEGACY_QUERIES`` — the original three-case regression set kept for
  backwards compatibility.

Every case pairs a free-text ``query`` with one or more
``expected_asset_ids``. Matching is prefix-tolerant: a case "hits" if
any expected id is a substring of any actual id, or vice versa, so
bare model names like ``clip`` still match
``Learning Transferable Visual Models From Natural Language Supervision_79e328a2``
once the search returns the full asset id.

Use :func:`run_eval` to compute the raw results and
:func:`write_eval_report` to persist them to
``$MM_ASSET_RAG_HOME/eval_report.json``. The full per-query details
plus aggregate metrics (hit_rate, MRR, NDCG@k, MAP, ...) are dumped
as JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .metrics import aggregate_metrics
from .paths import get_asset_index_path, get_eval_report
from .retrieval import hybrid_search


def _load_asset_id_index() -> dict[str, str]:
    """Build a ``bare_id`` (no _hash) → ``full_id`` map from ``asset_index.jsonl``.

    The chapter11 corpus is parsed with content-hash asset_ids, but the
    :data:`EVAL_CASES` reference bare model names (``"Alexnet"``,
    ``"Bert"``, ...). To compute standard IR metrics
    (hit_rate@k / MRR / NDCG) the expected set has to be the same shape
    as the returned actual set, so we expand bare names to the full
    hashed ids the index actually returns.
    """
    index_path = get_asset_index_path()
    if not index_path.exists():
        return {}
    latest: dict[str, dict] = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("deleted"):
                continue
            latest[row["sha256"]] = row
    bare_to_full: dict[str, str] = {}
    for row in latest.values():
        full = row.get("asset_id", "")
        m = re.match(r"^(.+)_([0-9a-f]{8})$", full)
        bare = m.group(1) if m else full
        # If multiple rows share the same bare id (e.g. reparse of the
        # same source under a new hash), keep the most recent by
        # file mtime ordering — first-wins is fine since the latest
        # row wins in our ``latest`` dict.
        bare_to_full.setdefault(bare, full)
    return bare_to_full


# ── English paper queries ─────────────────────────────────────────────
# Format: (query, [expected_asset_id_or_substring, ...])
EN_PAPER_QUERIES: list[dict] = [
    {
        "query": "ImageNet classification with deep convolutional neural networks",
        "expected_asset_ids": ["Alexnet"],
    },
    {
        "query": "attention is all you need transformer architecture",
        "expected_asset_ids": ["Attention Is All You Need"],
    },
    {
        "query": "BERT pre-training bidirectional transformer language model",
        "expected_asset_ids": ["Bert"],
    },
    {
        "query": "learning transferable visual models natural language supervision contrastive",
        "expected_asset_ids": [
            "Learning Transferable Visual Models From Natural Language Supervision"
        ],
    },
    {"query": "denoising diffusion probabilistic models", "expected_asset_ids": ["Ddpm"]},
    {"query": "densely connected convolutional networks", "expected_asset_ids": ["Densenet"]},
    {
        "query": "end-to-end object detection with transformers set prediction",
        "expected_asset_ids": ["Detr"],
    },
    {
        "query": "compound scaling of convolutional neural networks efficientnet",
        "expected_asset_ids": ["EfficientNet"],
    },
    {
        "query": "flamingo visual language model few-shot learning",
        "expected_asset_ids": ["Flamingo"],
    },
    {"query": "generative adversarial networks", "expected_asset_ids": ["Gan"]},
    {"query": "GloVe global vectors for word representation", "expected_asset_ids": ["Glove"]},
    {"query": "language models are few-shot learners GPT-3", "expected_asset_ids": ["Gpt3"]},
    {
        "query": "LayoutLM pre-training text and layout document image understanding",
        "expected_asset_ids": ["LayoutLM"],
    },
    {
        "query": "LLaMA open and efficient foundation language models",
        "expected_asset_ids": ["Llama"],
    },
    {"query": "LoRA low-rank adaptation of large language models", "expected_asset_ids": ["Lora"]},
    {
        "query": "MobileNets efficient convolutional networks for mobile vision",
        "expected_asset_ids": ["Mobilenet"],
    },
    {
        "query": "MobileNetV2 inverted residuals linear bottlenecks",
        "expected_asset_ids": ["Mobilenetv2"],
    },
    {
        "query": "Pix2Pix image-to-image translation conditional adversarial",
        "expected_asset_ids": ["Pix2Pix"],
    },
    {
        "query": "rich feature hierarchies object detection semantic segmentation R-CNN",
        "expected_asset_ids": ["Rich feature hierarchies"],
    },
    {"query": "deep residual learning image recognition ResNet", "expected_asset_ids": ["Resnet"]},
    {
        "query": "aggregated residual transformations ResNeXt",
        "expected_asset_ids": ["Aggregated Residual Transformations"],
    },
    {
        "query": "retrieval augmented generation RAG",
        "expected_asset_ids": ["Retrieval Augmented Generation"],
    },
    {"query": "segment anything foundation model", "expected_asset_ids": ["Segment Anything"]},
    {"query": "SSD single shot multibox detector", "expected_asset_ids": ["Ssd"]},
    {
        "query": "high-resolution image synthesis latent diffusion stable diffusion",
        "expected_asset_ids": ["Stable Diffusion"],
    },
    {
        "query": "U-Net convolutional networks biomedical image segmentation",
        "expected_asset_ids": ["U-Net Convolutional Networks for Biomedical Image Segmentation"],
    },
    {"query": "auto-encoding variational Bayes", "expected_asset_ids": ["Vae"]},
    {"query": "vision transformer image recognition ViT", "expected_asset_ids": ["Vit"]},
    {
        "query": "word2vec efficient estimation of word representations",
        "expected_asset_ids": ["Word2Vec"],
    },
    {
        "query": "YOLO you only look once unified real-time object detection",
        "expected_asset_ids": ["You Only Look Once"],
    },
    # Chinese-language non-paper assets:
    {
        "query": "2026 年 AI 技术趋势 Codex 模型发展",
        "expected_asset_ids": ["2026 年 AI 技术趋势与 Codex 模型发展"],
    },
    {
        "query": "Obsidian 的 10 大 AI Skill 工具介绍",
        "expected_asset_ids": ["Obsidian 的 10 大 AI Skill"],
    },
]

# ── Chinese queries on the English corpus (cross-language) ────────────
ZH_PAPER_QUERIES: list[dict] = [
    {
        "query": "哪份资料讲了 retrieval augmented generation？",
        "expected_asset_ids": ["Retrieval Augmented Generation"],
    },
    {
        "query": "找和 CLIP 图文对齐有关的资料",
        "expected_asset_ids": [
            "Learning Transferable Visual Models From Natural Language Supervision"
        ],
    },
    {"query": "有没有包含文档版面理解或 OCR 的资料？", "expected_asset_ids": ["LayoutLM"]},
    {"query": "讲生成对抗网络 GAN 的论文", "expected_asset_ids": ["Gan"]},
    {
        "query": "关于自注意力 Transformer 的原始论文",
        "expected_asset_ids": ["Attention Is All You Need"],
    },
    {"query": "图像分类 深度卷积神经网络 AlexNet", "expected_asset_ids": ["Alexnet"]},
]

# ── Original 3-case regression (kept for backwards compat) ────────────
LEGACY_QUERIES: list[dict] = [
    {
        "query": "哪份资料讲了 retrieval augmented generation？",
        "expected_asset_ids": ["Retrieval Augmented Generation"],
    },
    {
        "query": "找和 CLIP 图文对齐有关的资料",
        "expected_asset_ids": [
            "Learning Transferable Visual Models From Natural Language Supervision"
        ],
    },
    {"query": "有没有包含文档版面理解或 OCR 的资料？", "expected_asset_ids": ["LayoutLM"]},
]

EVAL_CASES: list[dict] = EN_PAPER_QUERIES + ZH_PAPER_QUERIES


@dataclass
class EvalResult:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool
    rank: int | None  # 1-based rank of the first hit, None if missed
    group: str  # "en", "zh", or "legacy"


def _match(actual: list[str], expected: list[str]) -> int | None:
    """Return 1-based rank of the first hit, or ``None`` if missed.

    Prefix-tolerant: ``exp in act or act in exp``.
    """
    for rank, act in enumerate(actual, start=1):
        for exp in expected:
            if not exp:
                continue
            if exp in act or act in exp:
                return rank
    return None


def run_eval(top_k: int = 5) -> list[EvalResult]:
    """Run the full regression set against the live index.

    Returns a list of :class:`EvalResult` — one per case, in declared
    order. The combined EN + ZH + legacy sets are scored independently
    so the report can break down cross-language accuracy.
    """
    bare_to_full = _load_asset_id_index()
    results: list[EvalResult] = []
    for group, cases in (("en", EN_PAPER_QUERIES), ("zh", ZH_PAPER_QUERIES)):
        for case in cases:
            hits = hybrid_search(str(case["query"]), top_k=top_k)
            actual = [hit.asset_id for hit in hits]
            # Resolve bare expected ids to the full asset_ids the index
            # actually returns. The aggregate metrics
            # (hit_rate / MRR / NDCG) use set membership, so the two
            # sets must be in the same form.
            expected = [
                bare_to_full.get(str(item), str(item)) for item in case["expected_asset_ids"]
            ]
            rank = _match(actual, expected)
            results.append(
                EvalResult(
                    query=str(case["query"]),
                    expected_asset_ids=expected,
                    actual_asset_ids=actual,
                    hit=rank is not None,
                    rank=rank,
                    group=group,
                )
            )
    return results


def write_eval_report(results: list[EvalResult], path=None) -> None:
    """Write per-query results + aggregate metrics to ``eval_report.json``.

    The aggregate block uses :func:`mm_asset_rag.metrics.aggregate_metrics`
    which computes hit_rate / precision / recall / f1 / ndcg at k=1,3,5,10
    plus MRR and MAP. Metrics are reported for the full set and per
    language group.
    """
    target = path or get_eval_report()

    def _to_dict(r: EvalResult) -> dict:
        d = asdict(r)
        return d

    per_query = [_to_dict(r) for r in results]

    def _agg(rs: list[EvalResult]) -> dict:
        if not rs:
            return {}
        return aggregate_metrics(
            [{"actual_ids": r.actual_asset_ids, "expected_ids": r.expected_asset_ids} for r in rs]
        )

    by_group: dict[str, list[EvalResult]] = {"all": list(results), "en": [], "zh": []}
    for r in results:
        by_group.setdefault(r.group, []).append(r)

    payload = {
        "total": len(results),
        "hit_count": sum(1 for r in results if r.hit),
        "hit_rate": (sum(1 for r in results if r.hit) / max(len(results), 1)),
        "per_query": per_query,
        "metrics": {
            "all": _agg(by_group["all"]),
            "en": _agg(by_group.get("en", [])),
            "zh": _agg(by_group.get("zh", [])),
        },
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    from .config import load_env

    load_env()
    res = run_eval(top_k=5)
    write_eval_report(res)
    print(
        json.dumps(
            {
                k: v
                for k, v in [("hit_rate", sum(r.hit for r in res) / len(res)), ("total", len(res))]
            },
            ensure_ascii=False,
        )
    )
