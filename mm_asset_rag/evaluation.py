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


def _load_bare_to_all_fulls() -> dict[str, list[str]]:
    """Build a ``bare`` → ``[full, ...]`` map covering every hash variant.

    Unlike :func:`_load_asset_id_index` (which keeps only the latest
    hash per bare title), this preserves all duplicates so the matcher
    accepts the retriever returning *any* hash of the same source —
    relevant when re-parsing produces a new SHA but the document
    content is unchanged. Keys are :func:`strip_trailing_hash`-normalised
    (hash stripped + casefolded) so a case-different expected id still
    resolves to the full-variant set.
    """
    index_path = get_asset_index_path()
    if not index_path.exists():
        return {}
    bare_to_all: dict[str, list[str]] = {}
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
            full = row.get("asset_id", "")
            if not full:
                continue
            bare = strip_trailing_hash(full)
            if not bare:
                continue
            bare_to_all.setdefault(bare, [])
            if full not in bare_to_all[bare]:
                bare_to_all[bare].append(full)
    return bare_to_all


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

# ── Text-to-image (CLIP) queries on the Caltech-101 image set ──────────
# Each case is a free-text category name; the prefix-tolerant matcher
# picks up any image whose asset_id starts with ``Caltech <Category>``.
TEXT_TO_IMAGE_QUERIES: list[dict] = [
    {"query": "airplane", "expected_asset_ids": ["Caltech Airplanes"]},
    {"query": "motorbike", "expected_asset_ids": ["Caltech Motorbikes"]},
    {"query": "pizza", "expected_asset_ids": ["Caltech Pizza"]},
    {"query": "panda", "expected_asset_ids": ["Caltech Panda"]},
    {"query": "dolphin", "expected_asset_ids": ["Caltech Dolphin"]},
    {"query": "sunflower", "expected_asset_ids": ["Caltech Sunflower"]},
    {"query": "helicopter", "expected_asset_ids": ["Caltech Helicopter"]},
    {"query": "laptop computer", "expected_asset_ids": ["Caltech Laptop"]},
    {"query": "wristwatch", "expected_asset_ids": ["Caltech Watch"]},
    {"query": "saxophone", "expected_asset_ids": ["Caltech Saxophone"]},
    # Chinese queries on the same corpus (cross-language CLIP).
    {"query": "飞机", "expected_asset_ids": ["Caltech Airplanes"]},
    {"query": "熊猫", "expected_asset_ids": ["Caltech Panda"]},
    {"query": "披萨", "expected_asset_ids": ["Caltech Pizza"]},
]

# ── Image-to-image (CLIP) queries ──────────────────────────────────────
# ``image_path`` is the on-disk file used as the query vector. The
# expected ids use the ``Caltech <Category>`` prefix so the prefix-
# tolerant matcher picks up any of the 3 samples per category.
IMAGE_TO_IMAGE_QUERIES: list[dict] = [
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Airplanes 01_9fe67b3f.jpg",
        "expected_asset_ids": ["Caltech Airplanes"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Motorbikes 01_00a780a9.jpg",
        "expected_asset_ids": ["Caltech Motorbikes"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Pizza 01_ffa99a8f.jpg",
        "expected_asset_ids": ["Caltech Pizza"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Panda 01_3443a5d5.jpg",
        "expected_asset_ids": ["Caltech Panda"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Dolphin 01_bbd397c6.jpg",
        "expected_asset_ids": ["Caltech Dolphin"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Laptop 01_1b73ff27.jpg",
        "expected_asset_ids": ["Caltech Laptop"],
    },
]


@dataclass
class EvalResult:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool
    rank: int | None  # 1-based rank of the first hit, None if missed
    group: str  # "en", "zh", or "legacy"


def strip_trailing_hash(asset_id: str) -> str:
    """Normalise an asset id for eval matching.

    Drops a trailing ``_<8-hex>`` content-hash suffix (the per-parse
    SHA that distinguishes re-parses of the same source) and casefolds
    the remainder so ``Rich feature hierarchies`` matches
    ``Rich Feature Hierarchies for Accurate Object Detection And Semantic Segmentation_b857cf69``.
    Returns the original string (casefolded) when no hash suffix is
    present, so bare model names like ``Alexnet`` still compare cleanly.
    """
    if not asset_id:
        return ""
    if "_" in asset_id:
        head, _, tail = asset_id.rpartition("_")
        if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
            return head.casefold()
    return asset_id.casefold()


def _match(actual: list[str], expected: list[str]) -> int | None:
    """Return 1-based rank of the first hit, or ``None`` if missed.

    Matching is normalised + bidirectional-substring:

    1. Both ids are run through :func:`strip_trailing_hash` (drop the
       ``_<8-hex>`` hash suffix + ``casefold``) so a re-parse with a
       different SHA and a different casing still counts as the same
       document (the R-CNN failure: the retriever returned
       ``Rich Feature Hierarchies for Accurate Object Detection And Semantic Segmentation_b857cf69``
       but the expected was ``Rich feature hierarchies`` — case-sensitive
       substring + hash mismatch marked it a miss).
    2. A hit is ``norm_expected in norm_actual or norm_actual in norm_expected``
       — symmetric containment covers bare→full and full→bare pairs.
    """
    for rank, act in enumerate(actual, start=1):
        norm_act = strip_trailing_hash(act)
        for exp in expected:
            if not exp:
                continue
            norm_exp = strip_trailing_hash(exp)
            if not norm_exp:
                continue
            if norm_exp in norm_act or norm_act in norm_exp:
                return rank
    return None


def run_eval(top_k: int = 5) -> list[EvalResult]:
    """Run the full regression set against the live index.

    Returns a list of :class:`EvalResult` — one per case, in declared
    order. The combined EN + ZH + legacy sets are scored independently
    so the report can break down cross-language accuracy.
    """
    bare_to_all_fulls = _load_bare_to_all_fulls()
    results: list[EvalResult] = []
    for group, cases in (("en", EN_PAPER_QUERIES), ("zh", ZH_PAPER_QUERIES)):
        for case in cases:
            hits = hybrid_search(str(case["query"]), top_k=top_k)
            actual = [hit.asset_id for hit in hits]
            # Resolve expected ids to the set of full ids the index
            # actually returns. Accepts both bare titles and full
            # ``<title>_<hash>`` ids, and expands to all hash variants
            # of the same bare document so duplicate parses don't
            # count as misses.
            expected: list[str] = []
            for item in case["expected_asset_ids"]:
                bare = strip_trailing_hash(str(item))
                if bare in bare_to_all_fulls:
                    for f in bare_to_all_fulls[bare]:
                        if f not in expected:
                            expected.append(f)
                elif str(item) not in expected:
                    # User gave a bare or full id that didn't match any
                    # known asset — keep it as a literal expected so
                    # the strict match still has something to compare.
                    expected.append(str(item))
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


def _expand_bare_expected_to_full(bare_ids: list[str], full_ids: set[str]) -> list[str]:
    """Expand a list of bare expected ids (e.g. ``"Caltech Airplanes"``)
    to all full hashed asset_ids that start with the bare prefix.

    Used by the image-route evals so that the strict set match inside
    :func:`aggregate_metrics` accepts any of the 3 Caltech-101 samples
    as a valid hit instead of only the exact ``Caltech Airplanes 01_*``
    one would happen to test against.
    """
    expanded: list[str] = []
    for bare in bare_ids:
        matches = sorted(f for f in full_ids if f.startswith(bare))
        if matches:
            expanded.extend(matches)
        else:
            # Bare id isn't a prefix of any full id (e.g. user gave
            # the full id verbatim). Keep as-is so the strict match
            # still works for full-id cases.
            expanded.append(bare)
    return expanded


def _all_active_full_asset_ids() -> set[str]:
    """Return the set of full asset_ids (not the bare → full map) for
    the currently-active (non-deleted) rows in ``asset_index.jsonl``."""
    latest: dict[str, dict] = {}
    index_path = get_asset_index_path()
    if index_path.exists():
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
    return {row["asset_id"] for row in latest.values()}


def run_text_to_image_eval(top_k: int = 5) -> list[EvalResult]:
    """Run the text-to-image regression set against the CLIP-backed image index.

    Each case is a free-text query; ``qdrant_text_to_image_search``
    embeds the text with the same CLIP model used for image indexing
    and returns the top-k nearest images. The image collection must
    exist (run ``mmrag reindex --image-only`` after ingesting images).
    """
    from .backends.qdrant_backend import qdrant_text_to_image_search

    full_ids = _all_active_full_asset_ids()
    results: list[EvalResult] = []
    for case in TEXT_TO_IMAGE_QUERIES:
        hits = qdrant_text_to_image_search(str(case["query"]), top_k=top_k)
        actual = [hit.asset_id for hit in hits]
        expected = _expand_bare_expected_to_full(
            [str(item) for item in case["expected_asset_ids"]], full_ids
        )
        rank = _match(actual, expected)
        results.append(
            EvalResult(
                query=str(case["query"]),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=rank is not None,
                rank=rank,
                group="text_to_image",
            )
        )
    return results


def run_image_to_image_eval(cases: list[dict] | None = None, top_k: int = 5) -> list[EvalResult]:
    """Run the image-to-image regression set using a real image as query.

    Each case carries ``image_path`` (the on-disk file used to embed the
    query vector) and ``expected_asset_ids`` (the asset_id prefix or full
    id of the asset that *should* be in the top-k). The default set
    covers the 6 most-confident Caltech-101 categories that came up
    clean in the text-to-image sweep.
    """
    from .backends.qdrant_backend import qdrant_image_to_image_search

    if cases is None:
        cases = IMAGE_TO_IMAGE_QUERIES
    full_ids = _all_active_full_asset_ids()
    results: list[EvalResult] = []
    for case in cases:
        from pathlib import Path

        image_path = Path(case["image_path"])
        if not image_path.exists():
            results.append(
                EvalResult(
                    query=str(image_path),
                    expected_asset_ids=list(case["expected_asset_ids"]),
                    actual_asset_ids=[],
                    hit=False,
                    rank=None,
                    group="image_to_image",
                )
            )
            continue
        hits = qdrant_image_to_image_search(image_path, top_k=top_k)
        actual = [hit.asset_id for hit in hits]
        expected = _expand_bare_expected_to_full(
            [str(item) for item in case["expected_asset_ids"]], full_ids
        )
        rank = _match(actual, expected)
        results.append(
            EvalResult(
                query=str(image_path.name),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=rank is not None,
                rank=rank,
                group="image_to_image",
            )
        )
    return results


def _normalize_id_list(ids: list[str]) -> list[str]:
    """Normalise an id list for aggregate_metrics' strict set match.

    ``aggregate_metrics`` (in :mod:`mm_asset_rag.metrics`) uses exact
    set membership, so a re-parse with a different ``_<8-hex>`` suffix
    would otherwise count as a miss. We strip the hash + casefold here
    (keeping it in the eval harness only — metrics itself stays
    exact-set semantics for non-eval consumers).
    """
    out: list[str] = []
    for aid in ids:
        norm = strip_trailing_hash(aid)
        if norm and norm not in out:
            out.append(norm)
    return out


def write_eval_report(results: list[EvalResult], path=None) -> None:
    """Write per-query results + aggregate metrics to ``eval_report.json``.

    The aggregate block uses :func:`mm_asset_rag.metrics.aggregate_metrics`
    which computes hit_rate / precision / recall / f1 / ndcg at k=1,3,5,10
    plus MRR and MAP. Metrics are reported for the full set and per
    language group. Ids are normalised via :func:`strip_trailing_hash`
    before being handed to metrics so re-parses with a different content
    hash don't dilute the aggregate scores.
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
            [
                {
                    "actual_ids": _normalize_id_list(r.actual_asset_ids),
                    "expected_ids": _normalize_id_list(r.expected_asset_ids),
                }
                for r in rs
            ]
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
