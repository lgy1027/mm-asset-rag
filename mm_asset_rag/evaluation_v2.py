"""v2 regression set: 83 cases, Chinese-primary, multi-dimensional.

v2 is the new-generation retrieval eval set (83 cases, Chinese-primary
multi-dimensional: cross-language, multi-relevant, negative, typo,
image-to-image). It is opt-in: pass ``--v2`` on ``mmrag eval`` or set
``v2: true`` on ``POST /eval``; the default is still v1 so existing
scripts / dashboards keep their numbers.

Adds three new groups on top of the v1 ``EVAL_CASES`` (32 EN + 6 ZH
text→text) — focus is the actual content the user shipped in
``examples/data/chapter11_assets/``:

- 40 + 8 Chinese PDFs (联宝 series + Codex + AI 趋势 + Obsidian)
- 639 images: 50 Caltech categories x 2 naming variants (300) + 130
  Picsum + 20 OpenCV + 3 Chinese images (2026 KO 活动 / 联宝发展史 /
  联宝体育活动)
- v2 cases target: cross-language on the English corpus, multi-relevant
  queries (any-of), Chinese-only documents, negative samples, typo
  tolerance, multi-hop, and category fine-grained disambiguation.

Every case pairs a free-text ``query`` with one or more
``expected_asset_ids``. The ``_match`` helper uses prefix-tolerant
matching so a case "hits" if any expected id is a substring of any
actual id, or vice versa, so bare model names like ``clip`` still
match ``Learning Transferable Visual Models From Natural Language
Supervision_79e328a2`` once the search returns the full asset id.

Use :func:`run_eval_v2` to run text→text, :func:`run_text_to_image_eval_v2`
to run the new text→image cases, and :func:`run_image_to_image_eval_v2`
to run the new image→image cases. The full per-query results plus
aggregate metrics (hit_rate / precision / recall / f1 / ndcg + MRR + MAP)
are dumped to ``$MM_ASSET_RAG_HOME/eval_report_v2.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

from .metrics import _is_relevant, aggregate_metrics
from .paths import get_asset_index_path, get_eval_report

# ── ZH queries on the English arxiv corpus (cross-language) ────────────
# Tests whether the dense + Chinese-BM25 hybrid actually maps ZH keywords
# onto the right English paper. Expected ids use the model name so the
# prefix-tolerant matcher accepts any of the per-paper ``_NN_hash``
# variants as a valid hit.
V2_ZH_ON_EN_PAPERS: list[dict] = [
    {
        "query": "CLIP 模型",
        "expected_asset_ids": [
            "Learning Transferable Visual Models From Natural Language Supervision"
        ],
    },
    {"query": "BERT 预训练双向 transformer", "expected_asset_ids": ["Bert"]},
    {"query": "扩散模型 论文", "expected_asset_ids": ["Ddpm"]},
    {"query": "RAG 检索增强生成", "expected_asset_ids": ["Retrieval Augmented Generation"]},
    {"query": "YOLO 实时目标检测", "expected_asset_ids": ["You Only Look Once"]},
    {
        "query": "U-Net 医学图像分割",
        "expected_asset_ids": ["U-Net Convolutional Networks for Biomedical Image Segmentation"],
    },
    {"query": "残差网络 ResNet", "expected_asset_ids": ["Resnet"]},
    {"query": "变分自编码器 VAE", "expected_asset_ids": ["Vae"]},
    # Multi-relevant: any of these papers should hit.
    {
        "query": "目标检测模型",
        "expected_asset_ids": [
            "You Only Look Once",
            "Rich Feature Hierarchies For Accurate Object Detection And Semantic Segmentation",
            "Ssd",
            "Detr",
        ],
    },
    {
        "query": "图像分割方法",
        "expected_asset_ids": [
            "U-Net Convolutional Networks for Biomedical Image Segmentation",
            "Segment Anything",
        ],
    },
    {"query": "少样本学习", "expected_asset_ids": ["Flamingo", "Gpt3"]},
    {"query": "transformer 自注意力机制", "expected_asset_ids": ["Attention Is All You Need"]},
    # Specific section / concept
    {
        "query": "CLIP 的对比学习 contrastive loss",
        "expected_asset_ids": [
            "Learning Transferable Visual Models From Natural Language Supervision"
        ],
    },
    {"query": "BERT 的 masked language model 掩码语言模型", "expected_asset_ids": ["Bert"]},
    {
        "query": "U-Net 跳跃连接 skip connection",
        "expected_asset_ids": ["U-Net Convolutional Networks for Biomedical Image Segmentation"],
    },
    # Vague / long
    {"query": "讲图像分类的深度卷积神经网络", "expected_asset_ids": ["Alexnet"]},
    {"query": "transformer 用于视觉识别的", "expected_asset_ids": ["Vit"]},
    {"query": "讲 diffusion 去噪扩散的论文", "expected_asset_ids": ["Ddpm"]},
    # Synonym / paraphrase
    {"query": "Residual network 残差", "expected_asset_ids": ["Resnet"]},
    {"query": "embedding 词嵌入", "expected_asset_ids": ["Word2Vec", "Glove"]},
]

# ── EN queries, EN papers (paraphrase + multi-relevant) ─────────────────
V2_EN_ON_EN_PAPERS: list[dict] = [
    {"query": "image classification deep learning", "expected_asset_ids": ["Alexnet"]},
    {"query": "real-time object detection", "expected_asset_ids": ["You Only Look Once"]},
    {"query": "distributed word representations", "expected_asset_ids": ["Word2Vec"]},
    {"query": "image generation diffusion", "expected_asset_ids": ["Ddpm", "Stable Diffusion"]},
    {
        "query": "text to image generation",
        "expected_asset_ids": [
            "Stable Diffusion",
            "Learning Transferable Visual Models From Natural Language Supervision",
        ],
    },
    # Typos / case variance — tests fuzzy matching
    {"query": "transformr self attention", "expected_asset_ids": ["Attention Is All You Need"]},
    {"query": "RESNET residual learning", "expected_asset_ids": ["Resnet"]},
    {"query": "GAN generative adversarial nets", "expected_asset_ids": ["Gan"]},
    {"query": "BERT pretraining", "expected_asset_ids": ["Bert"]},
    {"query": "LORA parameter efficient", "expected_asset_ids": ["Lora"]},
]

# ── ZH queries on the Chinese (联宝/Codex/AI 趋势) corpus ───────────────
V2_ZH_ON_ZH_CORPUS: list[dict] = [
    {"query": "联宝 ESG 年度报告", "expected_asset_ids": ["责任联宝"]},
    {"query": "Codex 全景指南 AI 编程", "expected_asset_ids": ["所有深度用 AI 编程"]},
    {"query": "联宝 CES 未来 PC", "expected_asset_ids": ["CES 2026"]},
    {"query": "联宝 媒眼 安徽外贸", "expected_asset_ids": ["媒眼看联宝"]},
    {"query": "联宝 中试基地 省级备案", "expected_asset_ids": ["创新联宝"]},
    {"query": "Obsidian AI Skill 工具", "expected_asset_ids": ["Obsidian"]},
    # Cross-document, brand disambiguation
    {"query": "联宝 机器人 经开区", "expected_asset_ids": ["媒眼看联宝"]},
    {"query": "联宝 ESG", "expected_asset_ids": ["责任联宝"]},
    # corpus asset title is the连写 form (no spaces) — matching the
    # actual indexed title avoids a false miss in `_match`'s substring check.
    {"query": "联宝 2026 财年 启幕", "expected_asset_ids": ["敢AI敢为"]},
    # Cross-corpus (mixing English+Chinese)
    {"query": "transformer 论文 中文", "expected_asset_ids": ["Attention Is All You Need"]},
    # CLIP 论文在 corpus 里是英文原题,expected 用英文 title 才匹配;
    # 之前写中文译名导致 `_match` 字符串包含失败(假 miss)。
    {"query": "CLIP 中文版", "expected_asset_ids": ["Learning Transferable Visual Models"]},
]

# ── Negative samples (no expected hit) ──────────────────────────────────
# These deliberately probe the retriever's tendency to over-recall. A
# well-calibrated system should return an empty or low-confidence
# top-k; we mark a hit when *any* top-5 result is in the expected list,
# which is intentionally empty so every negative is a "miss" by design.
V2_NEGATIVE_QUERIES: list[dict] = [
    {"query": "强化学习算法 PPO DQN", "expected_asset_ids": []},
    {"query": "联邦学习框架", "expected_asset_ids": []},
    {"query": "元学习综述 few-shot learning survey", "expected_asset_ids": []},
    {"query": "图神经网络 GCN", "expected_asset_ids": []},
    {"query": "知识蒸馏综述 knowledge distillation", "expected_asset_ids": []},
    {"query": "推荐系统 deep learning recommendation", "expected_asset_ids": []},
    {"query": "speech recognition 语音识别", "expected_asset_ids": []},
    {"query": "neural machine translation 机器翻译", "expected_asset_ids": []},
]


# ── v2 text→image (CLIP, ZH-primary) ────────────────────────────────────
V2_TEXT_TO_IMAGE: list[dict] = [
    # Caltech categories in Chinese
    {"query": "飞机", "expected_asset_ids": ["Caltech Airplanes"]},
    {"query": "熊猫", "expected_asset_ids": ["Caltech Panda"]},
    {"query": "向日葵", "expected_asset_ids": ["Caltech Sunflower"]},
    {"query": "笔记本电脑", "expected_asset_ids": ["Caltech Laptop"]},
    {"query": "手表 腕表", "expected_asset_ids": ["Caltech Watch"]},
    {"query": "披萨 食物", "expected_asset_ids": ["Caltech Pizza"]},
    {"query": "海豚", "expected_asset_ids": ["Caltech Dolphin"]},
    {"query": "直升机", "expected_asset_ids": ["Caltech Helicopter"]},
    {"query": "萨克斯 乐器", "expected_asset_ids": ["Caltech Saxophone"]},
    {"query": "古董车 老爷车", "expected_asset_ids": ["Caltech Car Side"]},
    {"query": "手风琴", "expected_asset_ids": ["Caltech Accordion"]},
    {"query": "帆船 船", "expected_asset_ids": ["Caltech Ketch"]},
    {"query": "大象", "expected_asset_ids": ["Caltech Elephant"]},
    {"query": "大脑 MRI", "expected_asset_ids": ["Caltech Brain"]},
    # Chinese images (the new 3 we added)
    {"query": "KO 活动 2026", "expected_asset_ids": ["2026年KO活动"]},
    {"query": "联宝 发展史", "expected_asset_ids": ["联宝发展史"]},
    {"query": "联宝 体育活动", "expected_asset_ids": ["联宝体育活动"]},
    # English baseline
    {"query": "airplane", "expected_asset_ids": ["Caltech Airplanes"]},
    {"query": "panda bear", "expected_asset_ids": ["Caltech Panda"]},
    {"query": "sunflower flower", "expected_asset_ids": ["Caltech Sunflower"]},
    # Negative
    {"query": "毛绒玩具 plush toy", "expected_asset_ids": []},
    {"query": "高速公路 highway", "expected_asset_ids": []},
    {"query": "山脉 mountain", "expected_asset_ids": []},
]


# ── v2 image→image (CLIP, fine-grained) ────────────────────────────────
# Use the per-category ``01`` as query; expect any of ``02`` / ``03``
# from the same category to be in top-5. The bare category name
# expands to all 3 full ids via ``_expand_bare_expected_to_full`` at
# run time, so the strict set match works.
V2_IMAGE_TO_IMAGE: list[dict] = [
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
        "image_path": "examples/data/chapter11_assets/images/Caltech Sunflower 01_76ab29c7.jpg",
        "expected_asset_ids": ["Caltech Sunflower"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Laptop 01_1b73ff27.jpg",
        "expected_asset_ids": ["Caltech Laptop"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Helicopter 01_b81b5710.jpg",
        "expected_asset_ids": ["Caltech Helicopter"],
    },
    # Cross-domain: helicopter and airplanes are both aircraft, dolphin and sea-horse are both sea creatures.
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Helicopter 01_b81b5710.jpg",
        "expected_asset_ids": ["Caltech Helicopter", "Caltech Airplanes"],
    },
    {
        "image_path": "examples/data/chapter11_assets/images/Caltech Dolphin 01_bbd397c6.jpg",
        "expected_asset_ids": ["Caltech Dolphin", "Caltech Sea Horse"],
    },
]


# ── Runner / report writers (parallel to v1 helpers) ──────────────────


@dataclass
class V2Result:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool
    rank: int | None
    group: str
    # Parallel to actual_asset_ids; the LLM-derived paper title per hit (empty
    # when no title, e.g. image route or pre-AUTO_META index). Kept separate
    # so the reported actual_asset_ids stay plain ids while metrics still get
    # the title to match paper-title expected ids.
    actual_titles: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.actual_titles is None:
            self.actual_titles = []


def _bare_to_full(bare: str, full_ids: set[str]) -> str:
    """Resolve a bare expected id (e.g. ``Alexnet``) to a representative
    full hashed id. Used by v1's hybrid_search path; we re-export it
    here so v2 callers can build the same expected sets."""
    if bare in full_ids:
        return bare
    matches = sorted(f for f in full_ids if f.startswith(bare + "_") or f.startswith(bare + " "))
    return matches[0] if matches else bare


def _load_full_ids() -> set[str]:
    """Return the set of full asset_ids for the active rows.

    Multiple ``_NNN_hash`` variants of the same content can coexist
    in the index (re-ingestion of a slightly different file revision
    keeps the SHA256 stable but bumps a per-write hash). We dedupe by
    ``asset_id`` instead of ``sha256`` so ``_expand`` can return every
    hash variant — ``_match`` then accepts any of them as a valid hit.
    """
    seen: set[str] = set()
    out: set[str] = set()
    index_path = get_asset_index_path()
    if not index_path.exists():
        return out
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
            aid = row.get("asset_id", "")
            if aid in seen:
                continue
            seen.add(aid)
            out.add(aid)
    return out


def _expand(prefix: str, full_ids: set[str]) -> list[str]:
    """Expand a bare prefix (e.g. ``Caltech Airplanes``) to all full
    asset_ids that start with it. Returns ``[prefix]`` if no match
    so the strict match still works when the caller passed a full id.

    Multiple ``_NNN_hash`` variants of the same title are common (each
    parse run + content edit produces a new SHA). A single hash should
    *not* be treated as the canonical answer — the matcher below also
    accepts any actual id whose title is prefixed by the bare term.
    """
    matches = sorted(f for f in full_ids if f.startswith(prefix))
    return matches if matches else [prefix]


def _title_of(asset_id: str) -> str:
    """Strip the trailing ``_<8-hex-hash>`` from an asset id to get the
    bare title used for prefix-tolerant matching.

    Asset ids look like ``<title>_<8-hex>``. If the id has no ``_`` we
    return the whole id (e.g. for synthetic or user-supplied ids).
    """
    if "_" not in asset_id:
        return asset_id
    # The hash is the last ``_``-segment, exactly 8 lowercase hex chars.
    head, _, tail = asset_id.rpartition("_")
    if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
        return head
    return asset_id


def strip_trailing_hash(asset_id: str) -> str:
    """Normalise an asset id for eval matching.

    Drops a trailing ``_<8-hex>`` content-hash suffix and casefolds the
    remainder so ``Rich feature hierarchies`` matches
    ``Rich Feature Hierarchies for Accurate Object Detection And Semantic Segmentation_b857cf69``.
    Mirrors the v1 helper so both eval harnesses apply the same
    normalisation.
    """
    if not asset_id:
        return ""
    if "_" in asset_id:
        head, _, tail = asset_id.rpartition("_")
        if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
            return head.casefold()
    return asset_id.casefold()


def _actual_candidates(act: str | tuple[str, str]) -> list[str]:
    """Normalise an actual entry to the list of id-strings to compare.

    Accepts either a bare ``asset_id`` (str, for backwards compatibility with
    image-route evals and unit tests) or an ``(asset_id, title)`` pair. For the
    pair, both the asset_id and the title are candidates — a hit on either
    counts as relevant. This is what lets the CLIP case resolve when the
    retriever returns ``("clip_b14b418e", "Learning Transferable Visual Models
    From Natural Language Supervision")`` and the expected id is the paper's
    canonical title: the asset_id alone never matches, but the LLM-derived
    title does.
    """
    if isinstance(act, tuple):
        return [v for v in act if v]
    return [act] if act else []


def _match(actual: list[str | tuple[str, str]], expected: list[str]) -> int | None:
    # Relevance is the normalised bidirectional-substring check shared with
    # ``aggregate_metrics`` (via ``metrics._is_relevant``), so per-query
    # ``hit``/``rank`` and the aggregate ``hit_rate``/``MRR``/``NDCG`` agree
    # by construction — a bare short title matches a longer returned id both
    # here and in the reported metrics. Each actual may carry a title too
    # (see :func:`_actual_candidates`); the asset_id or the title matching any
    # expected id is a hit at that rank.
    for rank, act in enumerate(actual, start=1):
        if any(_is_relevant(c, expected) for c in _actual_candidates(act)):
            return rank
    return None


def run_eval_v2(top_k: int = 5) -> list[V2Result]:
    """Run the v2 text→text regression set (CLI / API entry point).

    Thin alias for :func:`run_text_to_text_eval_v2` so the CLI ``--v2``
    flag and the API ``v2: true`` field can treat v1 and v2 symmetrically
    (``run_eval`` vs ``run_eval_v2``). Returns the same :class:`V2Result`
    shape as the other v2 runners, which parallels v1's ``EvalResult`` so
    callers can swap the two without adapting the response shape.

    The text→image / image→image groups have their own runners; this
    convenience only covers the text→text set because that is what v1's
    ``run_eval`` covers and what the default ``mmrag eval`` output
    compares against.
    """
    return run_text_to_text_eval_v2(top_k=top_k)


def run_text_to_text_eval_v2(
    top_k: int = 5,
    *,
    search_fn: Callable[[str, int], list] | None = None,
    full_ids: set[str] | None = None,
) -> list[V2Result]:
    """Run all v2 text→text cases against the live hybrid index.

    ``search_fn`` defaults to ``retrieval.hybrid_search`` (production
    path). Tests pass a stub that returns canned ``SearchHit`` lists
    so the eval can run offline against a mock corpus — this is the
    ``auto-eval`` integration used by CI. The stub signature is
    ``(query: str, top_k: int) -> list[SearchHit]``; the query
    preprocessor / RRF / min_score are bypassed because the test
    ships its own pre-computed results.

    ``full_ids`` is the set of known asset_ids used by ``_expand`` to
    resolve bare expected ids. Tests inject a synthetic set so the
    eval runs without a real ``asset_index.jsonl``.
    """
    from .retrieval import hybrid_search

    search = search_fn or hybrid_search
    ids = full_ids if full_ids is not None else _load_full_ids()

    out: list[V2Result] = []
    for group, cases in (
        ("zh_on_en", V2_ZH_ON_EN_PAPERS),
        ("en_on_en", V2_EN_ON_EN_PAPERS),
        ("zh_on_zh", V2_ZH_ON_ZH_CORPUS),
        ("negative", V2_NEGATIVE_QUERIES),
    ):
        for case in cases:
            hits = search(str(case["query"]), top_k)
            # Carry (asset_id, title) pairs for _match so it can hit on either.
            # The asset_id is a filename stem (e.g. clip_b14b418e) which never
            # matches a paper-title expected id; with AUTO_META on, hit.title is
            # the LLM-derived canonical paper title, which does. The reported
            # actual_asset_ids stay plain asset_ids for readability.
            actual_pairs: list[tuple[str, str]] = [(hit.asset_id, hit.title or "") for hit in hits]
            expected: list[str] = []
            for item in case["expected_asset_ids"]:
                expected.extend(_expand(str(item), ids))
            rank = _match(actual_pairs, expected) if expected else None
            out.append(
                V2Result(
                    query=str(case["query"]),
                    expected_asset_ids=expected,
                    actual_asset_ids=[hit.asset_id for hit in hits],
                    hit=rank is not None,
                    rank=rank,
                    group=group,
                    actual_titles=[hit.title or "" for hit in hits],
                )
            )
    return out


def run_text_to_image_eval_v2(
    top_k: int = 5,
    *,
    search_fn: Callable[[str, int], list] | None = None,
    full_ids: set[str] | None = None,
) -> list[V2Result]:
    """Run the v2 text→image cases against the Qdrant image collection.

    ``search_fn`` is the dependency-injection hook for tests; default
    is the live ``qdrant_text_to_image_search`` call. See
    :func:`run_text_to_text_eval_v2` for the same pattern.
    """
    from .backends.qdrant_backend import qdrant_text_to_image_search

    search = search_fn or qdrant_text_to_image_search
    ids = full_ids if full_ids is not None else _load_full_ids()

    out: list[V2Result] = []
    for case in V2_TEXT_TO_IMAGE:
        hits = search(str(case["query"]), top_k)
        actual = [hit.asset_id for hit in hits]
        expected: list[str] = []
        for item in case["expected_asset_ids"]:
            expected.extend(_expand(str(item), ids))
        rank = _match(actual, expected) if expected else None
        out.append(
            V2Result(
                query=str(case["query"]),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=rank is not None,
                rank=rank,
                group="text_to_image",
            )
        )
    return out


def run_image_to_image_eval_v2(top_k: int = 5) -> list[V2Result]:
    """Run the v2 image→image cases."""
    from .backends.qdrant_backend import qdrant_image_to_image_search

    full_ids = _load_full_ids()
    out: list[V2Result] = []
    for case in V2_IMAGE_TO_IMAGE:
        image_path = Path(case["image_path"])
        if not image_path.exists():
            out.append(
                V2Result(
                    query=str(image_path.name),
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
        expected: list[str] = []
        for item in case["expected_asset_ids"]:
            expected.extend(_expand(str(item), full_ids))
        rank = _match(actual, expected) if expected else None
        out.append(
            V2Result(
                query=str(image_path.name),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=rank is not None,
                rank=rank,
                group="image_to_image",
            )
        )
    return out


def _normalize_id_list(ids: list[str]) -> list[str]:
    """Strip ``_<8-hex>`` hash suffixes + casefold, dedup preserving order.

    Used to collapse duplicate hash variants of the same document before
    writing them into the eval report's ``expected_ids``/``actual_ids``
    fields (so a re-parse producing a new hash doesn't bloat the stored
    lists). :mod:`mm_asset_rag.metrics` applies its own normalisation on
    top when computing relevance, so this pre-pass is purely cosmetic for
    the report payload — it does not change any metric value.
    """
    out: list[str] = []
    for aid in ids:
        norm = strip_trailing_hash(aid)
        if norm and norm not in out:
            out.append(norm)
    return out


def write_eval_report_v2(results_by_group: dict[str, list[V2Result]], path=None) -> None:
    """Write per-query results + per-group aggregate metrics to JSON."""
    target = path or get_eval_report().with_name("eval_report_v2.json")

    def _agg(rs: list[V2Result]) -> dict:
        if not rs:
            return {}
        return aggregate_metrics(
            [
                {
                    # Pair each actual asset_id with its title so metrics can
                    # match on the LLM-derived paper title too (matches the
                    # per-query _match contract → aggregate stays self-consistent
                    # with per_query hit/rank). Fall back to bare asset_id when
                    # no titles (image route / old reports). zip_longest (not
                    # zip) guards against a future code path where the two lists
                    # diverge in length — silent truncation would drop trailing
                    # asset_ids and skew the metrics.
                    "actual_ids": (
                        [
                            (aid, t) if t else aid
                            for aid, t in zip_longest(
                                r.actual_asset_ids, r.actual_titles, fillvalue=""
                            )
                        ]
                        if r.actual_titles
                        else list(r.actual_asset_ids)
                    ),
                    "expected_ids": _normalize_id_list(r.expected_asset_ids),
                }
                for r in rs
            ]
        )

    payload = {
        "version": "v2",
        "per_group": {
            g: {
                "total": len(rs),
                "hits": sum(1 for r in rs if r.hit),
                "hit_rate": sum(1 for r in rs if r.hit) / max(len(rs), 1),
                "metrics": _agg(rs),
                "per_query": [asdict(r) for r in rs],
            }
            for g, rs in results_by_group.items()
        },
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    from .config import load_env

    load_env()
    t2t = run_text_to_text_eval_v2(top_k=5)
    t2i = run_text_to_image_eval_v2(top_k=5)
    i2i = run_image_to_image_eval_v2(top_k=5)
    by_group: dict[str, list[V2Result]] = {
        "text_to_text": t2t,
        "text_to_image": t2i,
        "image_to_image": i2i,
    }
    write_eval_report_v2(by_group)
    for g, rs in by_group.items():
        hits = sum(1 for r in rs if r.hit)
        print(f"{g}: {hits}/{len(rs)} hit_rate={hits / max(len(rs), 1):.3f}")
