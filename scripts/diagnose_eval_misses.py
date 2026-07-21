#!/usr/bin/env python3
"""Diagnose why specific eval queries miss their target paper.

For each query in the miss list, prints:
- hybrid_search top-10 (what actually got returned)
- qdrant_text_search top-10 (the dense+BM25+BM25-zh RRF route alone)
- qdrant_text_to_image_search top-5 (CLIP route)
- whether the target paper appears anywhere in each route, and at what rank

Use this to localise B-class misses (target not in top-5) to a route:
dense / BM25-zh cross-language weakness / CLIP / RRF fusion dilution.

Usage::

    python scripts/diagnose_eval_misses.py
"""

from __future__ import annotations

from mm_asset_rag.evaluation import EVAL_CASES  # noqa: F401  (ensures import side)

MISSES = [
    ("ImageNet classification with deep convolutional neural networks", "alexnet"),
    ("learning transferable visual models natural language supervision contrastive", "clip"),
    ("GloVe global vectors for word representation", "glove"),
    ("deep residual learning image recognition ResNet", "resnet"),
    ("讲生成对抗网络 GAN 的论文", "gan"),
    ("去噪扩散概率模型", "ddpm"),
    ("图像分类 深度卷积神经网络 AlexNet", "alexnet"),
    ("找和 CLIP 图文对齐有关的资料", "clip"),
]


def _norm(s: str) -> str:
    import re

    s = s.lower()
    s = re.sub(r"[\s\-_]+", "-", s)
    # strip trailing hash segments
    while "_" in s:
        h, _, t = s.rpartition("_")
        if len(t) == 8 and all(c in "0123456789abcdef" for c in t):
            s = h
        else:
            break
    return s.strip("-")


def _find_target(hits: list, target: str) -> int | None:
    """1-based rank of the first hit whose asset_id or title matches target.

    Uses bidirectional substring (``a in b or b in a``) on normalised slugs to
    match the production ``metrics._matches`` contract — a single-direction
    ``nt in cand`` would miss the case where the target slug is longer than
    the candidate (e.g. bare-asset-id route where cand is a short stem) and
    falsely report ``NOT FOUND``, mis-attributing a B-class miss.
    """
    nt = _norm(target)
    for i, h in enumerate(hits, 1):
        for cand in (h.asset_id, getattr(h, "title", "") or ""):
            nc = _norm(cand)
            if nt and nc and (nt in nc or nc in nt):
                return i
    return None


def main() -> int:
    from mm_asset_rag.backends.qdrant_backend import (
        qdrant_text_search,
        qdrant_text_to_image_search,
    )
    from mm_asset_rag.retrieval import hybrid_search

    for query, target in MISSES:
        print(f"\n{'=' * 72}")
        print(f"query: {query}")
        print(f"target paper: {target}")
        try:
            hybrid = hybrid_search(query, top_k=10)
        except Exception as exc:  # pragma: no cover
            print(f"  hybrid_search failed: {exc}")
            hybrid = []
        try:
            text_route = qdrant_text_search(query, top_k=10)
        except Exception as exc:  # pragma: no cover
            print(f"  qdrant_text_search failed: {exc}")
            text_route = []
        try:
            t2i = qdrant_text_to_image_search(query, top_k=5)
        except Exception:  # pragma: no cover
            t2i = []

        print(f"\n  hybrid top-{len(hybrid)}:")
        for i, h in enumerate(hybrid, 1):
            print(f"    {i}. {h.asset_id[:48]:48s} sc={h.score:.4f} route={h.route}")

        print(f"\n  qdrant_text (dense+BM25+BM25-zh) top-{len(text_route)}:")
        for i, h in enumerate(text_route, 1):
            print(f"    {i}. {h.asset_id[:48]:48s} sc={h.score:.4f}")

        if t2i:
            print(f"\n  qdrant_text_to_image (CLIP) top-{len(t2i)}:")
            for i, h in enumerate(t2i, 1):
                print(f"    {i}. {h.asset_id[:48]:48s} sc={h.score:.4f}")

        hr = _find_target(hybrid, target)
        tr = _find_target(text_route, target)
        t2ir = _find_target(t2i, target) if t2i else None
        print(
            f"\n  target '{target}' rank: hybrid={'#' + str(hr) if hr else 'NOT FOUND'}, "
            f"text_route={'#' + str(tr) if tr else 'NOT FOUND'}"
            + (f", t2i={'#' + str(t2ir) if t2ir else 'NOT FOUND'}" if t2i else "")
        )
        if not hr and tr:
            print("  >>> target IS in text route but lost in hybrid fusion → RRF dilution")
        elif not hr and not tr:
            print(
                "  >>> target NOT in text route → dense/BM25 recall failure (tune BM25/dense/cross-lang)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
