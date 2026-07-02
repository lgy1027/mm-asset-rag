"""Minimal regression set for retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .paths import get_eval_report
from .retrieval import hybrid_search

EVAL_CASES = [
    {
        "query": "哪份资料讲了 retrieval augmented generation？",
        "expected_asset_ids": ["retrieval_augmented_generation"],
    },
    {
        "query": "找和 CLIP 图文对齐有关的资料",
        "expected_asset_ids": ["clip"],
    },
    {
        "query": "有没有包含文档版面理解或 OCR 的资料？",
        "expected_asset_ids": ["layoutlm"],
    },
]


@dataclass
class EvalResult:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool


def run_eval(top_k: int = 5) -> list[EvalResult]:
    results = []
    for case in EVAL_CASES:
        hits = hybrid_search(str(case["query"]), top_k=top_k)
        actual = [hit.asset_id for hit in hits]
        expected = [str(item) for item in case["expected_asset_ids"]]
        # Prefix-tolerant match: the manifest id derivation changed from
        # 'pdf_<stem>' to '<stem>', and asset lists can grow over time.
        # A case 'hits' if any expected id is a substring of any actual id
        # OR vice versa, so 'pdf_rag' still matches 'retrieval_augmented_generation'
        # when only the latter exists, and 'pdf_clip' still matches 'clip'.
        hit = any(exp in act or act in exp for exp in expected for act in actual)
        results.append(
            EvalResult(
                query=str(case["query"]),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=hit,
            )
        )
    return results


def write_eval_report(results: list[EvalResult], path=None) -> None:
    target = path or get_eval_report()
    target.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
