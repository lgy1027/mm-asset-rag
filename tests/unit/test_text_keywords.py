"""Tests for ``mm_asset_rag.text_keywords``."""

from __future__ import annotations

import pytest

from mm_asset_rag.text_keywords import (
    enrich_chunk_text,
    extract_keywords,
    extract_keywords_zh,
)


def test_extract_keywords_zh_empty() -> None:
    assert extract_keywords_zh("") == []
    assert extract_keywords_zh("   \n  ") == []


def test_extract_keywords_zh_returns_top_k() -> None:
    text = "联宝 ESG 联宝 联宝 ESG 责任 责任 联宝 媒眼 安徽 安徽 外贸 外贸 联宝"
    kws = extract_keywords_zh(text, top_k=4)
    assert len(kws) <= 4
    # TextRank ranks by graph centrality, not raw frequency, so the
    # result may not include the most-frequent token — but every
    # keyword must come from the input and at least one is expected
    # to be the headline noun.
    assert all(isinstance(k, str) and k for k in kws)
    assert "联宝" in kws or "安徽" in kws  # both are in-text nouns


def test_extract_keywords_zh_strips_stopwords() -> None:
    text = "的 联宝 的 ESG 的 责任 联宝"
    kws = extract_keywords_zh(text, top_k=5)
    # Stopword characters should not appear as keywords.
    assert "的" not in "".join(kws)


def test_extract_keywords_zh_handles_pure_latin() -> None:
    """Pure Latin text should not crash the CJK extractor."""
    kws = extract_keywords_zh("Codex programming guide", top_k=3)
    # No CJK chars, no bigrams — empty result is fine.
    assert isinstance(kws, list)


def test_enrich_chunk_text_appends_keywords() -> None:
    out = enrich_chunk_text("body", ["联宝", "ESG"])
    assert "body" in out
    assert "联宝" in out
    assert "ESG" in out
    assert "关键词" in out  # labelled header for BM25 visibility


def test_enrich_chunk_text_no_keywords() -> None:
    assert enrich_chunk_text("body", []) == "body"


def test_extract_keywords_dispatch_zh() -> None:
    kws = extract_keywords("联宝 ESG 年度报告 联宝", top_k=3, language="zh")
    assert isinstance(kws, list)


def test_extract_keywords_unsupported_language() -> None:
    with pytest.raises(NotImplementedError):
        extract_keywords("foo bar", language="ja")


def test_extract_keywords_en_skips_stopwords() -> None:
    from mm_asset_rag.text_keywords import extract_keywords_en

    kws = extract_keywords_en("the cat and the dog", top_k=3)
    assert "the" not in kws
    assert "cat" in kws or "dog" in kws


def test_extract_keywords_dispatch_auto_falls_back_to_english() -> None:
    """``auto`` falls back to the English extractor when jieba is silent."""
    kws = extract_keywords("the cat sat on the mat", top_k=3, language="auto")
    assert any(k in kws for k in ("cat", "mat", "sat"))
    assert "the" not in kws
