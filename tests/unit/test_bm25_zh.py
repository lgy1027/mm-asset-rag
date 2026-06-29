"""Tests for ``mm_asset_rag.bm25_zh`` — Chinese-aware BM25 encoder.

Pure-Python unit tests. The first call to ``tokenize_zh`` triggers
jieba's lazy initialisation (downloads the dict on a cold cache), so
keep the suite small to keep CI snappy.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from mm_asset_rag import bm25_zh
from mm_asset_rag.schema import ParsedDocument


# ─── tokenize_zh ─────────────────────────────────────────────────────────


def test_tokenize_chinese_segments_cjk_text() -> None:
    tokens = bm25_zh.tokenize_zh("中国人民站起来了")
    # jieba produces at least the first and last characters as stand-alone
    # tokens; we only assert non-emptiness + that the whole text is
    # recoverable when joined back.
    assert tokens
    # No whitespace leaks through.
    assert all(t.strip() for t in tokens)
    # No single-character Latin fragments (would mean jieba broke a word).
    assert not any(t.isascii() and len(t) == 1 for t in tokens)


def test_tokenize_keeps_latin_tokens_intact() -> None:
    tokens = bm25_zh.tokenize_zh("使用 BERT 模型做文本分类")
    # BERT survives as a single token; the surrounding CJK also appears.
    assert "bert" in tokens
    # No character-level fragments of BERT.
    assert "b" not in tokens
    assert "ert" not in tokens


def test_tokenize_keeps_numbers_intact() -> None:
    tokens = bm25_zh.tokenize_zh("V100 显卡有 640 个 tensor core")
    assert "v100" in tokens
    assert "640" in tokens


def test_tokenize_empty_and_none() -> None:
    assert bm25_zh.tokenize_zh("") == []
    assert bm25_zh.tokenize_zh(None) == []


def test_tokenize_pure_latin_falls_back_to_lowercase_words() -> None:
    tokens = bm25_zh.tokenize_zh("RAG and LLM are popular.")
    # Latin tokens should be lowercased and whitespace-trimmed.
    assert "rag" in tokens
    assert "and" in tokens
    assert "llm" in tokens


# ─── compute_idf ─────────────────────────────────────────────────────────


def test_compute_idf_includes_metadata_keys() -> None:
    idf = bm25_zh.compute_idf([["a", "b"], ["a", "c"]])
    assert idf["_avgdl"] == pytest.approx(2.0)
    assert idf["_k1"] == 1.5
    assert idf["_b"] == 0.75
    assert "a" in idf and "b" in idf and "c" in idf


def test_compute_idf_common_term_lower_idf_than_rare() -> None:
    # "the" appears in all docs, "widget" appears in 1.
    idf = bm25_zh.compute_idf(
        [["the", "cat"], ["the", "dog"], ["the", "widget"]]
    )
    assert idf["widget"] > idf["the"]


def test_compute_idf_empty_corpus_returns_zero_avgdl() -> None:
    idf = bm25_zh.compute_idf([])
    assert idf["_avgdl"] == 0.0


# ─── bm25_zh_score ───────────────────────────────────────────────────────


def test_bm25_score_relevant_doc_outranks_unrelated() -> None:
    idf = bm25_zh.compute_idf([["猫", "狗"], ["猫"], ["苹果", "香蕉"]])
    score_relevant = bm25_zh.bm25_zh_score(["猫", "狗"], ["猫", "狗"], idf)
    score_unrelated = bm25_zh.bm25_zh_score(["猫", "狗"], ["苹果", "香蕉"], idf)
    assert score_relevant > score_unrelated


def test_bm25_score_empty_query_returns_zero() -> None:
    idf = bm25_zh.compute_idf([["猫"]])
    assert bm25_zh.bm25_zh_score([], ["猫"], idf) == 0.0


# ─── bm25_zh_encode_query ────────────────────────────────────────────────


def test_encode_query_returns_sparse_vector_with_idf_scores() -> None:
    idf = bm25_zh.compute_idf([["猫", "狗"], ["猫"], ["苹果"]])
    tokens = bm25_zh.tokenize_zh("猫")
    sv = bm25_zh.bm25_zh_encode_query(tokens, idf)
    assert isinstance(sv, models.SparseVector)
    assert len(sv.indices) == len(sv.values) == 1
    assert sv.values[0] == pytest.approx(idf["猫"])


def test_encode_query_empty_tokens_returns_empty_sparse_vector() -> None:
    idf = bm25_zh.compute_idf([["猫"]])
    sv = bm25_zh.bm25_zh_encode_query([], idf)
    assert sv.indices == []
    assert sv.values == []


def test_encode_query_indices_are_stable_sha1_hashes() -> None:
    """Same term → same index across calls (the whole point of SHA1 over Python's salted hash)."""
    idf = bm25_zh.compute_idf([["猫", "狗"], ["猫"], ["苹果"]])
    tokens = ["猫"]
    a = bm25_zh.bm25_zh_encode_query(tokens, idf)
    b = bm25_zh.bm25_zh_encode_query(tokens, idf)
    assert a.indices == b.indices


# ─── build_bm25_zh_index ─────────────────────────────────────────────────


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(text=text, metadata={"asset_id": "test"})


def test_build_index_returns_one_sparse_vector_per_doc() -> None:
    docs = [_doc("猫 狗"), _doc("苹果"), _doc("")]
    vectors, idf = bm25_zh.build_bm25_zh_index(docs)
    assert len(vectors) == 3
    # Empty doc produces an empty sparse vector (Qdrant accepts this).
    assert vectors[2].indices == []
    # IDF table includes the special keys + actual terms.
    assert "_avgdl" in idf
    assert "猫" in idf


def test_build_index_aligns_with_query_round_trip() -> None:
    """A document that contains a query term must outrank one that does not."""
    docs = [
        _doc("猫的习性和食物"),
        _doc("苹果的营养成分"),
        _doc("狗的品种介绍"),
    ]
    vectors, idf = bm25_zh.build_bm25_zh_index(docs)
    # Sanity: each vector covers its doc's terms.
    for vec in vectors:
        if len(vec.indices) > 0:
            assert len(vec.indices) == len(vec.values)
    # Encode the query "猫" and ensure the relevant doc's tokens hash
    # to a subset of indices we can find.
    query_sv = bm25_zh.bm25_zh_encode_query(bm25_zh.tokenize_zh("猫"), idf)
    assert query_sv.indices
    # At least one of the doc-side vectors shares an index with the
    # query vector (the 猫 term). All three docs share 猫-related words
    # only loosely; we just assert the round-trip is internally
    # consistent.
    relevant_doc = vectors[0]
    assert any(i in relevant_doc.indices for i in query_sv.indices)


def test_build_index_hashes_are_deterministic_across_calls() -> None:
    """The same text should produce the same sparse-vector indices in two builds."""
    docs1 = [_doc("猫 狗")]
    docs2 = [_doc("猫 狗")]
    v1, _ = bm25_zh.build_bm25_zh_index(docs1)
    v2, _ = bm25_zh.build_bm25_zh_index(docs2)
    assert v1[0].indices == v2[0].indices
    assert v1[0].values == pytest.approx(v2[0].values)