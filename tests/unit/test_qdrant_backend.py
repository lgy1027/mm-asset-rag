"""Tests for ``mm_asset_rag.backends.qdrant_backend``.

Covers the BM25 Okapi helpers used by ``_select_top_chunks_per_pdf`` —
pure functions, no Qdrant / no embedding model required.
"""

from __future__ import annotations

import pytest

from mm_asset_rag.backends.qdrant_backend import (
    _bm25_okapi_scores,
    _select_top_chunks_per_pdf,
    _tokenize_for_bm25,
)
from mm_asset_rag.schema import ParsedDocument


def _doc(text: str, asset_id: str, title: str | None = None) -> ParsedDocument:
    return ParsedDocument(
        text=text,
        metadata={"asset_id": asset_id, "asset_title": title or asset_id},
    )


# ─── _tokenize_for_bm25 ─────────────────────────────────────────────────


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert _tokenize_for_bm25("LayoutLM: Pre-training of Text") == [
        "layoutlm",
        "pre",
        "training",
        "of",
        "text",
    ]


def test_tokenize_empty_and_punctuation_only() -> None:
    assert _tokenize_for_bm25("") == []
    assert _tokenize_for_bm25("!!! ... ---") == []


def test_tokenize_drops_empty_tokens() -> None:
    assert _tokenize_for_bm25("  BERT  ") == ["bert"]


# ─── _bm25_okapi_scores ─────────────────────────────────────────────────


def test_bm25_okapi_empty_inputs() -> None:
    assert _bm25_okapi_scores([], []) == []
    assert _bm25_okapi_scores(["bert"], []) == []


def test_bm25_okapi_relevant_doc_scores_higher() -> None:
    """A doc that contains all query terms must outrank one that contains none."""
    docs = [
        ["this", "is", "a", "passage", "about", "bert", "and", "transformers"],
        ["completely", "unrelated", "fish", "and", "chips"],
    ]
    scores = _bm25_okapi_scores(["bert", "transformer"], docs)
    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_bm25_okapi_idf_increases_with_rarity() -> None:
    """Terms that appear in fewer docs get a higher IDF contribution."""
    # "bert" appears in 1/3 docs; "the" appears in 3/3.
    docs = [
        ["bert", "lives", "here"],
        ["the", "cat", "sat"],
        ["the", "dog", "ran"],
    ]
    rare_score = _bm25_okapi_scores(["bert"], [docs[0]])[0]
    common_score = _bm25_okapi_scores(["the"], [docs[0]])[0]
    assert rare_score > common_score


# ─── _select_top_chunks_per_pdf ──────────────────────────────────────────


def test_select_top_chunks_returns_input_when_cap_is_none() -> None:
    docs = [_doc("a", "bert"), _doc("b", "bert"), _doc("c", "bert")]
    assert _select_top_chunks_per_pdf(docs, None) == docs


def test_select_top_chunks_returns_input_when_below_cap() -> None:
    docs = [_doc("a", "bert"), _doc("b", "bert")]
    assert _select_top_chunks_per_pdf(docs, 5) == docs
    # Input is not mutated.
    assert len(docs) == 2


def test_select_top_chunks_caps_oversized_pdf() -> None:
    """A 5-chunk PDF capped at 3 returns exactly 3 chunks for that asset."""
    docs = [
        _doc(
            "bert is the bidirectional encoder representation from transformers",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "we introduce a new language representation model called bert",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "bert achieves state of the art on eleven natural language "
            "processing tasks. completely unrelated cooking recipes follow",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "appendix: hyperparameter settings and additional ablations on "
            "the bert pretraining objective",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
        _doc(
            "this passage talks about penguins and arctic wildlife",
            "bert",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ),
    ]
    selected = _select_top_chunks_per_pdf(docs, max_per_pdf=3)
    assert len(selected) == 3
    # The lowest-scoring chunk (the off-topic penguins one) should be dropped.
    kept_texts = " ".join(d.text for d in selected)
    assert "penguins" not in kept_texts


def test_select_top_chunks_handles_multiple_assets_independently() -> None:
    """The cap is per-asset, not global."""
    bert_docs = [_doc(f"bert passage {i}", "bert") for i in range(4)]
    clip_docs = [_doc(f"clip passage {i}", "clip") for i in range(6)]
    selected = _select_top_chunks_per_pdf(bert_docs + clip_docs, max_per_pdf=3)
    by_asset = {d.metadata["asset_id"]: d for d in selected}
    # Order is preserved within each asset group (per-asset cap), but
    # we just check counts here.
    counts = {}
    for d in selected:
        counts[d.metadata["asset_id"]] = counts.get(d.metadata["asset_id"], 0) + 1
    assert counts == {"bert": 3, "clip": 3}


def test_select_top_chunks_falls_back_on_empty_title() -> None:
    """An asset with no title falls back to the asset_id rewritten with spaces."""
    docs = [
        _doc("text", "layout_l_m", title=""),
        _doc("layout_l_m is great", "layout_l_m", title=""),
        _doc("unrelated", "layout_l_m", title=""),
    ]
    selected = _select_top_chunks_per_pdf(docs, max_per_pdf=2)
    assert len(selected) == 2


def test_select_top_chunks_does_not_mutate_input() -> None:
    docs = [
        _doc("alpha", "a"),
        _doc("beta", "a"),
        _doc("gamma", "a"),
        _doc("delta", "a"),
    ]
    original_order = [d.text for d in docs]
    _select_top_chunks_per_pdf(docs, max_per_pdf=2)
    assert [d.text for d in docs] == original_order