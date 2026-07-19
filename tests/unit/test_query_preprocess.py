"""Tests for ``mm_asset_rag.query_preprocess``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.document_store import write_documents
from mm_asset_rag.paths import get_documents_jsonl
from mm_asset_rag.query_preprocess import invalidate_vocab_cache, preprocess
from mm_asset_rag.schema import ParsedDocument


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Each test gets a fresh ``$MM_ASSET_RAG_HOME`` so document
    store reads / writes don't pollute the real corpus.
    """
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(tmp_path))
    invalidate_vocab_cache()
    yield tmp_path
    invalidate_vocab_cache()


def _seed_corpus(home: Path, texts: list[str]) -> None:
    docs = [
        ParsedDocument(text=t, metadata={"asset_id": f"a{i}", "source_type": "pdf"})
        for i, t in enumerate(texts)
    ]
    write_documents(docs, get_documents_jsonl())


def test_preprocess_returns_structured_result(_isolated_home) -> None:
    _seed_corpus(_isolated_home, ["Resnet is a residual network."])
    pre = preprocess("Resnet")
    # Dense keeps original casing; BM25 lowercases.
    assert pre.dense_query == "Resnet"
    assert pre.bm25_query == "resnet"
    assert pre.corrections == {}


def test_preprocess_fixes_typos(_isolated_home) -> None:
    _seed_corpus(_isolated_home, ["transformer self-attention paper."])
    pre = preprocess("transformr")
    # "transformr" → "transformer" via fuzzy match (single char edit).
    assert pre.bm25_query == "transformer"
    assert pre.corrections == {"transformr": "transformer"}


def test_preprocess_fixes_uppercase_typos(_isolated_home) -> None:
    """Uppercase typos are corrected too — SequenceMatcher is case-sensitive
    and ``vocab`` is lowercase, so the comparison must run on the lowercased
    token or an all-caps typo would never match."""
    _seed_corpus(_isolated_home, ["transformer self-attention paper."])
    pre = preprocess("TRANSFORMR")
    # Dense keeps original casing; BM25 corrects + lowercases.
    assert pre.dense_query == "TRANSFORMR"
    assert pre.bm25_query == "transformer"
    assert pre.corrections == {"TRANSFORMR": "transformer"}


def test_preprocess_lowercase_keeps_chinese(_isolated_home) -> None:
    _seed_corpus(_isolated_home, ["Codex 全景指南"])
    pre = preprocess("Codex 全景指南")
    # Latin tokens lowercased; CJK block preserved.
    assert "codex" in pre.bm25_query
    assert "全景指南" in pre.bm25_query


def test_preprocess_skips_short_tokens(_isolated_home) -> None:
    _seed_corpus(_isolated_home, ["hello world."])
    pre = preprocess("hi ok")
    # 2-char tokens don't trigger fuzzy correction.
    assert pre.bm25_query == "hi ok"
    assert pre.corrections == {}


def test_preprocess_disabled_lowercase(_isolated_home, monkeypatch) -> None:
    _seed_corpus(_isolated_home, ["Resnet paper."])
    monkeypatch.setenv("QUERY_LOWERCASE", "false")
    pre = preprocess("Resnet")
    assert "Resnet" in pre.bm25_query


def test_preprocess_disabled_fuzzy(_isolated_home, monkeypatch) -> None:
    _seed_corpus(_isolated_home, ["transformer paper."])
    monkeypatch.setenv("QUERY_FUZZY", "false")
    pre = preprocess("transformr")
    # Without fuzzy the typo is preserved as-is.
    assert pre.bm25_query == "transformr"
    assert pre.corrections == {}


def test_preprocess_expansion_pairs(_isolated_home, monkeypatch, tmp_path) -> None:
    _seed_corpus(_isolated_home, ["resnet 残差"])
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(
        json.dumps({"resnet": ["残差", "residual"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUERY_EXPANSION", "true")
    monkeypatch.setenv("QUERY_EXPANSION_PAIRS", str(pairs_file))
    pre = preprocess("resnet")
    assert "残差" in pre.bm25_query
    assert "residual" in pre.bm25_query


def test_preprocess_empty_query(_isolated_home) -> None:
    pre = preprocess("")
    assert pre.raw == ""
    assert pre.dense_query == ""
    assert pre.bm25_query == ""
    assert pre.corrections == {}


def test_preprocess_handles_missing_vocab(_isolated_home) -> None:
    # No corpus seeded — vocab is empty, fuzzy should not crash.
    pre = preprocess("transformr")
    # Token left as-is; corrections empty.
    assert pre.corrections == {}
    assert pre.bm25_query == "transformr"
