"""Tests for mm_asset_rag.document_store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.document_store import read_documents, write_documents
from mm_asset_rag.paths import get_documents_jsonl
from mm_asset_rag.schema import ParsedDocument


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "docs.jsonl"
    docs = [
        ParsedDocument(text="alpha", metadata={"asset_id": "a", "page": 0}),
        ParsedDocument(text="beta", metadata={"asset_id": "b", "page": 1}),
    ]
    write_documents(docs, path=target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "alpha"

    read_back = read_documents(path=target)
    assert [d.text for d in read_back] == ["alpha", "beta"]
    assert [d.metadata["asset_id"] for d in read_back] == ["a", "b"]


def test_read_documents_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Document JSONL not found"):
        read_documents(path=tmp_path / "missing.jsonl")


def test_write_documents_uses_default_location(tmp_home: Path) -> None:

    docs = [ParsedDocument(text="hi", metadata={"asset_id": "x"})]
    write_documents(docs)
    assert get_documents_jsonl().exists()
    assert read_documents()[0].text == "hi"


def test_read_documents_skips_malformed_rows(tmp_path: Path) -> None:
    """A truncated / corrupted row from a mid-write crash must be skipped,
    not abort the whole index build. The surviving rows still load."""
    target = tmp_path / "docs.jsonl"
    target.write_text(
        # row 0: valid
        '{"text": "alpha", "metadata": {"asset_id": "a"}}\n'
        # row 1: truncated (half-written) — not valid JSON
        '{"text": "bet\n'
        # row 2: valid
        '{"text": "gamma", "metadata": {"asset_id": "c"}}\n',
        encoding="utf-8",
    )
    docs = read_documents(path=target)
    assert [d.text for d in docs] == ["alpha", "gamma"]


def test_read_documents_skips_row_missing_text(tmp_path: Path) -> None:
    """A row missing the ``text`` key (KeyError) is skipped, not fatal."""
    target = tmp_path / "docs.jsonl"
    target.write_text(
        '{"metadata": {"asset_id": "a"}}\n'  # no "text"
        '{"text": "beta", "metadata": {"asset_id": "b"}}\n',
        encoding="utf-8",
    )
    docs = read_documents(path=target)
    assert [d.text for d in docs] == ["beta"]
