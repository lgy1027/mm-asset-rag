"""Tests for mm_asset_rag.document_store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.document_store import read_documents, write_documents
from mm_asset_rag.knowledge_models import (
    AccessPolicy,
    Asset,
    Chunk,
    Document,
    Source,
)
from mm_asset_rag.paths import get_documents_jsonl
from mm_asset_rag.schema import ParsedChunk


def _chunk(text: str, ordinal: int = 0) -> Chunk:
    source = Source(source_id="upload:handbook")
    document = Document(
        document_id="handbook",
        title="Handbook",
        source=source,
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )
    return Chunk.create(
        document=document,
        asset=Asset(content_hash="a" * 64, source_type="pdf", relative_path="pdfs/handbook.pdf"),
        ordinal=ordinal,
        text=text,
        source=source,
        access_policy=document.access_policy,
    )


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "docs.jsonl"
    docs = [_chunk("alpha"), _chunk("beta", ordinal=1)]
    write_documents(docs, path=target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "alpha"

    read_back = read_documents(path=target)
    assert [d.text for d in read_back] == ["alpha", "beta"]
    assert [d.document_id for d in read_back] == ["handbook", "handbook"]


def test_read_documents_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Document JSONL not found"):
        read_documents(path=tmp_path / "missing.jsonl")


def test_write_documents_uses_default_location(tmp_home: Path) -> None:

    docs = [_chunk("hi")]
    write_documents(docs)
    assert get_documents_jsonl().exists()
    assert read_documents()[0].text == "hi"


def test_read_documents_skips_malformed_rows(tmp_path: Path) -> None:
    """A truncated / corrupted row from a mid-write crash must be skipped,
    not abort the whole index build. The surviving rows still load."""
    target = tmp_path / "docs.jsonl"
    target.write_text(
        # row 0: old asset-only row, which is deliberately unsupported
        '{"text": "alpha", "metadata": {"asset_id": "a"}}\n'
        # row 1: truncated (half-written) — not valid JSON
        '{"text": "bet\n'
        # row 2: another old asset-only row
        '{"text": "gamma", "metadata": {"asset_id": "c"}}\n',
        encoding="utf-8",
    )
    docs = read_documents(path=target)
    assert docs == []


def test_read_documents_skips_row_missing_text(tmp_path: Path) -> None:
    """A row missing the ``text`` key (KeyError) is skipped, not fatal."""
    target = tmp_path / "docs.jsonl"
    target.write_text(
        '{"metadata": {"asset_id": "a"}}\n'  # no v2 identity
        '{"text": "beta", "metadata": {"asset_id": "b"}}\n',
        encoding="utf-8",
    )
    docs = read_documents(path=target)
    assert docs == []


def test_read_documents_rejects_v2_row_missing_access_policy(tmp_path: Path) -> None:
    target = tmp_path / "docs.jsonl"
    row = _chunk("alpha").to_record()
    row.pop("access_policy")
    target.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert read_documents(path=target) == []


@pytest.mark.parametrize("identity", ["chunk"])
def test_read_documents_rejects_tampered_identity_ids(tmp_path: Path, identity: str) -> None:
    target = tmp_path / "docs.jsonl"
    row = _chunk("alpha").to_record()
    row["chunk_id"] = "tampered"
    target.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert read_documents(path=target) == []


def test_write_documents_rejects_transient_parsed_document(tmp_path: Path) -> None:
    target = tmp_path / "docs.jsonl"

    with pytest.raises(TypeError, match="Chunk records only"):
        write_documents([ParsedChunk(text="legacy", metadata={"asset_id": "a"})], path=target)

    assert not target.exists()
