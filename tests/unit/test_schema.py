"""Tests for mm_asset_rag.schema."""

from __future__ import annotations

import json

from mm_asset_rag.schema import ParsedDocument, SearchHit


def test_parsed_document_to_json_roundtrip() -> None:
    doc = ParsedDocument(text="hello", metadata={"asset_id": "a1", "page": 0})
    payload = doc.to_json()
    assert payload == {"text": "hello", "metadata": {"asset_id": "a1", "page": 0}}
    # serializable
    json.dumps(payload, ensure_ascii=False)


def test_search_hit_key_includes_page() -> None:
    hit = SearchHit(
        route="text",
        score=0.9,
        asset_id="a1",
        title="",
        source_type="pdf",
        source_path="x.pdf",
        metadata={"page": 3},
    )
    assert hit.key() == "a1:3:text"


def test_search_hit_key_without_page() -> None:
    hit = SearchHit(
        route="text",
        score=0.9,
        asset_id="a1",
        title="",
        source_type="image",
        source_path="x.png",
    )
    assert hit.key() == "a1::text"


def test_search_hit_default_metadata_is_per_instance() -> None:
    """Mutable default should not leak between instances."""
    hit1 = SearchHit(route="text", score=0.0, asset_id="", title="", source_type="", source_path="")
    hit1.metadata["routes"] = ["a"]
    hit2 = SearchHit(route="text", score=0.0, asset_id="", title="", source_type="", source_path="")
    assert "routes" not in hit2.metadata
