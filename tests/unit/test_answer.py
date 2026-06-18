"""Tests for mm_asset_rag.answer (focused on the offline fallback path)."""

from __future__ import annotations

from mm_asset_rag.answer import fallback_answer, format_sources
from mm_asset_rag.schema import SearchHit


def _hit(asset_id: str, evidence: str = "some text") -> SearchHit:
    return SearchHit(
        route="text",
        score=0.9,
        asset_id=asset_id,
        title=asset_id,
        source_type="pdf",
        source_path=f"{asset_id}.pdf",
        evidence=evidence,
        metadata={"page": 2, "parser": "pymupdf"},
    )


def test_format_sources_extracts_metadata() -> None:
    sources = format_sources([_hit("a"), _hit("b")])
    assert len(sources) == 2
    assert sources[0]["asset_id"] == "a"
    assert sources[0]["page"] == 2
    assert sources[0]["parser"] == "pymupdf"
    assert sources[0]["score"] == 0.9


def test_fallback_answer_mentions_unconfigured_llm() -> None:
    result = fallback_answer("q?", [_hit("a", evidence="evidence-A")])
    assert result["question"] == "q?"
    assert "未配置 LLM" in result["answer"]
    assert "evidence-A" in result["answer"]
    assert len(result["sources"]) == 1


def test_fallback_answer_skips_empty_evidence() -> None:
    result = fallback_answer("q?", [_hit("a", evidence=""), _hit("b", evidence="beta")])
    assert "beta" in result["answer"]
    assert result["answer"].count("\n\n") >= 1


def test_answer_json_returns_valid_json(monkeypatch) -> None:
    import json

    from mm_asset_rag.answer import answer_json

    monkeypatch.setattr(
        "mm_asset_rag.answer.hybrid_search",
        lambda query, top_k=5, image_path=None: [
            SearchHit(
                route="text",
                score=0.9,
                asset_id="a",
                title="a",
                source_type="pdf",
                source_path="a.pdf",
                evidence="evidence",
            )
        ],
    )
    payload = answer_json("any question?")
    parsed = json.loads(payload)
    assert "answer" in parsed
    assert "sources" in parsed
