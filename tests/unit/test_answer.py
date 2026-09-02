"""Tests for mm_asset_rag.answer (focused on the offline fallback path)."""

from __future__ import annotations

from unittest.mock import Mock

from mm_asset_rag.answer import answer_question, fallback_answer, format_sources
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.search_service import SearchCommand, SearchMode


def _hit(asset_id: str, evidence: str = "some text", *, score: float = 0.9) -> SearchHit:
    return SearchHit(
        route="text",
        score=score,
        asset_id=asset_id,
        title=asset_id,
        source_type="pdf",
        source_path=f"{asset_id}.pdf",
        evidence=evidence,
        metadata={
            "document_id": f"document-{asset_id}",
            "version_id": f"document-{asset_id}@1-deadbeefcafe",
            "chunk_id": f"document-{asset_id}@1-deadbeefcafe:0",
            "page": 2,
            "parser": "pymupdf",
        },
    )


def test_format_sources_exposes_document_version_chunk_not_asset_identity() -> None:
    sources = format_sources([_hit("a"), _hit("b")])
    assert len(sources) == 2
    assert sources[0]["document_id"] == "document-a"
    assert sources[0]["version_id"] == "document-a@1-deadbeefcafe"
    assert sources[0]["chunk_id"] == "document-a@1-deadbeefcafe:0"
    assert "asset_id" not in sources[0]
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


def test_answer_question_uses_search_service_when_hits_are_missing() -> None:
    backend = Mock()
    backend.execute.return_value = [_hit("a")]

    answer_question(
        "question",
        search_service=backend,
        collection="team",
        principal="alice",
        min_confidence=0.5,
    )

    backend.execute.assert_called_once_with(
        SearchCommand(
            query="question",
            mode=SearchMode.HYBRID,
            top_k=5,
            collection="team",
            principal="alice",
        )
    )


def test_answer_question_refuses_low_confidence_without_calling_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "mm_asset_rag.answer.llm_answer", Mock(side_effect=AssertionError("LLM called"))
    )
    result = answer_question("question", hits=[_hit("a", score=0.2)], min_confidence=0.5)
    assert result["sources"] == []
    assert "证据不足" in result["answer"]


def test_answer_json_returns_valid_json(monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    from mm_asset_rag.answer import answer_json

    monkeypatch.setattr(
        "mm_asset_rag.answer.get_search_service",
        lambda: SimpleNamespace(
            execute=lambda command: [
                SearchHit(
                    route="text",
                    score=0.9,
                    asset_id="a",
                    title="a",
                    source_type="pdf",
                    source_path="a.pdf",
                    evidence="evidence",
                )
            ]
        ),
    )
    payload = answer_json("any question?", collection="team", principal="alice", min_confidence=0.5)
    parsed = json.loads(payload)
    assert "answer" in parsed
    assert "sources" in parsed


def test_warn_insecure_base_url_warns_on_non_loopback_http(caplog) -> None:
    """A plain-HTTP base_url to a non-loopback host warns once."""
    import logging

    from mm_asset_rag.provider_security import (
        _warned_insecure_base_urls,
        warn_insecure_base_url,
    )

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.provider_security"):
        warn_insecure_base_url("http://10.0.0.5/v1")
    assert any("10.0.0.5" in r.message for r in caplog.records)
    assert any("HTTP" in r.message for r in caplog.records)


def test_warn_insecure_base_url_silent_on_loopback(caplog) -> None:
    """http:// to loopback hosts (local ollama) must not warn."""
    import logging

    from mm_asset_rag.provider_security import (
        _warned_insecure_base_urls,
        warn_insecure_base_url,
    )

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.provider_security"):
        warn_insecure_base_url("http://127.0.0.1:11434/v1")
        warn_insecure_base_url("http://localhost:11434/v1")
        warn_insecure_base_url("http://[::1]:11434/v1")
    assert not any("Authorization" in r.message for r in caplog.records)


def test_warn_insecure_base_url_dedups(caplog) -> None:
    """Same non-loopback http:// base_url warns only once per process."""
    import logging

    from mm_asset_rag.provider_security import (
        _warned_insecure_base_urls,
        warn_insecure_base_url,
    )

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.provider_security"):
        warn_insecure_base_url("http://10.0.0.5/v1")
        warn_insecure_base_url("http://10.0.0.5/v1")
    matching = [r for r in caplog.records if "10.0.0.5" in r.message]
    assert len(matching) == 1


def test_warn_insecure_base_url_silent_on_https(caplog) -> None:
    """HTTPS base_urls never warn, regardless of host."""
    import logging

    from mm_asset_rag.provider_security import (
        _warned_insecure_base_urls,
        warn_insecure_base_url,
    )

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.provider_security"):
        warn_insecure_base_url("https://10.0.0.5/v1")
    assert not any("HTTP" in r.message for r in caplog.records)
