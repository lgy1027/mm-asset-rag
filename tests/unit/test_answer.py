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


def test_warn_insecure_base_url_warns_on_non_loopback_http(caplog) -> None:
    """A plain-HTTP base_url to a non-loopback host warns once."""
    import logging

    from mm_asset_rag.answer import _warn_insecure_base_url, _warned_insecure_base_urls

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.answer"):
        _warn_insecure_base_url("http://10.0.0.5/v1")
    assert any("10.0.0.5" in r.message for r in caplog.records)
    assert any("HTTP" in r.message for r in caplog.records)


def test_warn_insecure_base_url_silent_on_loopback(caplog) -> None:
    """http:// to loopback hosts (local ollama) must not warn."""
    import logging

    from mm_asset_rag.answer import _warn_insecure_base_url, _warned_insecure_base_urls

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.answer"):
        _warn_insecure_base_url("http://127.0.0.1:11434/v1")
        _warn_insecure_base_url("http://localhost:11434/v1")
        _warn_insecure_base_url("http://[::1]:11434/v1")
    assert not any("Authorization" in r.message for r in caplog.records)


def test_warn_insecure_base_url_dedups(caplog) -> None:
    """Same non-loopback http:// base_url warns only once per process."""
    import logging

    from mm_asset_rag.answer import _warn_insecure_base_url, _warned_insecure_base_urls

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.answer"):
        _warn_insecure_base_url("http://10.0.0.5/v1")
        _warn_insecure_base_url("http://10.0.0.5/v1")
    matching = [r for r in caplog.records if "10.0.0.5" in r.message]
    assert len(matching) == 1


def test_warn_insecure_base_url_silent_on_https(caplog) -> None:
    """HTTPS base_urls never warn, regardless of host."""
    import logging

    from mm_asset_rag.answer import _warn_insecure_base_url, _warned_insecure_base_urls

    _warned_insecure_base_urls.clear()
    with caplog.at_level(logging.WARNING, logger="mm_asset_rag.answer"):
        _warn_insecure_base_url("https://10.0.0.5/v1")
    assert not any("HTTP" in r.message for r in caplog.records)
