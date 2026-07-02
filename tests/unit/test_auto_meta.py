"""Tests for ``mm_asset_rag.auto_meta``.

The VLM endpoint is mocked via ``monkeypatch`` — we never hit a real
model in unit tests. Each test installs a fake ``_vlm_chat_json``
implementation that returns a canned payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mm_asset_rag import auto_meta
from mm_asset_rag.auto_meta import (
    AutoMeta,
    _clean_optional_str,
    _clean_str_list,
    _parse_json_response,
    auto_meta_image,
    auto_meta_pdf_first_page,
)

# ─── helpers ───────────────────────────────────────────────────────────


def _stub_chat_json(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | Exception):
    """Replace ``_vlm_chat_json`` with a function that returns ``payload``.

    Pass an ``Exception`` to simulate a network/parsing failure.
    """
    if isinstance(payload, Exception):

        def boom(*_a, **_kw):
            raise payload

        monkeypatch.setattr(auto_meta, "_vlm_chat_json", boom)
    else:

        def ok(*_a, **_kw):
            return payload

        monkeypatch.setattr(auto_meta, "_vlm_chat_json", ok)


def _stub_creds(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    monkeypatch.setattr(
        auto_meta,
        "_vlm_creds",
        lambda: ("http://stub/v1", "k", "m") if present else None,
    )


def _png(tmp_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(p)
    return p


def _pdf(tmp_path: Path) -> Path:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    p = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(p))
    doc.close()
    return p


# ─── JSON parsing ──────────────────────────────────────────────────────


def test_parse_json_strict_object() -> None:
    assert _parse_json_response('{"title": "x", "tags": ["a"]}') == {
        "title": "x",
        "tags": ["a"],
    }


def test_parse_json_strips_markdown_fence() -> None:
    wrapped = '```json\n{"title": "y"}\n```'
    assert _parse_json_response(wrapped) == {"title": "y"}


def test_parse_json_extracts_embedded_block() -> None:
    wrapped = 'Sure! Here you go:\n{"title": "z"}\nHope that helps.'
    assert _parse_json_response(wrapped) == {"title": "z"}


def test_parse_json_unparseable_raises() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_json_response("no JSON at all")


# ─── Field cleaning ────────────────────────────────────────────────────


def test_clean_str_list_dedupes_and_caps() -> None:
    raw = ["a", "b", "a", "c", "", "  ", "d"]
    assert _clean_str_list(raw, limit=3) == ["a", "b", "c"]


def test_clean_str_list_from_string() -> None:
    assert _clean_str_list("only") == ["only"]


def test_clean_str_list_from_none() -> None:
    assert _clean_str_list(None) == []


def test_clean_optional_str_trims() -> None:
    assert _clean_optional_str("  hi  ") == "hi"
    assert _clean_optional_str("") is None
    assert _clean_optional_str(None) is None
    assert _clean_optional_str(42) == "42"


# ─── auto_meta_image ───────────────────────────────────────────────────


def test_auto_meta_image_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_creds(monkeypatch, present=True)
    _stub_chat_json(
        monkeypatch,
        {
            "title": "Sunset Beach",
            "description": "A calm beach at sunset.",
            "tags": ["beach", "sunset", "ocean", "sky", "sand"],
            "dominant_objects": ["ocean", "sky"],
        },
    )
    monkeypatch.setattr(auto_meta, "auto_meta_enabled", True, raising=False)
    monkeypatch.setattr(
        auto_meta,
        "_vlm_chat_json",
        lambda *a, **kw: {
            "title": "Sunset Beach",
            "description": "A calm beach at sunset.",
            "tags": ["beach", "sunset", "ocean", "sky", "sand"],
            "dominant_objects": ["ocean", "sky"],
        },
        raising=True,
    )

    p = _png(tmp_path)
    result = auto_meta_image(p)
    assert result is not None
    assert result.title == "Sunset Beach"
    assert "calm beach" in (result.description or "")
    assert result.tags == ["beach", "sunset", "ocean", "sky", "sand"]
    assert result.dominant_objects == ["ocean", "sky"]


def test_auto_meta_image_no_creds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_creds(monkeypatch, present=False)
    assert auto_meta_image(_png(tmp_path)) is None


def test_auto_meta_image_disabled_in_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_creds(monkeypatch, present=True)
    monkeypatch.setattr(auto_meta, "auto_meta_enabled", False, raising=False)
    assert auto_meta_image(_png(tmp_path)) is None


def test_auto_meta_image_request_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_creds(monkeypatch, present=True)
    _stub_chat_json(monkeypatch, RuntimeError("network down"))
    assert auto_meta_image(_png(tmp_path)) is None


def test_auto_meta_image_bad_json_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_creds(monkeypatch, present=True)
    monkeypatch.setattr(auto_meta, "_vlm_chat_json", lambda *a, **kw: {"title": None})
    result = auto_meta_image(_png(tmp_path))
    assert result is not None
    assert result.title is None  # missing field → None
    assert result.tags == []


# ─── auto_meta_pdf_first_page ─────────────────────────────────────────


def test_auto_meta_pdf_first_page_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_creds(monkeypatch, present=True)
    monkeypatch.setattr(
        auto_meta,
        "_vlm_chat_json",
        lambda *a, **kw: {
            "title": "Attention Is All You Need",
            "description": "Introduces the Transformer architecture.",
            "tags": ["transformer", "attention", "nlp"],
            "page_summary": "Abstract + intro.",
        },
    )
    p = _pdf(tmp_path)
    result = auto_meta_pdf_first_page(p)
    assert isinstance(result, AutoMeta)
    assert result.title == "Attention Is All You Need"
    assert result.page_summary == "Abstract + intro."
    # The render-temp PNG should be cleaned up.
    assert not (p.with_suffix(".preview.png")).exists()


def test_auto_meta_pdf_first_page_no_creds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_creds(monkeypatch, present=False)
    assert auto_meta_pdf_first_page(_pdf(tmp_path)) is None


def test_auto_meta_pdf_first_page_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_creds(monkeypatch, present=True)
    _stub_chat_json(monkeypatch, RuntimeError("VLM offline"))
    assert auto_meta_pdf_first_page(_pdf(tmp_path)) is None
