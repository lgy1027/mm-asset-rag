"""Tests for mm_asset_rag.pdf_parser and mm_asset_rag.image_parser."""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

from mm_asset_rag.assets import Asset
from mm_asset_rag.image_parser import parse_image
from mm_asset_rag.pdf_parser import parse_pdf


def _make_asset(tmp_path: Path, asset_id: str, file_path: Path, source_type: str) -> Asset:
    return Asset(
        asset_id=asset_id,
        title=f"Test {asset_id}",
        source_type=source_type,
        relative_path=file_path.name,
        source_url="",
        tags=["test"],
        asset_dir=tmp_path,
    )


def test_parse_pdf_pymupdf(tmp_path: Path, populated_home: Path) -> None:
    src_pdf = populated_home / "assets" / "sample.pdf"
    asset = _make_asset(populated_home / "assets", "pdf_sample", src_pdf, "pdf")

    docs = parse_pdf(asset, parser="pymupdf")
    assert len(docs) >= 1
    assert any("sample" in doc.text.lower() for doc in docs)
    assert all(doc.metadata["parser"] == "pymupdf" for doc in docs)


def test_parse_pdf_invalid_parser_raises(tmp_path: Path) -> None:
    asset = _make_asset(tmp_path, "x", tmp_path / "x.pdf", "pdf")
    with pytest.raises(ValueError, match="Unsupported PDF parser"):
        parse_pdf(asset, parser="bogus")


def test_parse_image_via_caption_only(tmp_path: Path, monkeypatch, populated_home: Path) -> None:
    """OCR off, VLM on; should produce one document that includes title + caption."""
    src_img = populated_home / "assets" / "sample.png"
    asset = _make_asset(populated_home / "assets", "img_sample", src_img, "image")
    monkeypatch.setenv("VLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("VLM_API_KEY", "k")
    monkeypatch.setenv("VLM_MODEL", "vlm")

    with responses.RequestsMock() as rsps:
        rsps.post(
            "https://api.example.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "A simple red square used for testing."}}]},
            status=200,
        )
        docs = parse_image(asset, enable_ocr=False, enable_vlm=True)

    assert len(docs) == 1
    text = docs[0].text
    assert "Test img_sample" in text
    assert "red square" in text


def test_parse_image_without_vlm_or_ocr(tmp_path: Path, populated_home: Path) -> None:
    src_img = populated_home / "assets" / "sample.png"
    asset = _make_asset(populated_home / "assets", "img_sample", src_img, "image")
    docs = parse_image(asset, enable_ocr=False, enable_vlm=False)
    assert len(docs) == 1
    # caption / ocr sections present but empty
    assert "VLM 描述：" in docs[0].text
