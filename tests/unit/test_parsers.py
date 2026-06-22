"""Tests for mm_asset_rag.pdf_parser and mm_asset_rag.image_parser.

Uses the real bundled sample PDFs / images from ``examples/data``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

from mm_asset_rag.assets import Asset
from mm_asset_rag.parsers.image_parser import parse_image
from mm_asset_rag.parsers.pdf_parser import parse_pdf


def _make_asset(assets_dir: Path, asset_id: str, file_path: Path, source_type: str) -> Asset:
    return Asset(
        asset_id=asset_id,
        title=f"Test {asset_id}",
        source_type=source_type,
        relative_path=file_path.relative_to(assets_dir).as_posix(),
        source_url="",
        tags=["test"],
        asset_dir=assets_dir,
    )


def test_parse_pdf_pymupdf(examples_home: Path) -> None:
    """Pick the RAG PDF — known to exist in the bundled sample data."""
    assets_dir = examples_home / "assets"
    src_pdf = assets_dir / "pdfs" / "retrieval-augmented-generation.pdf"
    assert src_pdf.exists(), f"missing test asset: {src_pdf}"
    asset = _make_asset(assets_dir, "pdf_rag", src_pdf, "pdf")

    docs = parse_pdf(asset, parser="pymupdf")
    assert len(docs) >= 1
    assert any("retrieval" in doc.text.lower() for doc in docs)
    assert all(doc.metadata["parser"] == "pymupdf" for doc in docs)
    assert all(doc.metadata["asset_id"] == "pdf_rag" for doc in docs)


def test_parse_pdf_invalid_parser_raises(examples_home: Path) -> None:
    asset = _make_asset(
        examples_home / "assets",
        "pdf_rag",
        examples_home / "assets" / "pdfs" / "retrieval-augmented-generation.pdf",
        "pdf",
    )
    with pytest.raises(ValueError, match="Unsupported PDF parser"):
        parse_pdf(asset, parser="bogus")


def test_parse_image_via_caption_only(examples_home: Path, monkeypatch) -> None:
    """OCR off, VLM on; one document that includes title + caption."""
    assets_dir = examples_home / "assets"
    src_img = assets_dir / "images" / "img_03_opencv-sample-data-happyfish-jpg.jpg"
    assert src_img.exists(), f"missing test asset: {src_img}"
    asset = _make_asset(assets_dir, "img_03_opencv-sample-data-happyfish-jpg", src_img, "image")

    monkeypatch.setenv("VLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("VLM_API_KEY", "k")
    monkeypatch.setenv("VLM_MODEL", "vlm")

    with responses.RequestsMock() as rsps:
        rsps.post(
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "An orange-and-white striped tropical fish."}}
                ]
            },
            status=200,
        )
        docs = parse_image(asset, enable_ocr=False, enable_vlm=True)

    assert len(docs) == 1
    text = docs[0].text
    assert "img_03_opencv-sample-data-happyfish-jpg" in text
    assert "tropical fish" in text


def test_parse_image_without_vlm_or_ocr(examples_home: Path) -> None:
    assets_dir = examples_home / "assets"
    src_img = assets_dir / "images" / "img_03_opencv-sample-data-happyfish-jpg.jpg"
    asset = _make_asset(assets_dir, "img_03_opencv-sample-data-happyfish-jpg", src_img, "image")
    docs = parse_image(asset, enable_ocr=False, enable_vlm=False)
    assert len(docs) == 1
    # caption / ocr sections present but empty
    assert "VLM 描述：" in docs[0].text
    assert "OCR 文本：" in docs[0].text
