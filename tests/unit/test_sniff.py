"""Tests for ``mm_asset_rag.sniff``.

The sniff module is a pure local file inspector; tests build
representative files in a tmp dir and assert the detected
``source_type`` / metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.sniff import _default_title, sniff

# ─── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def png_path(tmp_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "beach.png"
    Image.new("RGB", (640, 480), color=(200, 220, 240)).save(p)
    return p


@pytest.fixture()
def jpg_path(tmp_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "vacation.jpg"
    Image.new("RGB", (320, 200), color=(255, 200, 50)).save(p, "JPEG")
    return p


@pytest.fixture()
def gif_path(tmp_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "anim.gif"
    Image.new("P", (16, 16), color=42).save(p, "GIF")
    return p


@pytest.fixture()
def bmp_path(tmp_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "tiny.bmp"
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(p, "BMP")
    return p


@pytest.fixture()
def webp_path(tmp_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "modern.webp"
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(p, "WEBP")
    return p


@pytest.fixture()
def pdf_path(tmp_path: Path) -> Path:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    p = tmp_path / "stable_diffusion.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.set_metadata({"title": "Stable Diffusion Beats GANs", "author": "Rombach et al."})
    doc.save(str(p))
    doc.close()
    return p


# ─── _default_title ────────────────────────────────────────────────────


def test_default_title_underscores() -> None:
    assert _default_title("stable_diffusion_paper") == "Stable Diffusion Paper"


def test_default_title_hyphens() -> None:
    assert _default_title("high-res-image") == "High Res Image"


def test_default_title_empty_falls_back() -> None:
    assert _default_title("____") == "____"


# ─── sniff: images ──────────────────────────────────────────────────────


def test_sniff_png(png_path: Path) -> None:
    s = sniff(png_path)
    assert s.source_type == "image"
    assert s.width == 640
    assert s.height == 480
    assert s.file_size > 0
    assert s.title == "Beach"


def test_sniff_jpg(jpg_path: Path) -> None:
    s = sniff(jpg_path)
    assert s.source_type == "image"
    assert s.width == 320
    assert s.height == 200
    assert s.title == "Vacation"


def test_sniff_gif(gif_path: Path) -> None:
    s = sniff(gif_path)
    assert s.source_type == "image"
    assert s.width is not None and s.height is not None


def test_sniff_bmp(bmp_path: Path) -> None:
    s = sniff(bmp_path)
    assert s.source_type == "image"
    assert s.width == 8


def test_sniff_webp(webp_path: Path) -> None:
    s = sniff(webp_path)
    assert s.source_type == "image"


# ─── sniff: PDF ─────────────────────────────────────────────────────────


def test_sniff_pdf(pdf_path: Path) -> None:
    s = sniff(pdf_path)
    assert s.source_type == "pdf"
    assert s.page_count == 3
    assert s.pdf_metadata is not None
    assert s.pdf_metadata.get("title") == "Stable Diffusion Beats GANs"
    assert s.pdf_metadata.get("author") == "Rombach et al."
    # Title from /Info wins over filename-derived title.
    assert s.title == "Stable Diffusion Beats GANs"


def test_sniff_pdf_uses_filename_title_when_no_info(tmp_path: Path) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    p = tmp_path / "my_paper.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(p))
    doc.close()
    s = sniff(p)
    assert s.source_type == "pdf"
    assert s.page_count == 1
    assert s.title == "My Paper"


# ─── sniff: failure modes ──────────────────────────────────────────────


def test_sniff_missing_file(tmp_path: Path) -> None:
    s = sniff(tmp_path / "does_not_exist.pdf")
    assert s.source_type == "unknown"
    assert "not found" in (s.error or "")


def test_sniff_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    s = sniff(p)
    assert s.source_type == "unknown"
    assert s.error is not None


def test_sniff_garbage_bytes(tmp_path: Path) -> None:
    p = tmp_path / "garbage.bin"
    p.write_bytes(b"\x00\x01\x02\x03\x04not a real file")
    s = sniff(p)
    assert s.source_type == "unknown"


def test_sniff_corrupt_pdf_falls_back(tmp_path: Path) -> None:
    """A file with PDF magic but garbage body should still classify as PDF
    but with no metadata."""
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4\n% corrupted content, no xref")
    s = sniff(p)
    assert s.source_type == "pdf"
    assert s.page_count is None
    assert s.pdf_metadata is None


def test_sniff_wrong_extension_but_correct_magic(tmp_path: Path) -> None:
    """Magic bytes win over extension: a JPG with .png suffix is still image."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "lying.png"
    Image.new("RGB", (10, 10), color=(1, 2, 3)).save(p, "JPEG")
    s = sniff(p)
    assert s.source_type == "image"


# ─── sniff: default title fallback when no /Info ───────────────────────


def test_sniff_pdf_no_metadata_uses_stem_title(tmp_path: Path) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    p = tmp_path / "research_notes.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(p))
    doc.close()
    assert sniff(p).title == "Research Notes"


# ─── sniff: document formats (docx/pptx/xlsx/html/md) ─────────────────


def _write_minimal_office(tmp_path: Path, name: str) -> Path:
    """Write a minimal valid Office Open XML (ZIP) container.

    A real docx/pptx/xlsx is a zip with ``[Content_Types].xml``; the
    sniff guard only checks ``zipfile.is_zipfile``, so a zip with that
    member is enough to classify. The extension on the filename drives
    the source_type."""
    import zipfile

    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")
    return p


def test_sniff_docx(tmp_path: Path) -> None:
    p = _write_minimal_office(tmp_path, "report.docx")
    s = sniff(p)
    assert s.source_type == "document"
    assert s.title == "Report"


def test_sniff_pptx(tmp_path: Path) -> None:
    p = _write_minimal_office(tmp_path, "slides.pptx")
    s = sniff(p)
    assert s.source_type == "document"


def test_sniff_xlsx(tmp_path: Path) -> None:
    p = _write_minimal_office(tmp_path, "data.xlsx")
    s = sniff(p)
    assert s.source_type == "document"


def test_sniff_office_zip_without_content_types_still_classifies(tmp_path: Path) -> None:
    """The sniff guard checks ``is_zipfile`` + extension, not the
    ``[Content_Types].xml`` member — so any valid zip with a .docx name
    classifies as ``document``. Content validity is the parser's job (a
    malformed office file fails at parse time, not sniff time). This
    test pins that contract: sniff trusts the name, parser guards content."""
    import zipfile

    p = tmp_path / "not_really.docx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("readme.txt", "not an office file")
    s = sniff(p)
    assert s.source_type == "document"


def test_sniff_non_zip_with_office_extension_rejected(tmp_path: Path) -> None:
    """A non-zip file with a .docx name is rejected as unknown — the
    ``is_zipfile`` guard catches a plain-text file masquerading by name."""
    p = tmp_path / "fake.docx"
    p.write_text("not a zip at all", encoding="utf-8")
    s = sniff(p)
    assert s.source_type == "unknown"


def test_sniff_html(tmp_path: Path) -> None:
    p = tmp_path / "page.html"
    p.write_text("<!DOCTYPE html><html><body>hi</body></html>", encoding="utf-8")
    s = sniff(p)
    assert s.source_type == "document"
    assert s.title == "Page"


def test_sniff_markdown(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text("# Notes\nbody", encoding="utf-8")
    s = sniff(p)
    assert s.source_type == "document"


def test_sniff_text(tmp_path: Path) -> None:
    p = tmp_path / "readme.txt"
    p.write_text("plain text", encoding="utf-8")
    s = sniff(p)
    assert s.source_type == "document"
