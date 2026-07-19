"""Tests for mm_asset_rag PDF and image parsers.

No bundled sample corpus is required: tests create tiny PDF/image assets
inside ``tmp_home`` and feed them through the production parser functions.
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


@pytest.fixture()
def pdf_asset(tmp_home: Path) -> Asset:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    assets_dir = tmp_home / "assets"
    pdf_dir = assets_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    p = pdf_dir / "rag.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Retrieval augmented generation test document")
    doc.save(str(p))
    doc.close()
    return _make_asset(assets_dir, "pdf_rag", p, "pdf")


@pytest.fixture()
def image_asset(tmp_home: Path) -> Asset:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    assets_dir = tmp_home / "assets"
    img_dir = assets_dir / "images"
    img_dir.mkdir(parents=True)
    p = img_dir / "fish.jpg"
    Image.new("RGB", (32, 32), color=(255, 128, 0)).save(p, "JPEG")
    return _make_asset(assets_dir, "img_fish", p, "image")


def test_parse_pdf_pymupdf(pdf_asset: Asset) -> None:
    docs = parse_pdf(pdf_asset, parser="pymupdf")
    assert len(docs) >= 1
    assert any("retrieval" in doc.text.lower() for doc in docs)
    assert all(doc.metadata["parser"] == "pymupdf" for doc in docs)
    assert all(doc.metadata["asset_id"] == "pdf_rag" for doc in docs)


def test_parse_pdf_invalid_parser_raises(pdf_asset: Asset) -> None:
    with pytest.raises(ValueError, match="Unsupported PDF parser"):
        parse_pdf(pdf_asset, parser="bogus")


def test_parse_image_via_caption_only(image_asset: Asset, monkeypatch) -> None:
    """OCR off, VLM on; one document that includes title + caption."""
    monkeypatch.setenv("VLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("VLM_API_KEY", "k")
    monkeypatch.setenv("VLM_MODEL", "vlm")

    with responses.RequestsMock() as rsps:
        rsps.post(
            "https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "An orange-and-white striped tropical fish."}}]
            },
            status=200,
        )
        docs = parse_image(image_asset, enable_ocr=False, enable_vlm=True)

    assert len(docs) == 1
    text = docs[0].text
    assert "img_fish" in text
    assert "tropical fish" in text


def test_parse_image_without_vlm_or_ocr(image_asset: Asset) -> None:
    # Title + tags are still set on the fixture asset, so the parser has a
    # signal to emit — the new contract is "skip when *all* of title /
    # tags / VLM caption / OCR text are empty".
    docs = parse_image(image_asset, enable_ocr=False, enable_vlm=False)
    assert len(docs) == 1
    assert "VLM 描述：" in docs[0].text
    assert "OCR 文本：" in docs[0].text


def test_parse_image_skips_when_no_signal(tmp_home: Path) -> None:
    """Placeholder text must not be written into the text collection.

    When title, tags, VLM caption, and OCR text are all empty, the
    parser returns ``[]`` so BM25 does not see "Picsum 1015" /
    "图片标题" as a frequent document token. This is the v2 eval
    BUG 1 fix.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    assets_dir = tmp_home / "assets"
    img_dir = assets_dir / "images"
    img_dir.mkdir(parents=True)
    p = img_dir / "picsum.jpg"
    Image.new("RGB", (32, 32), color=(255, 128, 0)).save(p, "JPEG")
    asset = Asset(
        asset_id="picsum_no_signal",
        title="",
        source_type="image",
        relative_path="images/picsum.jpg",
        source_url="",
        tags=[],
        asset_dir=assets_dir,
    )
    docs = parse_image(asset, enable_ocr=False, enable_vlm=False)
    assert docs == []


def test_parse_image_emits_chunk_when_only_title_present(tmp_home: Path) -> None:
    """If even a single signal field is non-empty, emit one chunk."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    assets_dir = tmp_home / "assets"
    img_dir = assets_dir / "images"
    img_dir.mkdir(parents=True)
    p = img_dir / "titled.jpg"
    Image.new("RGB", (32, 32), color=(0, 128, 255)).save(p, "JPEG")
    asset = Asset(
        asset_id="img_titled",
        title="Linux logo",
        source_type="image",
        relative_path="images/titled.jpg",
        source_url="",
        tags=[],
        asset_dir=assets_dir,
    )
    docs = parse_image(asset, enable_ocr=False, enable_vlm=False)
    assert len(docs) == 1
    assert "Linux logo" in docs[0].text


def test_parse_pdf_auto_falls_back_to_paddle_on_scan(
    pdf_asset: Asset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auto now runs PyMuPDF first and falls back to PaddleOCR-VL only when
    the result looks scanned (near-zero text). The fallback needs the
    PADDLEOCR_VL_API_TOKEN — same token dependency as before, but gated on
    text density rather than always-on."""
    from mm_asset_rag.parsers import pdf_parser
    from mm_asset_rag.parsers.document_ir import DocumentIR
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "token")
    get_settings.cache_clear()
    called = {}

    def fake_paddle(asset: Asset):
        called["parser"] = "paddle"
        return []

    monkeypatch.setattr(pdf_parser, "parse_with_paddleocr_vl", fake_paddle)

    # Force PyMuPDF to look scanned: build_ir_pymupdf returns an IR whose
    # blocks carry almost no text, so looks_scanned() is True → fallback.
    def fake_build_ir_pymupdf(asset: Asset):
        from mm_asset_rag.parsers.document_ir import Block

        return DocumentIR(
            blocks=[Block(text="x", page=0)],  # 1 char << 50/page threshold
            images=[],
            asset=asset,
            parser="pymupdf",
        )

    monkeypatch.setattr(pdf_parser, "build_ir_pymupdf", fake_build_ir_pymupdf)
    parse_pdf(pdf_asset, parser="auto")
    assert called == {"parser": "paddle"}


def test_parse_pdf_auto_stays_on_pymupdf_for_text_pdf(
    pdf_asset: Asset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text-rich PDF stays on PyMuPDF even when a PaddleOCR token is
    configured — the old auto behaviour OCR'd every PDF unnecessarily."""
    from mm_asset_rag.parsers import pdf_parser
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "token")
    get_settings.cache_clear()
    called = {"paddle": 0}

    def fake_paddle(asset: Asset):
        called["paddle"] += 1
        return []

    monkeypatch.setattr(pdf_parser, "parse_with_paddleocr_vl", fake_paddle)
    # Real pdf_asset has real text → not scanned → no fallback.
    parse_pdf(pdf_asset, parser="auto")
    assert called == {"paddle": 0}


def test_submit_paddleocr_vl_job_uses_settings(
    pdf_asset: Asset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mm_asset_rag.parsers import pdf_parser
    from mm_asset_rag.parsers.pdf_parser import submit_paddleocr_vl_job
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "token")
    monkeypatch.setenv("PADDLEOCR_VL_JOB_URL", "https://ocr.example/jobs")
    monkeypatch.setenv("PADDLEOCR_VL_MODEL", "custom-model")
    monkeypatch.setenv("PADDLEOCR_VL_TIMEOUT", "12")
    get_settings.cache_clear()
    captured = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {"data": {"jobId": "job-1"}}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(pdf_parser.requests, "post", fake_post)
    job_id = submit_paddleocr_vl_job(pdf_asset.file_path)

    assert job_id == "job-1"
    assert captured["url"] == "https://ocr.example/jobs"
    assert captured["headers"]["Authorization"] == "bearer token"
    assert captured["data"]["model"] == "custom-model"
    assert captured["timeout"] == 12.0


def test_ocr_image_url_allowed_blocks_ssrf(monkeypatch) -> None:
    """The OCR image-download allow-list refuses private/loopback/metadata
    hosts and any host not on the configured PaddleOCR endpoint domain.

    A crafted PDF can make the OCR service echo back an internal URL; the
    fetcher must refuse it instead of becoming an SSRF proxy.
    """
    from mm_asset_rag.parsers import pdf_parser
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("PADDLEOCR_VL_JOB_URL", "https://ocr.example.com/api/v2/ocr/jobs")
    monkeypatch.delenv("PADDLEOCR_VL_IMAGE_HOSTS", raising=False)
    get_settings.cache_clear()
    # Same-domain CDN image: allowed.
    assert pdf_parser._ocr_image_url_allowed("https://ocr.example.com/imgs/a.png") is True
    # Different domain: refused.
    assert pdf_parser._ocr_image_url_allowed("https://evil.example.com/x.png") is False
    # AWS metadata endpoint: refused.
    assert pdf_parser._ocr_image_url_allowed("http://169.254.169.254/latest/meta-data/") is False
    # RFC1918 / loopback: refused even though scheme is http.
    assert pdf_parser._ocr_image_url_allowed("http://10.0.0.5/internal") is False
    assert pdf_parser._ocr_image_url_allowed("http://127.0.0.1:8080/x") is False
    # Non-http(s) / unparseable: refused.
    assert pdf_parser._ocr_image_url_allowed("ftp://x/y") is False
    assert pdf_parser._ocr_image_url_allowed("not a url") is False


def test_ocr_image_url_allowed_extra_hosts(monkeypatch) -> None:
    """PADDLEOCR_VL_IMAGE_HOSTS lets a deployer allow an extra CDN host."""
    from mm_asset_rag.parsers import pdf_parser
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("PADDLEOCR_VL_JOB_URL", "https://ocr.example.com/jobs")
    monkeypatch.setenv("PADDLEOCR_VL_IMAGE_HOSTS", "cdn.example.com, cdn2.example.com")
    get_settings.cache_clear()
    assert pdf_parser._ocr_image_url_allowed("https://cdn.example.com/a.png") is True
    assert pdf_parser._ocr_image_url_allowed("https://cdn2.example.com/a.png") is True
    assert pdf_parser._ocr_image_url_allowed("https://other.example.com/a.png") is False
