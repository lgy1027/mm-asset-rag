"""Tests for ``mm_asset_rag.parsers.pdf_images``.

The image-extraction + association logic is the tier-1 multimodal layer:
PDF embedded figures are pulled out and attached to the text chunks that
reference (or sit next to) them. These tests cover the pure helpers
(``scan_figure_refs``, ``detect_figure_captions``, ``associate_images``,
``extract_markdown_image_refs``) and the I/O-bound ``extract_page_images``
with a fake PyMuPDF page/doc so no real PDF or fitz install is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mm_asset_rag.parsers.pdf_images import (
    LineItem,
    PageImage,
    associate_images,
    detect_figure_captions,
    extract_markdown_image_refs,
    extract_page_images,
    scan_figure_refs,
)

# ─── scan_figure_refs ────────────────────────────────────────────────────


def test_scan_figure_refs_chinese_forms() -> None:
    assert scan_figure_refs("如图3所示") == {3}
    assert scan_figure_refs("见图 12") == {12}
    assert scan_figure_refs("参见图7中的流程") == {7}
    assert scan_figure_refs("图3-4 描述了") == {3}
    assert scan_figure_refs("图3、图5 都在") == {3, 5}


def test_scan_figure_refs_english_forms() -> None:
    assert scan_figure_refs("see Figure 5 for details") == {5}
    assert scan_figure_refs("as shown in Fig. 2") == {2}


def test_scan_figure_refs_no_ref() -> None:
    assert scan_figure_refs("该图展示了双碳目标") == set()
    assert scan_figure_refs("如左图所示") == set()
    assert scan_figure_refs("") == set()


# ─── detect_figure_captions ───────────────────────────────────────────────


def test_detect_figure_captions_nearest_image() -> None:
    # Two images on the page; a caption line below image #2.
    img1 = PageImage(path="images/p0_i0.png", page=0, bbox=(0, 0, 200, 100), index=0)
    img2 = PageImage(path="images/p0_i1.png", page=0, bbox=(0, 300, 200, 400), index=1)
    lines = [
        LineItem(text="正文段落…", bbox=(0, 120, 200, 140)),
        LineItem(text="图 2: 双碳目标路线图", bbox=(0, 410, 200, 425)),
    ]
    registry = detect_figure_captions(lines, [img1, img2])
    assert 2 in registry
    assert registry[2].image_path == "images/p0_i1.png"
    assert "双碳目标" in registry[2].caption


def test_detect_figure_captions_ignores_body_sentence() -> None:
    # "如图3所示" mid-paragraph must NOT be treated as a caption — only a
    # line *opening* with the figure label is a caption.
    img = PageImage(path="images/p0_i0.png", page=0, bbox=(0, 0, 200, 100), index=0)
    lines = [LineItem(text="如图3所示，本节讨论碳达峰路径。", bbox=(0, 120, 300, 135))]
    assert detect_figure_captions(lines, [img]) == {}


def test_detect_figure_captions_no_images() -> None:
    lines = [LineItem(text="图 1: foo", bbox=(0, 0, 100, 20))]
    assert detect_figure_captions(lines, []) == {}


# ─── associate_images ──────────────────────────────────────────────────────


def test_associate_precise_ref_match() -> None:
    fig3 = PageImage(path="images/p0_i0.png", page=0, bbox=(0, 0, 200, 100), index=0)
    page_figures = {
        3: type(
            "F",
            (),
            {"number": 3, "caption": "图3: foo", "image_path": "images/p0_i0.png", "page": 0},
        )()
    }
    out = associate_images(
        "如图3所示的双碳目标",
        chunk_bbox=(0, 110, 200, 130),
        page_images=[fig3],
        page_figures=page_figures,
    )
    assert len(out) == 1
    assert out[0]["path"] == "images/p0_i0.png"
    assert out[0]["figure_id"] == 3


def test_associate_spatial_fallback_attaches_nearby_image() -> None:
    # Chunk never names a figure, but an image sits just below it.
    img = PageImage(path="images/p0_i0.png", page=0, bbox=(0, 150, 200, 250), index=0)
    out = associate_images(
        "本节阐述了战略方向。", chunk_bbox=(0, 40, 200, 140), page_images=[img], page_figures={}
    )
    assert len(out) == 1
    assert out[0]["path"] == "images/p0_i0.png"
    assert out[0]["figure_id"] is None


def test_associate_spatial_fallback_skips_distant_image() -> None:
    img = PageImage(path="images/p0_i0.png", page=0, bbox=(0, 800, 200, 900), index=0)
    out = associate_images(
        "本节阐述了战略方向。", chunk_bbox=(0, 40, 200, 140), page_images=[img], page_figures={}
    )
    assert out == []


def test_associate_spatial_fallback_horizontal_disjoint_skipped() -> None:
    # Image in a different column (no horizontal overlap) is not attached.
    img = PageImage(path="images/p0_i0.png", page=0, bbox=(400, 100, 600, 200), index=0)
    out = associate_images(
        "本节阐述了战略方向。", chunk_bbox=(0, 100, 200, 200), page_images=[img], page_figures={}
    )
    assert out == []


def test_associate_dedupes_precise_and_spatial() -> None:
    img = PageImage(path="images/p0_i0.png", page=0, bbox=(0, 150, 200, 250), index=0)
    page_figures = {
        3: type(
            "F", (), {"number": 3, "caption": "图3", "image_path": "images/p0_i0.png", "page": 0}
        )()
    }
    out = associate_images(
        "如图3所示", chunk_bbox=(0, 40, 200, 140), page_images=[img], page_figures=page_figures
    )
    # Same image attached once, with the precise figure_id retained.
    assert len(out) == 1
    assert out[0]["figure_id"] == 3


# ─── extract_markdown_image_refs ─────────────────────────────────────────


def test_extract_markdown_image_refs() -> None:
    md = "段落A\n\n![](images/img_abc.png)\n\n图1: 双碳\n\n![](images/img_def.jpg)"
    refs = extract_markdown_image_refs(md)
    assert len(refs) == 2
    assert refs[0][2] == "images/img_abc.png"
    assert refs[1][2] == "images/img_def.jpg"
    # Spans point at the ref position.
    assert md[refs[0][0] : refs[0][1]].startswith("![](")


def test_extract_markdown_image_refs_empty() -> None:
    assert extract_markdown_image_refs("") == []
    assert extract_markdown_image_refs("无图的段落") == []


# ─── extract_page_images (I/O, mocked fitz) ────────────────────────────────


def test_extract_page_images_writes_files_and_filters_small(tmp_path) -> None:
    page = MagicMock()
    # Two images: one big enough, one tiny (icon).
    page.get_images.return_value = [
        (11, 0, 200, 200, 8, "DeviceRGB"),  # xref=11, 200x200
        (12, 0, 40, 40, 8, "DeviceRGB"),  # xref=12, 40x40 → filtered
    ]
    page.get_image_rects.side_effect = lambda xref: [
        MagicMock(x0=0, y0=0, x1=200, y1=200)
        if xref == 11
        else MagicMock(x0=0, y0=300, x1=40, y1=340)
    ]

    doc = MagicMock()
    doc.extract_image.side_effect = lambda xref: {
        11: {"ext": "png", "image": b"\x89PNG\r\n"},
        12: {"ext": "png", "image": b"\x89PNG\r\n"},
    }[xref]

    images_dir = tmp_path / "img"
    out = extract_page_images(page, doc, page_index=0, images_dir=images_dir, min_dim=80)
    assert len(out) == 1
    assert out[0].path == "images/p0_i0.png"
    assert (images_dir / "p0_i0.png").read_bytes() == b"\x89PNG\r\n"
    assert out[0].bbox == (0, 0, 200, 200)


def test_extract_page_images_skips_unextractable(tmp_path) -> None:
    page = MagicMock()
    page.get_images.return_value = [(11, 0, 200, 200, 8, "DeviceRGB")]
    page.get_image_rects.return_value = []

    doc = MagicMock()
    doc.extract_image.side_effect = RuntimeError("corrupt stream")

    out = extract_page_images(page, doc, 0, tmp_path / "img", min_dim=80)
    assert out == []
