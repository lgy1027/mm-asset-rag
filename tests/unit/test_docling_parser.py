"""Tests for the docling format adapter (``mm_asset_rag.parsers.docling_parser``).

docling is an optional extra; these tests stub the ``docling`` import so the
adapter is exercised without the heavy install. The stub injects a fake
``docling.document_converter.DocumentConverter`` whose ``convert`` returns a
``DoclingDocument``-shaped mock (``texts`` / ``tables`` / ``pictures`` lists
with ``label`` / ``text`` / ``prov`` / ``get_image`` attributes mirroring the
real API). This mirrors the ``test_pdf_images`` style of mocking fitz.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mm_asset_rag.assets import Asset
from mm_asset_rag.parsers.docling_parser import build_ir_docling


def _asset(tmp_path: Path) -> Asset:
    # build_ir_docling calls converter.convert(asset.file_path); the path
    # need not exist because the converter is stubbed.
    return Asset(
        asset_id="docling_test",
        title="Docling Test",
        source_type="document",
        relative_path="doc.docx",
        tags=["t"],
        asset_dir=tmp_path,
    )


def _prov(page_no: int, bbox=None):
    """A docling ProvenanceItem-shaped object (page_no 1-indexed + bbox)."""
    return SimpleNamespace(page_no=page_no, bbox=bbox)


def _text_item(text: str, label: str = "paragraph", page_no: int = 1, bbox=None):
    return SimpleNamespace(
        text=text,
        label=label,
        prov=[_prov(page_no, bbox)],
    )


def _install_docling_stub(monkeypatch, *, texts=None, pictures=None, tables=None):
    """Inject a fake ``docling`` package so build_ir_docling's lazy import resolves.

    The stub's ``DocumentConverter().convert(path)`` returns an object whose
    ``.document`` carries the given ``texts`` / ``pictures`` / ``tables``
    lists (mirroring ``DoclingDocument``). ``export_to_markdown`` is set on
    the doc for the fallback path.
    """
    doc = MagicMock()
    if texts is not None:
        doc.texts = texts
    if tables is not None:
        doc.tables = tables
    if pictures is not None:
        doc.pictures = pictures
    doc.export_to_markdown = MagicMock(return_value="# Fallback\nbody text")

    conv_res = SimpleNamespace(document=doc)
    converter = MagicMock()
    converter.convert.return_value = conv_res

    fake_module = SimpleNamespace(
        document_converter=SimpleNamespace(DocumentConverter=lambda: converter)
    )
    # Remove any cached real docling first (it isn't installed in CI, but a
    # developer with the extra would otherwise shadow the stub).
    monkeypatch.setitem(sys.modules, "docling", fake_module)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_module.document_converter)


# ─── build_ir_docling ───────────────────────────────────────────────────────


def test_build_ir_docling_turns_text_items_into_blocks(tmp_path, tmp_home, monkeypatch) -> None:
    """Text items become Blocks; section_header/title labels become headings."""
    texts = [
        _text_item("Introduction", label="section_header", page_no=1),
        _text_item("This is the intro paragraph.", label="paragraph", page_no=1),
        _text_item("Methods", label="title", page_no=2),
    ]
    _install_docling_stub(monkeypatch, texts=texts)

    ir = build_ir_docling(_asset(tmp_home))
    assert ir.parser == "docling"
    headings = [b for b in ir.blocks if b.heading]
    assert [b.heading for b in headings] == ["Introduction", "Methods"]
    # Page numbers are 0-indexed (docling prov is 1-indexed).
    assert ir.blocks[0].page == 0
    assert ir.blocks[2].page == 1
    # A markdown export file is written for the answer layer to cite.
    assert len(ir.markdown_paths) == 1
    assert Path(ir.markdown_paths[0]).exists()


def test_build_ir_docling_skips_empty_text_items(tmp_path, tmp_home, monkeypatch) -> None:
    """A text item with empty/whitespace text is dropped."""
    texts = [
        _text_item("", label="paragraph"),  # dropped
        _text_item("   ", label="paragraph"),  # dropped
        _text_item("real content", label="paragraph"),
    ]
    _install_docling_stub(monkeypatch, texts=texts)

    ir = build_ir_docling(_asset(tmp_home))
    assert len(ir.blocks) == 1
    assert ir.blocks[0].text == "real content"


def test_build_ir_docling_falls_back_to_whole_doc_markdown(tmp_path, tmp_home, monkeypatch) -> None:
    """When the structured walk yields no blocks, the whole-doc markdown
    export becomes one block (some formats only populate the markdown view)."""
    _install_docling_stub(monkeypatch, texts=[], tables=[], pictures=[])
    # The stub's doc.export_to_markdown returns "# Fallback\nbody text".

    ir = build_ir_docling(_asset(tmp_home))
    assert len(ir.blocks) == 1
    assert "Fallback" in ir.blocks[0].text


def test_build_ir_docling_saves_pictures(tmp_home, monkeypatch) -> None:
    """PictureItems are rendered to images/ via get_image(doc)."""

    # A fake PIL image whose save writes a real file so .exists() holds.
    class FakePIL:
        def save(self, path: Path, format: str = "PNG") -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\x89PNG-fake")

    picture = SimpleNamespace(
        prov=[_prov(1, None)],
        get_image=MagicMock(return_value=FakePIL()),
        self_ref="pic0",
    )
    _install_docling_stub(
        monkeypatch,
        texts=[_text_item("body so blocks non-empty", label="paragraph")],
        pictures=[picture],
    )

    asset = Asset(
        asset_id="docling_test",
        title="Docling Test",
        source_type="document",
        relative_path="doc.docx",
        tags=["t"],
        asset_dir=tmp_home,
    )
    ir = build_ir_docling(asset)
    assert len(ir.images) == 1
    assert ir.images[0].path.startswith("images/docling_")
    assert ir.images[0].path.endswith(".png")
    # The image file was actually written under the isolated home.
    img_abs = Path(ir.markdown_paths[0]).parent / ir.images[0].path
    assert img_abs.exists()
    assert img_abs.read_bytes() == b"\x89PNG-fake"
    # The picture's saved path is injected as a ``![](images/...)`` ref
    # into the last block on its page — without it ir_to_documents'
    # extract_markdown_image_refs would never find the image, leaving it
    # invisible to retrieval / answer-time image attachment.
    assert f"![]({ir.images[0].path})" in ir.blocks[-1].text
    # images_dir is now populated so the answer layer can locate the dir.
    assert ir.images_dir.endswith("images")


def test_build_ir_docling_picture_ref_reaches_chunk_meta(tmp_path, tmp_home, monkeypatch) -> None:
    """End-to-end: a picture's ref survives ir_to_documents and lands in the
    chunk's ``meta["images"]`` — the regression this test guards.

    Before the ref-injection fix, docling saved the picture to ``images/``
    but never put a ``![](images/...)`` ref into any block's text, so
    ``extract_markdown_image_refs`` found nothing and the chunk carried no
    images. This test exercises the full IR → chunk path to catch that.
    """

    class FakePIL:
        def save(self, path: Path, format: str = "PNG") -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\x89PNG-fake")

    picture = SimpleNamespace(
        prov=[_prov(1, None)],
        get_image=MagicMock(return_value=FakePIL()),
        self_ref="pic0",
    )
    _install_docling_stub(
        monkeypatch,
        texts=[_text_item("a paragraph of body text long enough to chunk", label="paragraph")],
        pictures=[picture],
    )

    from mm_asset_rag.parsers.document_ir import ir_to_documents

    ir = build_ir_docling(_asset(tmp_home))
    docs = ir_to_documents(ir)
    assert docs, "expected at least one chunk"
    # At least one chunk must carry the picture in its images metadata.
    chunk_image_paths = [img["path"] for d in docs for img in (d.metadata.get("images") or [])]
    assert any(p.startswith("images/docling_") for p in chunk_image_paths), (
        f"picture did not reach any chunk meta.images: {chunk_image_paths}"
    )


def test_build_ir_docling_coalesces_leaf_blocks_into_fewer_chunks(
    tmp_path, tmp_home, monkeypatch
) -> None:
    """docling emits one leaf block per text item; ir_to_documents must
    coalesce same-section leaves into section-sized chunks instead of
    emitting one micro-chunk per leaf (the regression this guards: a real
    docling parse produced 199 chunks, 86% under 100 chars, starving BM25
    + dense retrieval and inflating the point count). Heading blocks keep
    their heading on the resulting chunk's ``section`` metadata."""

    # A heading followed by several short body items on the same page —
    # the pre-coalesce shape that produced hundreds of micro-chunks.
    texts = [
        _text_item("Architecture Overview", label="section_header", page_no=1),
        _text_item("The encoder maps input to latent.", label="paragraph", page_no=1),
        _text_item("The decoder reconstructs from latent.", label="paragraph", page_no=1),
        _text_item("A normalization step sits between.", label="paragraph", page_no=1),
    ]
    _install_docling_stub(monkeypatch, texts=texts)

    from mm_asset_rag.parsers.document_ir import ir_to_documents

    ir = build_ir_docling(_asset(tmp_home))
    docs = ir_to_documents(ir)
    # Coalesced to one section chunk, not four leaf chunks.
    assert len(docs) <= 2, f"expected <=2 coalesced chunks, got {len(docs)}"
    # The heading reaches the chunk's section metadata (not just a label
    # the splitter drops).
    sections = [d.metadata.get("section") for d in docs if d.metadata.get("section")]
    assert any("Architecture Overview" in s for s in sections), (
        f"heading did not reach chunk section metadata: {sections}"
    )


def test_build_ir_docling_coalesced_chunk_still_attaches_picture(
    tmp_path, tmp_home, monkeypatch
) -> None:
    """After coalescing, a picture ref injected into a page's last leaf block
    still survives into the merged section's chunk ``meta["images"]`` —
    coalescing must not break image↔chunk association."""

    class FakePIL:
        def save(self, path: Path, format: str = "PNG") -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\x89PNG-fake")

    picture = SimpleNamespace(
        prov=[_prov(1, None)],
        get_image=MagicMock(return_value=FakePIL()),
        self_ref="pic0",
    )
    # Multiple body items on page 1: the picture ref lands on the page's
    # last block, which coalescing then merges into the section body.
    texts = [
        _text_item(
            "A first paragraph of substantial body content here.", label="paragraph", page_no=1
        ),
        _text_item("A second paragraph continuing the same section.", label="paragraph", page_no=1),
    ]
    _install_docling_stub(monkeypatch, texts=texts, pictures=[picture])

    from mm_asset_rag.parsers.document_ir import ir_to_documents

    ir = build_ir_docling(_asset(tmp_home))
    docs = ir_to_documents(ir)
    chunk_image_paths = [img["path"] for d in docs for img in (d.metadata.get("images") or [])]
    assert any(p.startswith("images/docling_") for p in chunk_image_paths), (
        f"picture did not reach coalesced chunk meta.images: {chunk_image_paths}"
    )


def test_build_ir_docling_picture_on_image_only_page_gets_standalone_block(
    tmp_path, tmp_home, monkeypatch
) -> None:
    """A picture on a page with no text/table block still gets a block
    (image-only PPTX slide) so the picture is not silently dropped."""

    class FakePIL:
        def save(self, path: Path, format: str = "PNG") -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\x89PNG-fake")

    # Text on page 1; picture on page 2 (which has no text block).
    picture = SimpleNamespace(
        prov=[_prov(2, None)],
        get_image=MagicMock(return_value=FakePIL()),
        self_ref="pic0",
    )
    _install_docling_stub(
        monkeypatch,
        texts=[_text_item("body on page 1", label="paragraph", page_no=1)],
        pictures=[picture],
    )

    ir = build_ir_docling(_asset(tmp_home))
    # A standalone block was added for the page-2 picture.
    page2_blocks = [b for b in ir.blocks if b.page == 1]
    assert len(page2_blocks) == 1
    assert f"![]({ir.images[0].path})" in page2_blocks[0].text


def test_build_ir_docling_skips_unrenderable_pictures(tmp_path, tmp_home, monkeypatch) -> None:
    """A picture whose get_image returns None is skipped, not crashed."""
    picture = SimpleNamespace(
        prov=[_prov(1, None)],
        get_image=MagicMock(return_value=None),
        self_ref="pic0",
    )
    _install_docling_stub(
        monkeypatch,
        texts=[_text_item("body", label="paragraph")],
        pictures=[picture],
    )

    ir = build_ir_docling(_asset(tmp_home))
    assert ir.images == []


def test_build_ir_docling_tables_become_markdown_blocks(tmp_path, tmp_home, monkeypatch) -> None:
    """TableItems are exported to markdown and become body blocks."""
    table = SimpleNamespace(
        prov=[_prov(1, None)],
        export_to_markdown=MagicMock(return_value="| a | b |\n|---|---|"),
    )
    _install_docling_stub(
        monkeypatch,
        texts=[_text_item("body", label="paragraph")],
        tables=[table],
    )

    ir = build_ir_docling(_asset(tmp_home))
    table_blocks = [b for b in ir.blocks if b.text.startswith("| a | b |")]
    assert len(table_blocks) == 1


# ─── missing-install error path ────────────────────────────────────────────


def test_parse_with_docling_raises_friendly_when_extra_missing(
    tmp_path, tmp_home, monkeypatch
) -> None:
    """Without the [docling] extra, parse_with_docling raises a RuntimeError
    pointing at the install command — not a raw ImportError."""
    # Ensure docling is genuinely unimportable for this test.
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)

    from mm_asset_rag.parsers.pdf_parser import parse_with_docling

    with pytest.raises(RuntimeError, match="\\[docling\\] extra"):
        parse_with_docling(_asset(tmp_home))
