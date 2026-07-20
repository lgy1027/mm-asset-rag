"""Tests for the Document IR layer (``mm_asset_rag.parsers.document_ir``).

The IR is a pure intermediate: format adapters produce a ``DocumentIR``,
and ``ir_to_documents`` turns it into the flat ``ParsedDocument`` chunk
list. These tests pin the two contracts the downstream relies on:

1. ``looks_scanned`` — corpus-agnostic char-density threshold that gates
   the ``auto`` parser's OCR fallback.
2. ``ir_to_documents`` — both PDF adapters reduce to the same
   ``metadata`` shape (asset_id / page / chunk_index / section / parser /
   markdown_path / tags), and the section field is normalised (paddle no
   longer omits it).
"""

from __future__ import annotations

from pathlib import Path

from mm_asset_rag.assets import Asset
from mm_asset_rag.parsers.document_ir import (
    Block,
    DocumentIR,
    ImageRef,
    _coalesce_blocks,
    ir_to_documents,
    looks_scanned,
)


def _asset(tmp_path: Path) -> Asset:
    return Asset(
        asset_id="ir_test",
        title="IR Test",
        source_type="pdf",
        relative_path="ir_test.pdf",
        tags=["t"],
        asset_dir=tmp_path,
    )


# ─── looks_scanned ─────────────────────────────────────────────────────────


def _ir(blocks: list[Block], asset: Asset, parser: str = "pymupdf") -> DocumentIR:
    return DocumentIR(blocks=blocks, images=[], asset=asset, parser=parser)


def test_looks_scanned_true_for_zero_text(tmp_path: Path) -> None:
    """A scan yields ~0 text from PyMuPDF → all pages blank → scanned."""
    asset = _asset(tmp_path)
    # PyMuPDF skips empty pages, so a scan produces no blocks at all.
    ir = _ir([], asset)
    assert looks_scanned(ir, text_threshold_per_page=10) is True


def test_looks_scanned_true_for_sparse_text(tmp_path: Path) -> None:
    """50 pages with one 1-char block: total 1 < 10*50 → scanned."""
    asset = _asset(tmp_path)
    ir = _ir([Block(text="x", page=0)] + [Block(text="", page=i) for i in range(1, 50)], asset)
    assert looks_scanned(ir, text_threshold_per_page=10) is True


def test_looks_scanned_false_for_text_pdf(tmp_path: Path) -> None:
    """A real text page (~45 chars) on one page: 45 >= 10*1 → not scanned."""
    asset = _asset(tmp_path)
    ir = _ir([Block(text="Retrieval augmented generation test document", page=0)], asset)
    assert looks_scanned(ir, text_threshold_per_page=10) is False


def test_looks_scanned_threshold_scales_with_pages(tmp_path: Path) -> None:
    """More pages tighten the budget: 100 chars on 1 page is fine, but
    100 chars spread thin across 50 pages is a scan."""
    asset = _asset(tmp_path)
    one_page = _ir([Block(text="x" * 100, page=0)], asset)
    assert looks_scanned(one_page, text_threshold_per_page=10) is False
    many_pages = _ir([Block(text="xx", page=i) for i in range(50)], asset)
    # total 100, budget 10*50=500 → 100 < 500 → scanned
    assert looks_scanned(many_pages, text_threshold_per_page=10) is True


# ─── ir_to_documents: pymupdf path ──────────────────────────────────────────


def test_ir_to_documents_pymupdf_assembles_full_metadata(tmp_path: Path) -> None:
    """A pymupdf IR yields ParsedDocuments with the full pre-IR metadata
    shape — the downstream (contextual / index / answer) is byte-equivalent."""
    asset = _asset(tmp_path)
    body = "正文内容一。" * 40  # long enough to stay one chunk under budget
    ir = DocumentIR(
        blocks=[Block(text=f"# 标题\n{body}", page=0)],
        images=[],
        asset=asset,
        parser="pymupdf",
        markdown_paths=["/tmp/page_0.md"],
    )
    docs = ir_to_documents(ir)
    assert len(docs) >= 1
    meta = docs[0].metadata
    for key in (
        "asset_id",
        "asset_title",
        "source_type",
        "source_path",
        "source_url",
        "page",
        "chunk_index",
        "section",
        "parser",
        "markdown_path",
        "tags",
    ):
        assert key in meta, f"missing metadata key: {key}"
    assert meta["parser"] == "pymupdf"
    assert meta["page"] == 0
    assert meta["markdown_path"] == "/tmp/page_0.md"
    assert meta["tags"] == ["t"]


def test_ir_to_documents_pymupdf_splits_long_body(tmp_path: Path) -> None:
    """A pymupdf body over the token budget is split into multiple chunks,
    all sharing the parent page + markdown_path."""
    asset = _asset(tmp_path)
    body = "句子。" * 1000  # ~4000 chars, well over max_tokens
    ir = DocumentIR(
        blocks=[Block(text=body, page=2)],
        images=[],
        asset=asset,
        parser="pymupdf",
        markdown_paths=["/tmp/page_0.md", "/tmp/page_1.md", "/tmp/page_2.md"],
    )
    docs = ir_to_documents(ir)
    assert len(docs) > 1
    # Every sub-chunk inherits page 2 + its markdown_path.
    assert all(d.metadata["page"] == 2 for d in docs)
    assert all(d.metadata["markdown_path"] == "/tmp/page_2.md" for d in docs)
    # chunk_index is globally unique within the asset (a single monotonic
    # counter across all blocks), so it never collides across pages —
    # important for the Contextual Retrieval cache key ``f"chunk:{ci}"``.
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))


def test_ir_to_documents_chunk_index_unique_across_blocks(tmp_path: Path) -> None:
    """Multiple blocks (e.g. multi-page PDF) get a globally-unique chunk_index.

    Regression guard: previously ``enumerate(sections)`` reset to 0 inside
    each block, so every page's first chunk was tagged ``chunk_index=0``.
    That collided with the Contextual Retrieval cache key and let a
    re-parse overwrite earlier chunks' cached context with later ones'.
    """
    asset = _asset(tmp_path)
    ir = DocumentIR(
        blocks=[
            Block(text="第一页的内容。", page=0),
            Block(text="第二页的内容。", page=1),
            Block(text="第三页的内容。", page=2),
        ],
        images=[],
        asset=asset,
        parser="pymupdf",
        markdown_paths=["/tmp/p0.md", "/tmp/p1.md", "/tmp/p2.md"],
    )
    docs = ir_to_documents(ir)
    assert len(docs) == 3
    indices = [d.metadata["chunk_index"] for d in docs]
    # Unique, monotonic, starting at 0 — NOT [0, 0, 0].
    assert indices == [0, 1, 2]
    assert len(set(indices)) == len(indices)


def test_ir_to_documents_pymupdf_skips_whitespace_body(tmp_path: Path) -> None:
    """A block whose split body is only whitespace produces no chunk —
    matches the pre-IR ``if not section.body.strip(): continue`` guard
    that keeps empty placeholder chunks out of the BM25 channel."""
    asset = _asset(tmp_path)
    ir = DocumentIR(
        blocks=[Block(text="   \n   \n   ", page=0)],
        images=[],
        asset=asset,
        parser="pymupdf",
        markdown_paths=["/tmp/page_0.md"],
    )
    docs = ir_to_documents(ir)
    assert docs == []


# ─── ir_to_documents: paddle path ───────────────────────────────────────────


def test_ir_to_documents_paddle_normalises_section(tmp_path: Path) -> None:
    """The paddle path never set ``section`` pre-IR (asymmetry). The IR
    layer normalises it to an empty string so the field is always present
    and the downstream never has to ``metadata.get("section", "")``."""
    asset = _asset(tmp_path)
    ir = DocumentIR(
        blocks=[Block(text="正文一段足够长的内容用于避免被切分。" * 5, page=0)],
        images=[],
        asset=asset,
        parser="paddleocr-vl-api",
        markdown_paths=["/tmp/page_0.md"],
        images_dir="/tmp/images",
    )
    docs = ir_to_documents(ir)
    assert len(docs) >= 1
    meta = docs[0].metadata
    assert meta["parser"] == "paddleocr-vl-api"
    # section normalised to "" rather than missing.
    assert meta["section"] == ""
    assert "section" in meta
    # paddle carries images_dir; pymupdf does not.
    assert meta["images_dir"] == "/tmp/images"


def test_ir_to_documents_paddle_assigns_images_by_span(tmp_path: Path) -> None:
    """A ``![]()`` ref physically inside a sub-chunk's text is attached to
    that chunk's metadata["images"] — span-based association, not bbox."""
    asset = _asset(tmp_path)
    # markdown with an inline image ref; long enough to be its own chunk.
    text = "正文内容一。" * 30 + "\n\n![图](images/p0_i0.png)\n\n" + "正文内容二。" * 30
    ir = DocumentIR(
        blocks=[Block(text=text, page=0)],
        images=[],
        asset=asset,
        parser="paddleocr-vl-api",
        markdown_paths=["/tmp/page_0.md"],
        images_dir="/tmp/images",
    )
    docs = ir_to_documents(ir)
    # The chunk containing the ref carries the image; others don't.
    chunk_with_image = [d for d in docs if d.metadata.get("images")]
    assert len(chunk_with_image) >= 1
    imgs = chunk_with_image[0].metadata["images"]
    assert imgs[0]["path"] == "images/p0_i0.png"
    assert imgs[0]["caption"] == ""  # paddle path has no caption text


# ─── ir_to_documents: pymupdf image association (bbox) ──────────────────────


def test_ir_to_documents_pymupdf_associates_image_by_figure_ref(tmp_path: Path) -> None:
    """A pymupdf chunk whose body references 图1 attaches the matching
    ImageRef (figure_id resolved via the ref number)."""
    from mm_asset_rag.parsers.document_ir import PageHint

    asset = _asset(tmp_path)
    body = "如图1所示，这是一个重要的示意图。" * 10
    ir = DocumentIR(
        blocks=[Block(text=body, page=0)],
        images=[
            ImageRef(
                path="images/p0_i0.png", page=0, bbox=(0, 0, 10, 10), figure_id=1, caption="示意图"
            ),
        ],
        asset=asset,
        parser="pymupdf",
        markdown_paths=["/tmp/page_0.md"],
        page_hints={0: PageHint()},
    )
    docs = ir_to_documents(ir)
    assert len(docs) >= 1
    chunk_with_image = [d for d in docs if d.metadata.get("images")]
    assert len(chunk_with_image) >= 1
    imgs = chunk_with_image[0].metadata["images"]
    assert imgs[0]["path"] == "images/p0_i0.png"
    assert imgs[0]["figure_id"] == 1
    assert imgs[0]["caption"] == "示意图"


# ─── keyword enrichment: image-ref noise guard ──────────────────────────────
# A chunk whose body is mostly an embedded-image ref must not inject the
# ref's path / hash tokens ("images", "markitdown", "png") as a "关键词:"
# footer — those never match a user query and pollute the BM25 channel.
# Corpus- and parser-agnostic: the guard keys off markdown image syntax,
# not off any particular file or parser.


def test_ir_to_documents_image_ref_only_chunk_has_no_keyword_footer(tmp_path: Path) -> None:
    """A chunk whose body is only ``![](images/x.png)`` keeps the ref (for
    image association) but gets no "关键词:" footer — the stripped source
    has no semantics to extract."""
    asset = _asset(tmp_path)
    ir = DocumentIR(
        blocks=[Block(text="![](images/markitdown_abc123.png)", page=0)],
        images=[],
        asset=asset,
        parser="paddleocr-vl-api",
        markdown_paths=["/tmp/page_0.md"],
        images_dir="/tmp/images",
    )
    docs = ir_to_documents(ir)
    assert len(docs) == 1
    text = docs[0].text
    # The original ref is preserved in the indexed text (association reads body).
    assert "images/markitdown_abc123.png" in text
    # No keyword footer injected — pre-fix this was "关键词: images markitdown png".
    assert "关键词" not in text


def test_ir_to_documents_prose_with_image_ref_keywords_from_prose(tmp_path: Path) -> None:
    """Prose + an image ref: the keyword footer draws on the prose, never
    the image path."""
    asset = _asset(tmp_path)
    body = "联宝科技灯塔工厂智能制造绿色工厂" * 5 + "\n\n![](images/markitdown_abc.png)"
    ir = DocumentIR(
        blocks=[Block(text=body, page=0)],
        images=[],
        asset=asset,
        parser="paddleocr-vl-api",
        markdown_paths=["/tmp/page_0.md"],
        images_dir="/tmp/images",
    )
    docs = ir_to_documents(ir)
    assert len(docs) >= 1
    text = docs[0].text
    assert "关键词" in text  # prose yielded real keywords
    # Path / hash tokens never leak into the footer.
    footer = text.split("关键词:", 1)[-1]
    assert "images" not in footer
    assert "markitdown" not in footer
    assert "png" not in footer


def test_ir_to_documents_plain_text_chunk_still_enriched(tmp_path: Path) -> None:
    """A plain-text chunk (no image ref) is enriched exactly as before —
    the noise guard is a no-op on ref-free bodies."""
    asset = _asset(tmp_path)
    ir = DocumentIR(
        blocks=[Block(text="联宝科技安徽绿色工厂智能制造示范基地" * 5, page=0)],
        images=[],
        asset=asset,
        parser="paddleocr-vl-api",
        markdown_paths=["/tmp/page_0.md"],
    )
    docs = ir_to_documents(ir)
    assert len(docs) >= 1
    assert "关键词" in docs[0].text


# ─── _coalesce_blocks (docling / paddle leaf-block merging) ────────────────


def test_coalesce_blocks_merges_same_section_leaves() -> None:
    """A heading followed by body leaves on the same page collapses into one
    section block — the leaf-per-item shape docling emits would otherwise
    become one micro-chunk per leaf."""
    blocks = [
        Block(text="Intro", page=0, heading="Intro", level=1),
        Block(text="first body line", page=0),
        Block(text="second body line", page=0),
    ]
    out = _coalesce_blocks(blocks, target_chars=2000)
    assert len(out) == 1
    assert out[0].heading == "Intro"
    assert "first body line" in out[0].text
    assert "second body line" in out[0].text


def test_coalesce_blocks_starts_new_section_on_heading() -> None:
    """A second heading opens a new section block rather than merging into
    the previous section."""
    blocks = [
        Block(text="A", page=0, heading="A", level=1),
        Block(text="body-a", page=0),
        Block(text="B", page=0, heading="B", level=1),
        Block(text="body-b", page=0),
    ]
    out = _coalesce_blocks(blocks, target_chars=2000)
    assert len(out) == 2
    assert [b.heading for b in out] == ["A", "B"]
    assert "body-a" in out[0].text and "body-b" not in out[0].text
    assert "body-b" in out[1].text


def test_coalesce_blocks_splits_on_page_change() -> None:
    """A page change closes the current section so a heading's section does
    not bleed across pages."""
    blocks = [
        Block(text="H", page=0, heading="H", level=1),
        Block(text="page0 body", page=0),
        Block(text="page1 body", page=1),
    ]
    out = _coalesce_blocks(blocks, target_chars=2000)
    assert len(out) == 2
    assert "page0 body" in out[0].text
    assert "page1 body" in out[1].text
    assert out[1].page == 1


def test_coalesce_blocks_caps_headingless_run() -> None:
    """A long run of body blocks with no heading is split at target_chars so
    the downstream splitter gets splitter-friendly chunks, not one giant."""
    body = "x" * 300
    blocks = [Block(text=body, page=0) for _ in range(10)]
    out = _coalesce_blocks(blocks, target_chars=1000)
    # target 1000 / 300-per-block → ~3-4 blocks per section.
    assert len(out) >= 3
    assert all(len(b.text) <= 1000 + 300 for b in out)
