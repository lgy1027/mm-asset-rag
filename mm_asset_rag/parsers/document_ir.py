"""Document IR — structured intermediate representation between a format
adapter and the flat ``ParsedDocument`` chunk list.

Why this exists
---------------
``ParsedDocument`` is just ``(text, metadata: dict)`` — a *chunk*, not a
*document*. Every format adapter therefore re-implements "extract text +
images + image↔chunk association + heading detection + assembly" from
scratch. The IR captures the document structure once, in typed fields, so:

* format adapters only produce a ``DocumentIR`` (the "format → IR" step);
* the shared processing layer turns one ``DocumentIR`` into the chunk list
  (the "IR → ParsedDocument" step), reusing the existing splitter / image
  association / keyword enrichment unchanged.

``ParsedDocument`` and ``documents.jsonl`` are untouched — the IR is a pure
intermediate layer, and the chunk output is byte-equivalent to the
pre-IR two PDF paths (verified by the regression diff in the plan).

The ``Block`` fields mirror ``chunk_splitter.Section`` (heading + body +
bbox) so the splitter can be reused without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..assets import Asset


# BBox in page space as (x0, y0, x1, y1). None when the adapter has no
# geometry (e.g. the PaddleOCR-VL markdown path, or docling without prov).
BBox = tuple[float, float, float, float]


@dataclass
class Block:
    """A leaf text unit in reading order — a paragraph or a heading line.

    ``heading`` empty means an ordinary paragraph; non-empty means this
    block is itself a heading (its ``text`` is the heading text). ``level``
    is the heading depth (0 for non-headings) — currently informational;
    the splitter keys off ``heading`` being non-empty, matching
    ``Section.heading`` semantics.
    """

    text: str
    page: int | None = None
    bbox: BBox | None = None
    heading: str = ""
    level: int = 0


@dataclass
class ImageRef:
    """An image extracted from the source, plus its provenance.

    ``path`` is relative to ``parsed/<asset_id>/`` (e.g. ``"images/p3_i0.png"``)
    so it matches the ``meta["images"][*]["path"]`` shape the answer layer
    already reads. ``bbox`` is the image's page-space bbox when the adapter
    has geometry (PyMuPDF); None otherwise (PaddleOCR span-based, docling
    without prov). ``figure_id``/``caption`` carry the detected figure
    number / caption text when available (PyMuPDF caption detection).
    """

    path: str
    page: int | None = None
    bbox: BBox | None = None
    figure_id: int | None = None
    caption: str = ""


@dataclass
class DocumentIR:
    """Structured parse of a single asset, format-agnostic.

    Produced by a format adapter (``build_ir_*``) and consumed by the
    shared ``ir_to_documents`` processing layer. Carries the originating
    ``asset`` so the processing layer can fill ``metadata`` (asset_id /
    title / source_path / tags / …) without re-deriving it.

    ``page_hints`` is an optional escape hatch for adapters that produce
    per-page geometry the splitter needs but that doesn't fit the
    ``Block`` shape — currently only PyMuPDF, which feeds per-line font
    sizes + bboxes into ``split_by_heading``'s heading heuristic. Keyed
    by page index. Adapters without such geometry (PaddleOCR-VL,
    docling) leave it empty and the splitter runs on block text alone.
    """

    blocks: list[Block]
    images: list[ImageRef]
    asset: Asset
    parser: str  # "pymupdf" | "paddleocr-vl-api" | "docling" | …
    markdown_paths: list[str] = field(default_factory=list)
    images_dir: str = ""
    page_hints: dict[int, PageHint] = field(default_factory=dict)

    def total_text_chars(self) -> int:
        """Sum of non-empty block text lengths — used by the scanned-PDF
        heuristic. Empty blocks (blank lines, headings with no body) don't
        count toward "did this page actually yield text"."""
        return sum(len(b.text) for b in self.blocks if b.text)


@dataclass
class PageHint:
    """Per-page splitter inputs an adapter may supply beyond ``Block``.

    PyMuPDF populates ``font_sizes`` / ``line_bboxes`` (parallel to the
    page's text lines) so ``split_by_heading``'s font-size and
    standalone-short-line heading heuristics keep working. Other adapters
    leave these empty.
    """

    font_sizes: list[float] = field(default_factory=list)
    line_bboxes: list = field(default_factory=list)


def looks_scanned(
    ir: DocumentIR,
    *,
    text_threshold_per_page: int,
) -> bool:
    """Heuristic: is this PDF likely a scan (image-only, near-zero text)?

    Corpus-agnostic: purely a character-density threshold. A document is
    treated as scanned when its total non-empty text is below
    ``text_threshold_per_page * page_count`` — i.e. the whole document
    has fewer chars than the threshold allows for one page. This catches
    true scans (PyMuPDF yields ~0 text from image-only pages, so even a
    50-page scan stays under the budget) while leaving short-but-real
    text PDFs (a one-page memo with 45 chars) above it.

    Pages with no blocks still count toward ``page_count`` so a scan's
    many blank pages tighten the budget rather than being ignored.
    """

    pages: set[int] = set()
    total = 0
    for block in ir.blocks:
        if block.page is not None:
            pages.add(block.page)
        if block.text:
            total += len(block.text)
    page_count = len(pages) if pages else 1
    return total < text_threshold_per_page * page_count


def ir_to_documents(ir: DocumentIR) -> list:
    """Turn one ``DocumentIR`` into the flat ``ParsedDocument`` chunk list.

    The shared "IR → ParsedDocument" half. Encapsulates the four steps
    that used to be duplicated in each format adapter: token-budget
    chunking, image↔chunk association, keyword enrichment, and metadata
    assembly. The output ``metadata`` shape is identical to the pre-IR
    PDF paths so the downstream (contextual, index, retrieval, answer)
    is byte-equivalent.

    Image association is the one step that genuinely differs by source:
    PyMuPDF carries per-image bboxes and associates by spatial proximity
    to the chunk's section bbox (``associate_images``); PaddleOCR-VL has
    no geometry and associates by re-scanning each sub-chunk's text for
    ``![]()`` refs (``extract_markdown_image_refs``). We branch on
    ``ir.parser`` — both reduce to the same ``meta["images"]`` shape.
    """
    # Local imports to keep document_ir.py free of parser-internal deps
    # at import time (chunk_splitter pulls in transformers lazily).
    from ..schema import ParsedDocument
    from ..settings import get_settings
    from .chunk_splitter import _make_token_counter, recursive_split, split_with_recursion
    from .pdf_images import (
        PageImage,
        associate_images,
        extract_markdown_image_refs,
        scan_figure_refs,
    )

    settings = get_settings()
    _tok, count_tokens = _make_token_counter(settings.chunk_tokenizer)
    extract_images = settings.pdf_extract_images
    asset = ir.asset
    is_pymupdf = ir.parser == "pymupdf"

    # Index markdown_paths by page so each chunk can point at its page's
    # .md file (matching the pre-IR metadata["markdown_path"] field). Both
    # adapters append one path per page in ascending page order, so the
    # page number is the list index.
    markdown_path_by_page: dict[int, str] = {}
    for block in ir.blocks:
        page = block.page
        if page is not None and 0 <= page < len(ir.markdown_paths):
            markdown_path_by_page[page] = ir.markdown_paths[page]

    docs: list[ParsedDocument] = []
    # Global per-asset chunk counter: ``enumerate(sections)`` inside the
    # per-block loop would reset to 0 for every block (PyMuPDF emits one
    # block per page), so a multi-page asset would have N chunks all tagged
    # ``chunk_index=0``. That collides with the Contextual Retrieval cache
    # key ``f"chunk:{ci}"`` — re-parse would let later chunks overwrite
    # earlier ones' cached context. A single monotonic counter across all
    # blocks keeps ``chunk_index`` unique within the asset.
    global_chunk_index = 0
    for block in ir.blocks:
        page = block.page
        markdown_path = markdown_path_by_page.get(page, "") if page is not None else ""
        hint = ir.page_hints.get(page) if page is not None else None

        # Chunking: PyMuPDF feeds per-line font/bbox hints into the
        # heading-aware two-layer splitter; PaddleOCR-VL has no geometry
        # and runs the pure recursive token-budget splitter on its
        # markdown (its ATX headings are handled by the separator
        # hierarchy, not a heading-detection pass). Each sub-chunk keeps
        # the parent Block's heading + bbox (split_with_recursion already
        # propagates Section.bbox from line_bboxes).
        if is_pymupdf and hint is not None:
            sections = split_with_recursion(
                block.text,
                font_sizes=hint.font_sizes,
                line_bboxes=hint.line_bboxes,
                target_tokens=settings.chunk_target_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                count_tokens=count_tokens,
            )
        else:
            # PaddleOCR / docling: pure recursive split. Wrap each piece
            # in a lightweight Section-like (heading + body + bbox) so the
            # assembly loop below is uniform. recursive_split returns
            # strings; the paddle path has no section heading to carry.
            from .chunk_splitter import Section

            pieces = recursive_split(
                block.text,
                target_tokens=settings.chunk_target_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                count_tokens=count_tokens,
            )
            sections = [Section(heading="", body=p, bbox=block.bbox) for p in pieces]

        # Group this page's images for association. PyMuPDF collects them
        # here (bbox-based); PaddleOCR resolves refs per-sub-chunk below
        # (span-based) and ignores this group.
        page_images: list[PageImage] = []
        page_figures: dict = {}
        if is_pymupdf and extract_images:
            for ref in ir.images:
                if ref.page != page:
                    continue
                page_images.append(PageImage(path=ref.path, page=ref.page, bbox=ref.bbox, index=0))
                if ref.figure_id is not None:
                    from .pdf_images import Figure

                    page_figures[ref.figure_id] = Figure(
                        number=ref.figure_id,
                        caption=ref.caption,
                        image_path=ref.path,
                        page=ref.page or 0,
                    )

        for _chunk_index, section in enumerate(sections):
            body = section.body.strip()
            # Skip sections with no body — empty chunks (e.g. a bare "1"
            # heading with no following text) would pollute the BM25
            # channel with a placeholder payload that drags down dense
            # ranking.
            if not body:
                continue
            # Use the global counter so chunk_index is unique across blocks;
            # the local index from enumerate only orders sections within this
            # block and is otherwise unused.
            ci = global_chunk_index
            global_chunk_index += 1
            enriched_text = _maybe_enrich_with_keywords(body)

            chunk_images: list = []
            if is_pymupdf and extract_images and page_images:
                chunk_images = associate_images(body, section.bbox, page_images, page_figures)
            elif not is_pymupdf:
                # PaddleOCR / docling: re-scan the sub-chunk's own text for
                # ``![]()`` refs — a ref belongs to whichever sub-chunk it
                # physically appears in. Overlap may attach a ref to two
                # adjacent sub-chunks; the answer layer's per-hit image
                # cap de-dupes.
                sub_refs = extract_markdown_image_refs(body)
                if sub_refs:
                    ref_numbers = sorted(scan_figure_refs(body))
                    for i, (_, _, ref_path) in enumerate(sub_refs):
                        fig_id = ref_numbers[i] if i < len(ref_numbers) else None
                        chunk_images.append(
                            {
                                "path": ref_path,
                                "figure_id": fig_id,
                                "caption": "",
                                "page": page,
                            }
                        )

            meta: dict = {
                "asset_id": asset.asset_id,
                "asset_title": asset.title,
                "source_type": asset.source_type,
                "source_path": asset.relative_path,
                "source_url": asset.source_url,
                "page": page,
                "chunk_index": ci,
                "section": section.heading,
                "parser": ir.parser,
                "markdown_path": markdown_path,
                "tags": asset.tags,
            }
            if ir.images_dir:
                meta["images_dir"] = ir.images_dir
            if chunk_images:
                meta["images"] = chunk_images
            docs.append(ParsedDocument(text=enriched_text, metadata=meta))
    return docs


def _maybe_enrich_with_keywords(text: str) -> str:
    """Append a "关键词: ..." footer to a chunk when Settings enables it.

    Forwarded from pdf_parser to keep a single enrichment implementation;
    defined here (not imported) only to avoid a circular import with
    pdf_parser, which imports this module.

    Embedded-image refs (``![](images/x.png)``) are stripped from the
    keyword-extraction source so a chunk whose body is mostly a figure ref
    does not inject path/hash tokens ("images", "markitdown", "png") as
    keywords — those never match a user query and pollute the BM25 channel.
    The chunk's own ``text`` keeps the ref so image↔chunk association (which
    reads the body) is unaffected. Corpus- and parser-agnostic.
    """
    from ..settings import get_settings
    from ..text_keywords import _strip_markdown_images, enrich_chunk_text, extract_keywords

    s = get_settings()
    if not s.enrich_chunk_with_keywords:
        return text
    keyword_source = _strip_markdown_images(text)
    kws = extract_keywords(
        keyword_source, top_k=s.enrich_chunk_keyword_top_k, language=s.enrich_chunk_language
    )
    return enrich_chunk_text(text, kws)
