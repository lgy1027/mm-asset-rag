"""docling format adapter — multi-format parsing via the ``docling`` library.

docling (IBM) parses PDF / DOCX / PPTX / XLSX / HTML / Markdown / images /
… into a single ``DoclingDocument`` with layout, tables, pictures and
provenance (page + bbox). This adapter is the "format → IR" half: it turns
a ``DoclingDocument`` into the project's format-agnostic ``DocumentIR``.
The shared ``ir_to_documents`` layer then does chunking / image association
/ enrichment / metadata assembly, exactly as for the PyMuPDF and
PaddleOCR-VL paths.

docling is an optional extra (``pip install -e ".[docling]"``) because it
pulls in heavy ML deps (torch / transformers). The import is lazy so the
project imports fine without it; the dispatch surface
(``parse_pdf(parser="docling")``, ``source_type="document"``) is always
wired, and a missing install surfaces as a friendly RuntimeError at parse
time rather than an ImportError at import time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..assets import Asset
from ..paths import get_parsed_dir
from .document_ir import BBox, Block, DocumentIR, ImageRef

# Heading labels that mark a TextItem as a structural heading rather than
# body prose. docling's ``DocItemLabel`` enum uses these names; we match
# by string so a docling-core version that adds new labels still works.
_HEADING_LABELS = {"section_header", "title", "page_header", "title1", "title2", "title3"}


def build_ir_docling(asset: Asset) -> DocumentIR:
    """Parse any docling-supported format → ``DocumentIR``.

    Walks the ``DoclingDocument`` in reading order: text items become
    ``Block``s (section headers / titles become heading blocks), tables
    are exported to markdown and become body blocks, and pictures are
    rendered to ``parsed/<id>/images/`` as ``ImageRef``s with page + bbox
    provenance. Falls back to a whole-document markdown export when the
    structured walk yields nothing (some formats only populate the
    markdown view).

    Each picture's saved path is also appended as a ``![](images/...)``
    line to the last block on its page (or a dedicated block when the
    page has none) — the shared ``ir_to_documents`` layer associates
    chunk ↔ image by re-scanning each sub-chunk's body for these refs
    (``extract_markdown_image_refs``), the same mechanism the
    PaddleOCR-VL path uses. Without the injected ref a picture would be
    saved but never reach any chunk's ``meta["images"]``, so it would be
    invisible to retrieval / answer-time image attachment.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover - exercised via parse_with_docling
        raise RuntimeError(
            'docling parsing requires the [docling] extra: pip install -e ".[docling]"'
        ) from exc

    converter = DocumentConverter()
    conv_res = converter.convert(asset.file_path)
    doc = conv_res.document

    output_dir = get_parsed_dir() / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    markdown_paths: list[str] = []
    # A single per-asset markdown export mirrors what the PyMuPDF / Paddle
    # paths write (page_N.md) and gives the answer layer a source file to
    # cite. docling doesn't paginate the markdown export, so one file.
    md_path = output_dir / "docling_export.md"

    blocks: list[Block] = []
    images: list[ImageRef] = []

    # ── Text items ──────────────────────────────────────────────────────
    for item in doc.texts:
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        page, bbox = _prov(item, doc)
        is_heading = str(getattr(item, "label", "")).lower() in _HEADING_LABELS
        blocks.append(
            Block(
                text=text,
                page=page,
                bbox=bbox,
                heading=text if is_heading else "",
                level=1 if is_heading else 0,
            )
        )

    # ── Tables → markdown body blocks ───────────────────────────────────
    for item in doc.tables:
        md = _safe_table_markdown(item, doc)
        if not md:
            continue
        page, bbox = _prov(item, doc)
        blocks.append(Block(text=md, page=page, bbox=bbox))

    # ── Pictures → saved image files + ImageRefs ────────────────────────
    # The ref line is injected into a block below, after we know which
    # blocks exist on each page (so a picture attaches to the last block
    # of its page rather than always to block[0]).
    page_refs: dict[int | None, list[str]] = {}
    for idx, item in enumerate(doc.pictures):
        page, bbox = _prov(item, doc)
        path = _save_picture(item, doc, images_dir, idx)
        if path is None:
            continue
        images.append(ImageRef(path=path, page=page, bbox=bbox))
        page_refs.setdefault(page, []).append(path)

    # Inject each page's picture refs into that page's last block (or a
    # standalone block when the page has no text/table block yet — e.g. an
    # image-only page in a PPTX). The ``images/``-prefixed path matches
    # ``extract_markdown_image_refs``'s regex, so ir_to_documents finds it
    # and attaches the image to whichever sub-chunk physically contains
    # the ref line (overlap may attach it to two adjacent sub-chunks; the
    # answer layer's per-hit image cap de-dupes).
    if page_refs:
        _inject_picture_refs(blocks, page_refs)

    # ── Fallback: structured walk yielded nothing → whole-doc markdown ─
    if not blocks:
        try:
            whole_md = doc.export_to_markdown().strip()
        except Exception:
            whole_md = ""
        if whole_md:
            blocks.append(Block(text=whole_md, page=None, bbox=None))

    if blocks:
        md_path.write_text("\n\n".join(b.text for b in blocks), encoding="utf-8")
        markdown_paths.append(str(md_path))

    return DocumentIR(
        blocks=blocks,
        images=images,
        asset=asset,
        parser="docling",
        markdown_paths=markdown_paths,
        images_dir=str(images_dir) if images_dir.exists() else "",
    )


def _inject_picture_refs(blocks: list[Block], page_refs: dict[int | None, list[str]]) -> None:
    """Append ``![](images/...)`` ref lines to the last block of each page.

    ``page_refs`` maps a 0-indexed page (or ``None`` for items with no
    provenance, e.g. some HTML/Markdown sources) to the list of saved
    picture paths for that page. A page with no existing block gets a new
    standalone block carrying just the refs (a picture with no caption is
    still worth indexing — it attaches to the page's chunk). Existing
    blocks are mutated in place; new ones are appended (page order is
    preserved because the text/table walks above already appended in
    reading order, and picture-only pages are rare).

    The ref is a markdown image with an empty alt so a renderer shows only
    the picture; the ``images/`` prefix is what ``extract_markdown_image_refs``
    keys off of.
    """
    # Last block index per page (None falls back to the page-less bucket).
    last_block_on_page: dict[int | None, int] = {}
    for i, block in enumerate(blocks):
        last_block_on_page[block.page] = i  # keep the highest index seen

    for page, paths in page_refs.items():
        ref_block = "\n\n" + "\n\n".join(f"![]({p})" for p in paths)
        idx = last_block_on_page.get(page)
        if idx is not None:
            blocks[idx].text += ref_block
        else:
            # No block on this page (image-only page). Add a standalone
            # block so the picture still associates with a chunk.
            blocks.append(Block(text=ref_block.lstrip("\n"), page=page, bbox=None))


def _prov(item, doc) -> tuple[int | None, BBox | None]:
    """Extract ``(page, bbox)`` from a docling DocItem's first provenance.

    docling's ``ProvenanceItem`` carries ``page_no`` (1-indexed) and a
    ``bbox`` (``BoundingRectangle``). Page numbers are converted to 0-indexed
    to match the PyMuPDF / PaddleOCR paths. Returns ``(None, None)`` when
    the item has no provenance (e.g. HTML / markdown sources have none).
    """
    provs = getattr(item, "prov", None) or []
    if not provs:
        return None, None
    prov = provs[0]
    page_no = getattr(prov, "page_no", None)
    page = (page_no - 1) if isinstance(page_no, int) and page_no > 0 else None
    bbox = _bbox_from_rect(getattr(prov, "bbox", None))
    return page, bbox


def _bbox_from_rect(rect) -> BBox | None:
    """docling ``BoundingRectangle`` → ``(x0, y0, x1, y1)`` tuple, or None."""
    if rect is None:
        return None
    # BoundingRectangle uses r_x0/r_y0/r_x1/r_y1 (possibly more corners);
    # the min/max of the corners is a safe axis-aligned box.
    xs = [getattr(rect, name, None) for name in ("r_x0", "r_x1", "r_x2", "r_x3")]
    ys = [getattr(rect, name, None) for name in ("r_y0", "r_y1", "r_y2", "r_y3")]
    xs = [v for v in xs if isinstance(v, (int, float))]
    ys = [v for v in ys if isinstance(v, (int, float))]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _safe_table_markdown(item, doc=None) -> str:
    """Export a docling TableItem to markdown, tolerating API differences.

    docling >=2.112 deprecates ``TableItem.export_to_markdown()`` without a
    ``doc`` argument (the call still works but warns, and a future major
    may drop it). Pass ``doc`` when available; fall back to a no-arg call
    for older docling (<2.112) whose method doesn't accept it, so the
    ``[docling]`` extra keeps working across the pinned ``>=2.0,<3.0``
    range.
    """
    for method_name in ("export_to_markdown", "to_markdown"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                # Try the modern (doc=...) call first; TypeError on an older
                # method that rejects the kwarg falls through to the bare call.
                return str(method(doc=doc)).strip()
            except TypeError:
                try:
                    return str(method()).strip()
                except Exception:
                    continue
            except Exception:
                continue
    return ""


def _save_picture(item, doc, images_dir: Path, idx: int) -> str | None:
    """Render a docling PictureItem to ``images/`` and return its relative path.

    Uses ``DocItem.get_image(doc)`` (inherited by PictureItem). Returns
    ``None`` when the picture has no renderable image (e.g. a placeholder).
    """
    try:
        pil_image = item.get_image(doc)
    except Exception:
        return None
    if pil_image is None:
        return None
    images_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(f"{getattr(item, 'self_ref', '')}-{idx}".encode()).hexdigest()[:12]
    # Prefer PNG for lossless re-encode; the format is irrelevant to the
    # text-to-image / image-to-image routes, which read the file bytes.
    fname = f"docling_{digest}_{idx}.png"
    try:
        pil_image.save(images_dir / fname, format="PNG")
    except Exception:
        return None
    return f"images/{fname}"
