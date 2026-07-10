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
        md = _safe_table_markdown(item)
        if not md:
            continue
        page, bbox = _prov(item, doc)
        blocks.append(Block(text=md, page=page, bbox=bbox))

    # ── Pictures → saved image files + ImageRefs ────────────────────────
    for idx, item in enumerate(doc.pictures):
        page, bbox = _prov(item, doc)
        path = _save_picture(item, doc, images_dir, idx)
        if path is None:
            continue
        images.append(ImageRef(path=path, page=page, bbox=bbox))

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
    )


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


def _safe_table_markdown(item) -> str:
    """Export a docling TableItem to markdown, tolerating API differences."""
    for method_name in ("export_to_markdown", "to_markdown"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                return str(method()).strip()
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
