"""Extract embedded images from PDF pages and associate them with text chunks.

Tier-1 of multimodal PDF handling: PyMuPDF parses text only, dropping
every embedded figure. This module recovers those images so a text hit
that references "如图3所示" can carry the actual figure path alongside
its evidence — the figure is *not* embedded into the vector index (that
is tier 2), it lives in ``ParsedDocument.metadata["images"]`` and is
surfaced to the LLM / web UI as an attachment of the text hit.

The association chain is:

1. ``extract_page_images`` pulls every image a page references, writes it
   to ``parsed/<id>/images/p{N}_i{I}.{ext}``, and records its bbox.
2. ``detect_figure_captions`` scans the page's text lines for caption
   patterns ("图 3: ...", "Figure 5 ...") and assigns each caption to the
   nearest image on the page (by bbox distance) → a ``{fig_num: Figure}``
   registry.
3. ``scan_figure_refs`` finds the figure numbers a chunk's body refers to
   ("如图3所示", "见图12", "Figure 5").
4. ``associate_images`` joins them: precise match on referenced figure
   numbers first, then a spatial fallback that attaches any un-referenced
   image whose bbox is adjacent to the chunk's own bbox (covers chunks
   that sit next to a figure without naming it).

bbox is carried as a plain ``(x0, y0, x1, y1)`` tuple so this module does
not import ``fitz`` — the PyMuPDF caller converts ``fitz.Rect`` to a tuple
at the boundary, and unit tests can build fakes without the dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# bbox = (x0, y0, x1, y1) in PDF page coordinates (points, origin top-left).
BBox = tuple[float, float, float, float]

# ─── Figure-number reference patterns ──────────────────────────────────────
# Order matters: the "如图/见图" forms pin the *referenced* figure number;
# the bare "图 N" form is also captured but is the noisiest (matches
# captions too — handled by de-duping against the page's caption registry
# in ``associate_images``).
_REF_PATTERNS = [
    re.compile(r"如\s*图\s*(\d+)"),
    re.compile(r"见\s*图\s*(\d+)"),
    re.compile(r"参\s*见\s*图\s*(\d+)"),
    re.compile(r"图\s*(\d+)\s*所示"),
    re.compile(r"图\s*(\d+)\s*[-.．、]"),
    re.compile(r"图\s*(\d+)"),
    re.compile(r"Figure\s*(\d+)", re.IGNORECASE),
    re.compile(r"Fig\.?\s*(\d+)", re.IGNORECASE),
]

# Caption = a *line* (not a sentence fragment) opening with a figure label.
# Anchored at line start so "如图3所示" inside a body sentence does not get
# mistaken for a caption — the caption is the line that *introduces* a figure.
_CAPTION_RE = re.compile(r"^\s*(?:图|Figure|Fig\.?)\s*(\d+)\s*[:：.、\s]", re.IGNORECASE)

# Markdown inline image reference: ![alt](images/xxx.png) — PaddleOCR-VL output.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")

# Vertical gap (in PDF points) below/above a chunk within which an un-named
# image counts as "spatially adjacent" for the fallback. PDF text is ~12pt,
# so 120pt ≈ 10 lines of breathing room — loose enough to catch a figure
# placed right under a paragraph, tight enough to skip a figure one section
# away.
_SPATIAL_GAP = 120.0


@dataclass
class PageImage:
    """An image extracted from one PDF page, saved to disk."""

    path: str  # relative to parsed/<id>/, e.g. "images/p3_i0.png"
    page: int
    bbox: BBox | None
    index: int  # order within the page, for stable naming


@dataclass
class Figure:
    """A figure number resolved to its image + caption."""

    number: int
    caption: str
    image_path: str
    page: int


@dataclass
class LineItem:
    """A single text line with its page bbox + font size.

    Built by the PyMuPDF caller from ``page.get_text("dict")`` blocks; passed
    to :func:`detect_figure_captions` so this module stays fitz-free.
    """

    text: str
    bbox: BBox | None
    size: float = 0.0


# ─── Pure helpers (no I/O) ─────────────────────────────────────────────────


def scan_figure_refs(text: str) -> set[int]:
    """Return the set of figure numbers referenced in ``text``.

    "如图3所示" → {3}; "见图 12" → {12}; "Figure 5" → {5}; "该图" → set().
    Multiple references dedupe into one number.
    """
    if not text:
        return set()
    found: set[int] = set()
    for pat in _REF_PATTERNS:
        for m in pat.finditer(text):
            try:
                found.add(int(m.group(1)))
            except (ValueError, IndexError):
                continue
    return found


def _line_height(bbox: BBox | None) -> float:
    if bbox is None:
        return 0.0
    return abs(bbox[3] - bbox[1])


def _v_distance(a: BBox | None, b: BBox | None) -> float:
    """Vertical gap between two bboxes (0.0 if they overlap vertically)."""
    if a is None or b is None:
        return float("inf")
    # a above b: a.y1 <= b.y0
    if a[3] <= b[1]:
        return b[1] - a[3]
    if b[3] <= a[1]:
        return a[1] - b[3]
    return 0.0  # vertical overlap


def _h_overlap(a: BBox | None, b: BBox | None) -> bool:
    """True if a and b share any horizontal span (columns overlap)."""
    if a is None or b is None:
        return True  # unknown → don't filter on it
    return not (a[2] <= b[0] or b[2] <= a[0])


def detect_figure_captions(
    line_items: list[LineItem], page_images: list[PageImage]
) -> dict[int, Figure]:
    """Map each caption line to its nearest image → ``{fig_num: Figure}``.

    A caption is a line whose stripped text matches :data:`_CAPTION_RE`
    (anchored at line start so body sentences like "如图3所示" don't pose
    as captions). Each caption is paired with the page image whose bbox is
    vertically closest (and horizontally overlapping); that image is then
    labelled with the caption's figure number.
    """
    registry: dict[int, Figure] = {}
    if not page_images:
        return registry
    for item in line_items:
        stripped = (item.text or "").strip()
        if not stripped:
            continue
        m = _CAPTION_RE.match(stripped)
        if not m:
            continue
        try:
            fig_num = int(m.group(1))
        except ValueError:
            continue
        # Closest image: minimal vertical gap, requiring horizontal overlap.
        best_img: PageImage | None = None
        best_gap = float("inf")
        for img in page_images:
            if not _h_overlap(img.bbox, item.bbox):
                continue
            gap = _v_distance(img.bbox, item.bbox)
            if gap < best_gap:
                best_gap = gap
                best_img = img
        if best_img is None:
            # No horizontally-aligned image — skip rather than mis-attribute.
            continue
        registry[fig_num] = Figure(
            number=fig_num,
            caption=stripped,
            image_path=best_img.path,
            page=best_img.page,
        )
    return registry


def associate_images(
    chunk_text: str,
    chunk_bbox: BBox | None,
    page_images: list[PageImage],
    page_figures: dict[int, Figure],
) -> list[dict[str, Any]]:
    """Return the image refs to attach to a chunk.

    Two passes, precise-first:

    1. **Precise**: figure numbers referenced in ``chunk_text`` are looked
       up in ``page_figures``; each hit yields the figure's image.
    2. **Spatial fallback**: any page image *not* already attached is added
       if its bbox sits within :data:`_SPATIAL_GAP` above or below the
       chunk and shares a horizontal span. Catches a chunk that sits next
       to a figure it never names.

    Returns a list of ``{"path", "figure_id", "caption", "page"}`` dicts,
    de-duplicated by path. Empty list when nothing matches — the caller
    then leaves ``metadata["images"]`` unset, keeping the hit clean.
    """
    attached: dict[str, dict[str, Any]] = {}

    # 1. Precise figure-number matching.
    refs = scan_figure_refs(chunk_text)
    for num in sorted(refs):
        fig = page_figures.get(num)
        if fig is None:
            continue
        attached[fig.image_path] = {
            "path": fig.image_path,
            "figure_id": fig.number,
            "caption": fig.caption,
            "page": fig.page,
        }

    # 2. Spatial fallback for un-referenced images.
    for img in page_images:
        if img.path in attached:
            continue
        if chunk_bbox is None or img.bbox is None:
            continue
        if not _h_overlap(img.bbox, chunk_bbox):
            continue
        if _v_distance(img.bbox, chunk_bbox) <= _SPATIAL_GAP:
            attached[img.path] = {
                "path": img.path,
                "figure_id": None,
                "caption": "",
                "page": img.page,
            }

    return list(attached.values())


def extract_markdown_image_refs(markdown: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, path)`` spans for ``![](images/xxx.png)`` refs.

    Used by the PaddleOCR-VL path: markdown preserves reading order, so a
    chunk's images are the refs whose span falls inside the chunk's text
    slice. Path is returned as-is ("images/xxx.png", relative to parsed/).
    """
    if not markdown:
        return []
    return [(m.start(), m.end(), m.group(1)) for m in _MD_IMAGE_RE.finditer(markdown)]


# ─── I/O: image extraction ─────────────────────────────────────────────────


def extract_page_images(
    page: Any,
    doc: Any,
    page_index: int,
    images_dir: Path,
    *,
    min_dim: int = 80,
) -> list[PageImage]:
    """Pull every image referenced on ``page`` to ``images_dir``.

    Uses PyMuPDF's ``page.get_images(full=True)`` for the xref list and
    ``doc.extract_image`` for the bytes. Filters out tiny logos / icons
    (either dimension below ``min_dim``). Skips images that fail to extract
    (corrupt stream, unsupported codec) — a single bad image must not abort
    the page. Returns images in xref-list order with stable
    ``p{N}_i{I}.{ext}`` names so a re-parse produces the same paths.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    out: list[PageImage] = []
    raw_images = []
    try:
        raw_images = list(page.get_images(full=True))
    except Exception:
        return out

    for idx, img_info in enumerate(raw_images):
        # get_images(full=True) tuple: (xref, smask, w, h, bpc, colorspace, ...)
        xref = img_info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        bbox: BBox | None = None
        if rects:
            r = rects[0]
            # fitz.Rect → tuple; clamp to positive width/height
            bbox = (r.x0, r.y0, r.x1, r.y1)

        # Width/height from the image info tuple — cheaper than decoding.
        try:
            w = int(img_info[2])
            h = int(img_info[3])
        except (IndexError, ValueError, TypeError):
            w = h = 0
        if w and h and (w < min_dim or h < min_dim):
            continue

        try:
            extracted = doc.extract_image(xref)
        except Exception:
            continue
        ext = (extracted.get("ext") or "png").lower().replace(".", "")
        data = extracted.get("image")
        if not data:
            continue
        fname = f"p{page_index}_i{idx}.{ext}"
        (images_dir / fname).write_bytes(data)
        out.append(PageImage(path=f"images/{fname}", page=page_index, bbox=bbox, index=idx))
    return out
