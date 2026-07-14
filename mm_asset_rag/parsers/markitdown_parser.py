"""MarkItDown format adapter — multi-format parsing via the ``markitdown`` library.

MarkItDown (Microsoft) is a lightweight pure-Python file→Markdown converter
covering docx / pptx / xlsx / html / markdown / … without the heavy ML
stack docling pulls in (torch / transformers). For Office/text documents
(``source_type="document"``) it is the default backend — installed in
core deps so upload works out of the box. docling remains available as an
optional heavy backend (``--document-parser docling``).

This adapter is the "format → IR" half: it converts MarkItDown's markdown
output into the project's format-agnostic ``DocumentIR``. The shared
``ir_to_documents`` layer then does chunking / image association /
enrichment / metadata assembly, exactly as for the PyMuPDF /
PaddleOCR-VL / docling paths.

The one MarkItDown-specific wrinkle is images: docx/pptx embed pictures as
``![](data:image/png;base64,...)`` **data URLs**, but the image-association
machinery (``extract_markdown_image_refs`` → ``pdf_images._MD_IMAGE_RE``)
only matches ``images/...`` relative paths (the shape the answer layer
reads). So this adapter decodes every data URL to a file under
``parsed/<id>/images/`` and rewrites the ref to ``images/...`` before
emitting blocks — same on-disk layout docling / PaddleOCR produce.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from ..assets import Asset
from ..paths import get_parsed_dir
from .document_ir import Block, DocumentIR, ImageRef

# Markdown inline image reference whose target is a base64 data URL.
# Group 1 = the full data URL, group 2 = mime subtype (png/jpg/…), group 3
# = the base64 payload. ``[^)]+`` is deliberately greedy-within-parens: the
# payload has no ``)`` so it stops at the closing paren of the ``![](...)``
# even though base64 may contain ``/`` ``+`` ``=``.
_DATA_URL_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((data:image/([a-zA-Z0-9.+-]+);base64,([^)]+))\)")

# Best-effort mime-subtype → filename suffix. ``svg+xml`` collapses to svg;
# anything unknown falls back to png (a safe default the answer layer's
# image readers and CLIP both tolerate via Pillow).
_SUFFIX_BY_MIME: dict[str, str] = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "gif": "gif",
    "webp": "webp",
    "bmp": "bmp",
    "svg+xml": "svg",
}

# ATX heading line: ``#``, ``##``, … up to 6. Captures the level and the
# heading text (stripped). MarkItDown emits ATX headings for docx/pptx/xlsx
# titles; we promote them to ``Block(heading=...)`` so downstream has the
# section structure (informational today; the recursive splitter runs on
# block text and keys off ``\n\n`` boundaries — but carrying headings keeps
# parity with the docling adapter).
_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def build_ir_markitdown(asset: Asset) -> DocumentIR:
    """Parse any MarkItDown-supported format → ``DocumentIR``.

    MarkItDown converts the source to a single markdown string
    (``MarkItDown().convert(path, keep_data_uris=True).text_content``).
    ``keep_data_uris=True`` is essential: MarkItDown's markdownify layer
    *strips* data URLs to ``data:image/png;base64...`` by default (see
    ``_markdownify.convert_img``), which would discard the base64 payload
    this adapter needs to decode. This function:

    * decodes every ``![](data:image/...;base64,...)`` ref to a file under
      ``parsed/<id>/images/`` and rewrites the ref to ``images/...`` (so
      ``extract_markdown_image_refs`` finds it and the answer layer can
      attach the image);
    * splits the rewritten markdown into ``Block``s on ``\\n\\n``
      boundaries, promoting ATX heading lines to heading blocks;
    * writes a single ``markitdown_export.md`` (the rewritten markdown) so
      the answer layer has a source file to cite — mirroring the
      ``docling_export.md`` / ``page_N.md`` the other adapters write.

    MarkItDown has no page geometry, so every ``Block.page`` / ``bbox`` is
    ``None`` (same as the PaddleOCR-VL markdown path's image-only span
    association — ``ir_to_documents`` resolves refs per sub-chunk).
    """
    try:
        from markitdown import MarkItDown
    except ImportError as exc:  # pragma: no cover - exercised via parse_with_markitdown
        raise RuntimeError(
            "MarkItDown parsing requires the markitdown package: "
            'pip install -e "."  (or pip install "markitdown[docx,pptx,xlsx]")'
        ) from exc

    md = MarkItDown().convert(asset.file_path, keep_data_uris=True)
    markdown_text = getattr(md, "text_content", "") or ""

    output_dir = get_parsed_dir() / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    # Decode data-URL images to files and rewrite the refs to images/...
    # before splitting, so the ref physically lives in a block's text and
    # extract_markdown_image_refs (run per sub-chunk in ir_to_documents)
    # attaches the image to the right chunk.
    images: list[ImageRef] = []
    markdown_text = _rewrite_data_url_images(markdown_text, images_dir, images)

    blocks = _markdown_to_blocks(markdown_text)

    markdown_paths: list[str] = []
    if blocks:
        md_path = output_dir / "markitdown_export.md"
        md_path.write_text("\n\n".join(b.text for b in blocks), encoding="utf-8")
        markdown_paths.append(str(md_path))

    return DocumentIR(
        blocks=blocks,
        images=images,
        asset=asset,
        parser="markitdown",
        markdown_paths=markdown_paths,
        images_dir=str(images_dir) if images_dir.exists() else "",
    )


def _rewrite_data_url_images(markdown_text: str, images_dir: Path, images: list[ImageRef]) -> str:
    """Decode every base64 data-URL image ref to disk; rewrite to ``images/...``.

    Mutates ``images`` in place, appending one ``ImageRef`` per decoded
    image (``page=None`` / ``bbox=None`` — MarkItDown has no geometry).
    Returns the markdown with each data URL replaced by an
    ``images/markitdown_<hash>.<ext>`` relative ref. A decode/write failure
    drops just that image (ref removed) rather than failing the whole parse
    — an unrenderable inline image shouldn't abort a text-heavy document.
    """
    if not markdown_text:
        return markdown_text

    def _replace(match: re.Match[str]) -> str:
        mime_subtype = match.group(2).lower()
        payload = match.group(3)
        suffix = _SUFFIX_BY_MIME.get(mime_subtype, "png")
        try:
            raw = base64.b64decode(payload)
        except Exception:
            return ""
        digest = hashlib.md5(raw).hexdigest()[:12]
        images_dir.mkdir(parents=True, exist_ok=True)
        fname = f"markitdown_{digest}.{suffix}"
        try:
            (images_dir / fname).write_bytes(raw)
        except Exception:
            return ""
        rel = f"images/{fname}"
        images.append(ImageRef(path=rel, page=None, bbox=None))
        return f"![]({rel})"

    return _DATA_URL_IMAGE_RE.sub(_replace, markdown_text)


def _markdown_to_blocks(markdown_text: str) -> list[Block]:
    """Split markdown on blank-line boundaries into ``Block``s.

    An ATX heading line (``# Title``) becomes a heading block carrying the
    heading text (``heading=text``, ``level`` = the ``#`` count). Anything
    else is a body block. Empty fragments are dropped. MarkItDown emits
    ``\\n\\n`` between paragraphs/sections, so that is the natural block
    boundary; ``ir_to_documents``' recursive splitter then re-chunks any
    oversized block on token budget.
    """
    blocks: list[Block] = []
    for raw in re.split(r"\n{2,}", markdown_text):
        text = raw.strip()
        if not text:
            continue
        m = _ATX_HEADING_RE.match(text)
        if m:
            heading = m.group(2).strip()
            if heading:
                blocks.append(Block(text=heading, heading=heading, level=len(m.group(1))))
                continue
        blocks.append(Block(text=text))
    return blocks
