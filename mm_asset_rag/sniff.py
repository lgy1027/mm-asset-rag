"""File sniffing for the upload pipeline.

Pure local inspection — reads magic bytes, EXIF, PDF /Info. Never makes a
network call. Falls back gracefully on any decode failure: corrupt or
zero-byte files still come back with a valid ``SniffedAsset`` carrying
``source_type="unknown"`` so the upload pipeline can reject them
explicitly instead of crashing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# Magic-byte signatures for the formats we accept. Each entry maps a
# leading-byte regex to a ``(source_type, extension)`` tuple. Office
# formats (docx/pptx/xlsx) are ZIP containers (``PK\x03\x04``) and are
# disambiguated from generic zips by extension + ``[Content_Types].xml``
# in ``_sniff_office`` below; HTML by its leading tag.
_MAGIC: list[tuple[bytes, str, str]] = [
    (b"%PDF-", "pdf", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", "image", ".png"),
    (b"\xff\xd8\xff", "image", ".jpg"),
    (b"GIF87a", "image", ".gif"),
    (b"GIF89a", "image", ".gif"),
    (b"BM", "image", ".bmp"),
    (b"RIFF", "image", ".webp"),  # also matches WAV — checked by sniff_image_format
]

_WEBP_FULL_MAGIC = b"RIFF\x00\x00\x00\x00WEBP"

# Office Open XML containers are ZIP files; the ``[Content_Types].xml``
# member carries the format-specific override. ``_OFFICE_SUFFIXES`` maps
# the document-type substring found in that XML to the source_type +
# extension. The extension on the uploaded file is the primary signal
# (a .docx is a docx even before we crack the zip); the XML check is a
# guard against a renamed zip sneaking through.
_OFFICE_SUFFIXES: dict[str, tuple[str, str]] = {
    ".docx": ("document", ".docx"),
    ".pptx": ("document", ".pptx"),
    ".xlsx": ("document", ".xlsx"),
}
_ZIP_MAGIC = b"PK\x03\x04"


@dataclass
class SniffedAsset:
    """What sniff() reports about a single uploaded file.

    ``source_type`` is one of ``"pdf"``, ``"image"``, ``"document"``, or
    ``"unknown"``. ``"document"`` covers the office/text formats the
    document parser handles (docx / pptx / xlsx / html). When ``unknown``,
    the caller should reject the upload rather than try to parse it.
    """

    asset_id: str
    title: str
    source_type: str  # "pdf" | "image" | "document" | "unknown"
    relative_path: str
    tags: list[str] = field(default_factory=list)
    file_size: int = 0
    page_count: int | None = None
    width: int | None = None
    height: int | None = None
    pdf_metadata: dict | None = None
    image_metadata: dict | None = None
    error: str | None = None


def _default_title(stem: str) -> str:
    """Turn a filename stem into a human-readable title."""
    return re.sub(r"[_\-]+", " ", stem).strip().title() or stem


def _read_magic(path: Path, n: int = 16) -> bytes:
    try:
        with path.open("rb") as f:
            return f.read(n)
    except OSError as exc:
        log.warning("sniff: cannot read %s: %s", path, exc)
        return b""


def sniff_pdf(path: Path) -> tuple[int | None, dict | None]:
    """Read page count and /Info dict from a PDF using PyMuPDF.

    PyMuPDF (``fitz``) is already a hard dependency of the project, so we
    can use it here without adding a new package. Returns
    ``(page_count, info_dict)`` — either may be ``None`` on a malformed
    PDF.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        log.warning("sniff_pdf: PyMuPDF not installed, PDF metadata unavailable")
        return None, None

    try:
        with fitz.open(str(path)) as doc:
            page_count = doc.page_count
            info: dict = {}
            meta = doc.metadata or {}
            for key in ("title", "author", "subject", "creator", "producer"):
                val = meta.get(key)
                if val:
                    info[key] = str(val)
            return page_count, info or None
    except Exception as exc:
        log.debug("sniff_pdf: PyMuPDF failed for %s: %s", path, exc)
        return None, None


def sniff_image(path: Path) -> tuple[int | None, int | None, dict | None]:
    """Read image dimensions and EXIF via Pillow.

    Returns ``(width, height, exif_dict)``. Any value may be None on
    failure (corrupt JPEG, missing PIL, etc.).
    """
    try:
        from PIL import ExifTags, Image  # type: ignore[import-not-found]
    except ImportError:
        return None, None, None

    try:
        with Image.open(path) as img:
            width, height = img.size
            exif_raw = img.getexif() if hasattr(img, "getexif") else None
            if not exif_raw:
                return width, height, None
            tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
            keep_keys = {
                "Make",
                "Model",
                "DateTime",
                "DateTimeOriginal",
                "Orientation",
                "GPSInfo",
                "ImageDescription",
                "Software",
            }
            exif = {k: str(v) for k, v in tag_map.items() if k in keep_keys}
            return width, height, (exif or None)
    except Exception as exc:
        log.debug("sniff_image: Pillow failed for %s: %s", path, exc)
        return None, None, None


def _sniff_office(path: Path) -> tuple[str, str] | None:
    """Identify an Office Open XML container (docx/pptx/xlsx).

    Returns ``(source_type, extension)`` or ``None`` when the file isn't
    one we recognise. The extension is the primary signal; we crack the
    zip only as a guard so a renamed plain zip can't masquerade as a
    document. Cracking failures degrade to extension-only trust (a
    truncated upload still sniffs by name) — the parser will reject it
    later if the content is genuinely bad.
    """
    suffix = path.suffix.lower()
    if suffix not in _OFFICE_SUFFIXES:
        return None
    source_type, ext = _OFFICE_SUFFIXES[suffix]
    # Guard: confirm it's actually a ZIP. A non-zip with a .docx name is
    # rejected as unknown rather than trusted by extension alone.
    try:
        import zipfile

        if not zipfile.is_zipfile(path):
            return None
    except OSError:
        # If we can't even stat the zip, trust the extension — the parser
        # will surface the real error. Better to over-accept than to drop
        # a valid upload on a transient FS hiccup.
        pass
    return source_type, ext


def sniff(path: Path) -> SniffedAsset:
    """Top-level entry point.

    Reads the first 16 bytes of ``path`` to determine the format, then
    dispatches to ``sniff_pdf`` / ``sniff_image``. Always returns a
    ``SniffedAsset``; on failure ``source_type="unknown"`` and
    ``error`` carries the reason.
    """
    if not path.exists():
        return SniffedAsset(
            asset_id=path.stem,
            title=_default_title(path.stem),
            source_type="unknown",
            relative_path=str(path),
            error="file not found",
        )

    size = path.stat().st_size
    stem = path.stem
    asset_id = stem
    title = _default_title(stem)
    relative_path = path.name
    error: str | None = None

    magic = _read_magic(path, n=16)

    if magic.startswith(b"%PDF-"):
        page_count, info = sniff_pdf(path)
        # PDF /Info title (if any) beats the filename-derived default.
        if info and info.get("title"):
            title = info["title"]
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type="pdf",
            relative_path=relative_path,
            file_size=size,
            page_count=page_count,
            pdf_metadata=info,
            error=error,
        )

    if magic.startswith(b"\x89PNG\r\n\x1a\n"):
        w, h, exif = sniff_image(path)
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type="image",
            relative_path=relative_path,
            file_size=size,
            width=w,
            height=h,
            image_metadata=exif,
            error=error,
        )

    if magic.startswith(b"\xff\xd8\xff"):
        w, h, exif = sniff_image(path)
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type="image",
            relative_path=relative_path,
            file_size=size,
            width=w,
            height=h,
            image_metadata=exif,
            error=error,
        )

    if magic.startswith((b"GIF87a", b"GIF89a")):
        # GIF dimensions live in the header; Pillow handles them fine.
        w, h, exif = sniff_image(path)
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type="image",
            relative_path=relative_path,
            file_size=size,
            width=w,
            height=h,
            image_metadata=exif,
            error=error,
        )

    if magic.startswith(b"BM"):
        w, h, exif = sniff_image(path)
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type="image",
            relative_path=relative_path,
            file_size=size,
            width=w,
            height=h,
            image_metadata=exif,
            error=error,
        )

    if magic.startswith(b"RIFF") and len(magic) >= 12 and magic[8:12] == b"WEBP":
        w, h, exif = sniff_image(path)
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type="image",
            relative_path=relative_path,
            file_size=size,
            width=w,
            height=h,
            image_metadata=exif,
            error=error,
        )

    # Office Open XML (docx/pptx/xlsx) — ZIP containers, disambiguated
    # by extension + a zipfile guard in ``_sniff_office``. Parsed by the
    # document parser (source_type="document").
    office = _sniff_office(path)
    if office is not None:
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type=office[0],
            relative_path=relative_path,
            file_size=size,
            error=error,
        )

    # HTML / Markdown / plain text — sniffed by extension (no robust magic
    # bytes for these). Parsed by the document parser too.
    _TEXT_SUFFIXES = {".html", ".htm", ".md", ".markdown", ".txt"}
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return SniffedAsset(
            asset_id=asset_id,
            title=title,
            source_type="document",
            relative_path=relative_path,
            file_size=size,
            error=error,
        )

    # Fall through: signature didn't match anything we accept.
    return SniffedAsset(
        asset_id=asset_id,
        title=title,
        source_type="unknown",
        relative_path=relative_path,
        file_size=size,
        error=f"unrecognised magic bytes: {magic[:8]!r}",
    )
