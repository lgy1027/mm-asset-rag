"""Parser implementations and their registration.

Adding a new modality is a three-line change:

1. Drop a file like ``audio_parser.py`` here whose class satisfies
   ``mm_asset_rag.protocols.Parser``.
2. Import + ``register_parser`` it below.
3. (Optional) ship a CLI flag in ``mm_asset_rag.cli`` that calls
   ``get_parser("audio", name)`` based on user input.
"""

from __future__ import annotations

from ..registry import register_parser
from .image_parser import parse_image
from .pdf_parser import (
    parse_pdf,
    parse_pdf_with_pymupdf,
    parse_with_docling,
    parse_with_markitdown,
    parse_with_paddleocr_vl,
)

__all__ = [
    "parse_image",
    "parse_pdf",
    "parse_pdf_with_pymupdf",
    "parse_with_docling",
    "parse_with_markitdown",
    "parse_with_paddleocr_vl",
]


class _PyMuPdfParser:
    name = "pymupdf"
    source_type = "pdf"

    def parse(self, asset, **options):
        # PDFs carry their text natively — ``enable_ocr`` / ``enable_vlm``
        # from the image parser don't apply here. ``paddleocr_vl`` is
        # the OCR-equivalent path for image-only PDFs; callers choose
        # via ``options["pdf_parser"]`` instead.
        _ = options  # explicit discard; keep the signature so a future
        # option (e.g. ``enable_ocr_fallback``) doesn't
        # require a synchronous signature change here.
        return parse_pdf(asset, parser="pymupdf")


class _PaddleOcrVlParser:
    name = "paddleocr_vl"
    source_type = "pdf"

    def parse(self, asset, **options):
        _ = options
        return parse_pdf(asset, parser="paddleocr_vl")


class _AutoPdfParser:
    """Dispatch to ``paddleocr_vl`` when its API is configured, else ``pymupdf``.

    Registered as ``pdf/auto`` so the registry path mirrors the
    ``parse_pdf(..., parser="auto")`` branch in
    :mod:`mm_asset_rag.parsers.pdf_parser`. Without this, callers
    that pass ``pdf_parser="auto"`` (the default in ``ParseOptions``
    and ``Settings``) hit ``KeyError: parser ('pdf', 'auto') not
    registered`` before any dispatch logic can run.
    """

    name = "auto"
    source_type = "pdf"

    def parse(self, asset, **options):
        _ = options
        return parse_pdf(asset, parser="auto")


class _ImageParser:
    name = "image"
    source_type = "image"

    def parse(self, asset, **options):
        return parse_image(
            asset,
            enable_ocr=bool(options.get("enable_ocr", False)),
            enable_vlm=bool(options.get("enable_vlm", False)),
        )


class _DoclingPdfParser:
    """docling as a PDF backend (``--pdf-parser docling``).

    Registered under ``source_type="pdf"`` so ``parse_pdf(parser="docling")``
    has a registry entry. The real work is in ``parse_with_docling``; the
    lazy docling import means this registers fine even when the
    ``[docling]`` extra isn't installed (the error surfaces at parse time).
    """

    name = "docling"
    source_type = "pdf"

    def parse(self, asset, **options):
        _ = options
        return parse_with_docling(asset)


class _DoclingDocumentParser:
    """docling for office/text formats (docx / pptx / xlsx / html / md).

    Registered under ``source_type="document"`` — the source_type ``sniff``
    assigns to those formats. Same adapter as the PDF one; both call
    ``parse_with_docling``. Optional heavy backend; selected with
    ``--document-parser docling``.
    """

    name = "docling"
    source_type = "document"

    def parse(self, asset, **options):
        _ = options
        return parse_with_docling(asset)


class _MarkItDownDocumentParser:
    """MarkItDown for office/text formats (docx / pptx / xlsx / html / md).

    The default ``document`` backend (core dependency, no ML stack). Same
    ``source_type="document"`` registration as docling; ``_do_parse`` picks
    which by ``options.document_parser`` (default ``markitdown``).
    """

    name = "markitdown"
    source_type = "document"

    def parse(self, asset, **options):
        _ = options
        return parse_with_markitdown(asset)


# Register all built-in parsers. Image parsing now flows through the same
# registry contract as PDFs, so future modalities don't need special cases.
register_parser(_PyMuPdfParser())
register_parser(_PaddleOcrVlParser())
register_parser(_AutoPdfParser())
register_parser(_DoclingPdfParser())
register_parser(_DoclingDocumentParser())
register_parser(_MarkItDownDocumentParser())
register_parser(_ImageParser())
