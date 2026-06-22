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
from .pdf_parser import parse_pdf, parse_with_paddleocr_vl, parse_pdf_with_pymupdf

__all__ = [
    "parse_image",
    "parse_pdf",
    "parse_with_paddleocr_vl",
    "parse_pdf_with_pymupdf",
]


class _PyMuPdfParser:
    name = "pymupdf"
    source_type = "pdf"

    def parse(self, asset, **options):
        from ..schema import ParsedDocument  # noqa: F401  (avoid circular at import)

        return parse_pdf(asset, parser="pymupdf")


class _PaddleOcrVlParser:
    name = "paddleocr_vl"
    source_type = "pdf"

    def parse(self, asset, **options):
        return parse_pdf(asset, parser="paddleocr_vl")


# Register PDF parsers. ``parse_image`` is dispatched via the legacy
# image_parser module path because the underlying ImageEmbeddingProvider
# differs by source; an AudioParser would slot in here the same way.
register_parser(_PyMuPdfParser())
register_parser(_PaddleOcrVlParser())