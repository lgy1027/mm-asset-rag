"""Deprecated import shim — use ``mm_asset_rag.parsers.pdf_parser`` instead.

Kept so external callers (``from mm_asset_rag.pdf_parser import parse_pdf``)
don't break while the codebase transitions to the new subpackage layout.
"""

from __future__ import annotations

import warnings

from .parsers.pdf_parser import parse_pdf, parse_pdf_with_pymupdf, parse_with_paddleocr_vl

warnings.warn(
    "mm_asset_rag.pdf_parser has moved to mm_asset_rag.parsers.pdf_parser; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["parse_pdf", "parse_pdf_with_pymupdf", "parse_with_paddleocr_vl"]