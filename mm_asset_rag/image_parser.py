"""Deprecated import shim — use ``mm_asset_rag.parsers.image_parser`` instead."""

from __future__ import annotations

import warnings

from .parsers.image_parser import parse_image

warnings.warn(
    "mm_asset_rag.image_parser has moved to mm_asset_rag.parsers.image_parser; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["parse_image"]