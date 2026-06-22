"""Deprecated import shim — use ``mm_asset_rag.backends.qdrant_backend`` instead."""

from __future__ import annotations

import warnings

from .backends.qdrant_backend import (  # noqa: F401  (re-export)
    build_qdrant_image_index,
    build_qdrant_text_index,
    get_qdrant_client,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)

warnings.warn(
    "mm_asset_rag.qdrant_store has moved to mm_asset_rag.backends.qdrant_backend; "
    "update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "build_qdrant_image_index",
    "build_qdrant_text_index",
    "get_qdrant_client",
    "qdrant_image_to_image_search",
    "qdrant_text_search",
    "qdrant_text_to_image_search",
]