"""Deprecated import shim — use ``mm_asset_rag.embedders`` instead.

The original ``providers.py`` was split into:

* ``mm_asset_rag.embedders.text_embedder.TextEmbedder`` — replaces
  ``EmbeddingProvider``.
* ``mm_asset_rag.embedders.image_embedder.ImageEmbedder`` — replaces
  ``ImageEmbeddingProvider``.
* ``EmbeddingConfigError`` / ``ImageEmbeddingUnavailable`` — re-exported
  from ``mm_asset_rag.embedders``.

The prior ``configure_embedding()`` helper (which set the LlamaIndex
``Settings.embed_model`` global) has been removed: ``TextEmbedder`` is a
normal instance and can be constructed / injected without touching global
state. There is no llama-index dependency in the codebase anymore.
"""

from __future__ import annotations

import warnings

from .embedders.image_embedder import ImageEmbeddingProvider, ImageEmbeddingUnavailable  # noqa: F401
from .embedders.text_embedder import EmbeddingConfigError, EmbeddingProvider  # noqa: F401

# ``TextEmbedder`` / ``ImageEmbedder`` are the new names. Re-export the old
# names for backward compatibility, sourced from the new classes.
from .embedders.image_embedder import ImageEmbedder
from .embedders.text_embedder import TextEmbedder

warnings.warn(
    "mm_asset_rag.providers has moved to mm_asset_rag.embedders; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EmbeddingConfigError",
    "EmbeddingProvider",
    "ImageEmbeddingProvider",
    "ImageEmbeddingUnavailable",
    "TextEmbedder",
    "ImageEmbedder",
]


# Backward-compat aliases: ``EmbeddingProvider`` is now ``TextEmbedder``,
# ``ImageEmbeddingProvider`` is now ``ImageEmbedder``. They were
# constructed differently (the old classes took no args; the new ones read
# from Settings). The aliases keep imports working but emit a deprecation
# warning when constructed.
def EmbeddingProvider(*args, **kwargs):  # type: ignore[no-redef]
    warnings.warn(
        "EmbeddingProvider is deprecated; use TextEmbedder instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return TextEmbedder(*args, **kwargs)


def ImageEmbeddingProvider(*args, **kwargs):  # type: ignore[no-redef]
    warnings.warn(
        "ImageEmbeddingProvider is deprecated; use ImageEmbedder instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ImageEmbedder(*args, **kwargs)