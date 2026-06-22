"""Embedder implementations and their registration.

Adding a new modality (audio, video frame, …) is a three-line change:

1. Drop ``audio_embedder.py`` here whose class satisfies
   ``mm_asset_rag.protocols.Embedder``.
2. ``register_embedder(...)`` below.
3. The active collection naming + dim lookup in ``backends.qdrant_backend``
   picks it up automatically.
"""

from __future__ import annotations

from ..registry import register_embedder
from .image_embedder import ImageEmbedder, ImageEmbeddingUnavailable
from .text_embedder import EmbeddingConfigError, TextEmbedder

__all__ = [
    "EmbeddingConfigError",
    "ImageEmbedder",
    "ImageEmbeddingUnavailable",
    "TextEmbedder",
]


# Register the default text embedder under a fixed well-known key so the
# rest of the codebase can call ``get_embedder("text", "default")``.
# An audio or video embedder would call ``register_embedder`` here too.
_text = TextEmbedder()
register_embedder(_text, replace=True)