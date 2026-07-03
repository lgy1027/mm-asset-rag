"""Embedder implementations and their registration.

Adding a new modality (audio, video frame, …) is a three-line change:

1. Drop ``audio_embedder.py`` here whose class satisfies
   ``mm_asset_rag.protocols.Embedder``.
2. ``register_embedder(...)`` below.
3. The active collection naming + dim lookup in ``backends.qdrant_backend``
   picks it up automatically.
"""

from __future__ import annotations

import contextlib
from threading import Lock

from ..registry import get_embedder, register_embedder
from .image_embedder import ImageEmbedder, ImageEmbeddingUnavailable
from .text_embedder import (
    EmbeddingConfigError,
    SentenceTransformerTextEmbedder,
    TextEmbedder,
    build_default_text_embedder,
)

__all__ = [
    "EmbeddingConfigError",
    "ImageEmbedder",
    "ImageEmbeddingUnavailable",
    "SentenceTransformerTextEmbedder",
    "TextEmbedder",
    "build_default_text_embedder",
    "get_default_image_embedder",
    "get_default_text_embedder",
    "register_embedder",
]


# Lazy registration: instantiating ``TextEmbedder`` at import time
# would crash in environments without embedding credentials. We instead
# defer construction to the first call into ``get_default_text_embedder``;
# tests can still ``register_embedder`` a custom instance via
# ``replace=True`` and it wins.
_DEFAULT_TEXT_KEY = ("text", "default")
_DEFAULT_IMAGE_KEY = ("image", "default")
_REGISTER_LOCK = Lock()


def _ensure_text_registered() -> None:
    from ..registry import embedders as _embedders

    with _REGISTER_LOCK, contextlib.suppress(EmbeddingConfigError):
        if _DEFAULT_TEXT_KEY in _embedders:
            return
        # ``suppress`` is correct here: the absence of an embedding
        # backend is a non-fatal runtime condition (the deployer
        # hasn't set credentials). Downstream callers that actually
        # try to embed something will get the same ``EmbeddingConfigError``
        # at use time, with full context — far better than crashing
        # the whole package import. ``build_default_text_embedder``
        # picks OpenAI or sentence-transformers based on Settings.
        _embedders.register(_DEFAULT_TEXT_KEY, build_default_text_embedder(), replace=False)


def _ensure_image_registered() -> None:
    from ..registry import embedders as _embedders

    with _REGISTER_LOCK, contextlib.suppress(ImageEmbeddingUnavailable):
        if _DEFAULT_IMAGE_KEY in _embedders:
            return
        # See note in ``_ensure_text_registered`` about why we
        # suppress the missing-CLIP / missing-Pillow error.
        _embedders.register(_DEFAULT_IMAGE_KEY, ImageEmbedder(), replace=False)


def get_default_text_embedder() -> TextEmbedder:
    """Return the process-wide default :class:`TextEmbedder`.

    The instance is created on first call and cached in the
    ``embedders`` registry under the ``("text", "default")`` slot;
    production code never needs to construct a ``TextEmbedder``
    directly. Tests can replace the default by registering a stub
    with ``embedders.register(("text", "default"), stub, replace=True)``.
    """
    _ensure_text_registered()
    return get_embedder(*_DEFAULT_TEXT_KEY)  # type: ignore[return-value]


def get_default_image_embedder() -> ImageEmbedder:
    """Return the process-wide default :class:`ImageEmbedder`.

    See :func:`get_default_text_embedder` for the slot convention.
    """
    _ensure_image_registered()
    return get_embedder(*_DEFAULT_IMAGE_KEY)  # type: ignore[return-value]
