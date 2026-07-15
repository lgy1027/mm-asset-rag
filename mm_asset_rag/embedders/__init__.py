"""Embedder implementations and their registration.

Adding a new modality (audio, video frame, …) is a three-line change:

1. Drop ``audio_embedder.py`` here whose class satisfies
   ``mm_asset_rag.protocols.Embedder``.
2. ``register_embedder(...)`` below.
3. The active collection naming + dim lookup in ``backends.qdrant_backend``
   picks it up automatically.
"""

from __future__ import annotations

from threading import Lock

from ..registry import get_embedder, register_embedder
from .image_embedder import ImageEmbedder, ImageEmbeddingUnavailable
from .reranker import Reranker, get_default_reranker, reset_reranker
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
    "Reranker",
    "SentenceTransformerTextEmbedder",
    "TextEmbedder",
    "build_default_text_embedder",
    "get_default_image_embedder",
    "get_default_reranker",
    "get_default_text_embedder",
    "register_embedder",
    "reset_reranker",
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

    with _REGISTER_LOCK:
        if _DEFAULT_TEXT_KEY in _embedders:
            return
        # The absence of an embedding backend is a non-fatal runtime
        # condition (the deployer hasn't set credentials), so we do not
        # let it crash package import. But we no longer *silently*
        # swallow ``EmbeddingConfigError``: a daemon thread's first
        # ``build_default_text_embedder()`` can transiently fail on a
        # race (credentials read before ``.env`` settled, an import
        # lock, etc.), and a silent suppress leaves the registry empty
        # so the downstream ``get_embedder`` raises an opaque
        # ``KeyError: ... not registered; available: []`` with no clue
        # why. Logging the real reason here makes that failure legible
        # the next time it happens — and ``get_default_text_embedder``
        # below retries the ensure once to ride over the transient race.
        try:
            _embedders.register(_DEFAULT_TEXT_KEY, build_default_text_embedder(), replace=False)
        except EmbeddingConfigError as exc:
            print(f"[embedders] default text embedder not registered: {exc}")
        except ValueError:
            # Another thread won the race and registered first under the
            # same key (``replace=False``); that is the desired end state.
            pass


def _ensure_image_registered() -> None:
    from ..registry import embedders as _embedders

    with _REGISTER_LOCK:
        if _DEFAULT_IMAGE_KEY in _embedders:
            return
        # See note in ``_ensure_text_registered``: log instead of
        # silently suppressing the missing-CLIP / missing-Pillow error.
        try:
            _embedders.register(_DEFAULT_IMAGE_KEY, ImageEmbedder(), replace=False)
        except ImageEmbeddingUnavailable as exc:
            print(f"[embedders] default image embedder not registered: {exc}")
        except ValueError:
            pass


def get_default_text_embedder() -> TextEmbedder:
    """Return the process-wide default :class:`TextEmbedder`.

    The instance is created on first call and cached in the
    ``embedders`` registry under the ``("text", "default")`` slot;
    production code never needs to construct a ``TextEmbedder``
    directly. Tests can replace the default by registering a stub
    with ``embedders.register(("text", "default"), stub, replace=True)``.

    If the first ``_ensure_text_registered()`` left the slot empty
    (a transient race made ``build_default_text_embedder()`` raise
    ``EmbeddingConfigError`` once), retry the ensure once before
    surfacing the error — the transient failure usually does not
    recur. If the slot is *still* empty after the retry, re-invoke
    ``build_default_text_embedder()`` so its ``EmbeddingConfigError``
    (the real, legible cause — missing credentials) propagates
    instead of the opaque ``KeyError: ... not registered; available: []``.
    """
    _ensure_text_registered()
    try:
        return get_embedder(*_DEFAULT_TEXT_KEY)  # type: ignore[return-value]
    except KeyError:
        _ensure_text_registered()
        try:
            return get_embedder(*_DEFAULT_TEXT_KEY)  # type: ignore[return-value]
        except KeyError:
            # Registry still empty after a retry → build_default is
            # genuinely failing (missing creds). Let its real error
            # propagate so the caller sees *why*, not just "not registered".
            return build_default_text_embedder()  # raises EmbeddingConfigError


def get_default_image_embedder() -> ImageEmbedder:
    """Return the process-wide default :class:`ImageEmbedder`.

    See :func:`get_default_text_embedder` for the slot convention and
    the one-shot retry on a transient empty registry.
    """
    _ensure_image_registered()
    try:
        return get_embedder(*_DEFAULT_IMAGE_KEY)  # type: ignore[return-value]
    except KeyError:
        _ensure_image_registered()
        try:
            return get_embedder(*_DEFAULT_IMAGE_KEY)  # type: ignore[return-value]
        except KeyError:
            # Still empty after retry → CLIP/Pillow genuinely unavailable.
            return ImageEmbedder()  # raises ImageEmbeddingUnavailable
