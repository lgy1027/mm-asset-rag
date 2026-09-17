"""Embedder implementations and their registration.

Add an embedder by registering an implementation of ``Embedder`` here. The
active backend remains responsible for indexing and querying that modality.
"""

from __future__ import annotations

from threading import Lock

from ..core.registry import get_embedder, register_embedder
from .cn_clip_embedder import CnClipImageEmbedder, CnClipImageUnavailable
from .image_embedder import ImageEmbedder, ImageEmbeddingUnavailable
from .reranker import Reranker, get_default_reranker, reset_reranker
from .text_embedder import (
    EmbeddingConfigError,
    TextEmbedder,
    build_default_text_embedder,
)

__all__ = [
    "CnClipImageEmbedder",
    "CnClipImageUnavailable",
    "EmbeddingConfigError",
    "ImageEmbedder",
    "ImageEmbeddingUnavailable",
    "Reranker",
    "TextEmbedder",
    "build_default_image_embedder",
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
    from ..core.registry import embedders as _embedders

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
    from ..core.registry import embedders as _embedders

    with _REGISTER_LOCK:
        if _DEFAULT_IMAGE_KEY in _embedders:
            return
        # See note in ``_ensure_text_registered``: log instead of
        # silently suppressing the missing-CLIP / missing-Pillow error.
        try:
            _embedders.register(_DEFAULT_IMAGE_KEY, build_default_image_embedder(), replace=False)
        except (ImageEmbeddingUnavailable, CnClipImageUnavailable) as exc:
            print(f"[embedders] default image embedder not registered: {exc}")
        except ValueError:
            pass


def build_default_image_embedder():
    """Build the configured CLIP image embedder."""
    from ..core.settings import get_settings

    s = get_settings()
    if s.image_provider == "cn_clip":
        return CnClipImageEmbedder()
    return ImageEmbedder()


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


def get_default_image_embedder():
    """Return the process-wide default image embedder.

    See :func:`get_default_text_embedder` for the slot convention and
    the one-shot retry on a transient empty registry. The concrete
    subclass (sentence-transformers ``ImageEmbedder`` or
    ``CnClipImageEmbedder``) depends on ``Settings.image_provider``;
    see :func:`build_default_image_embedder`.
    """
    _ensure_image_registered()
    try:
        return get_embedder(*_DEFAULT_IMAGE_KEY)  # type: ignore[return-value]
    except KeyError:
        _ensure_image_registered()
        try:
            return get_embedder(*_DEFAULT_IMAGE_KEY)  # type: ignore[return-value]
        except KeyError:
            # Registry still empty after a retry → factory genuinely failing.
            # 走工厂让其真实错误上报(cn_clip 缺 transformers / sentence-transformers 缺
            # [clip] extra),而不是统一报 "需要 [clip] extra"。
            return build_default_image_embedder()
