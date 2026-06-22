"""CLIP-based image + text encoder.

Implements the :class:`Embedder` Protocol with ``modality == "image"`` and
``name == model_name``. Requires the optional ``[clip]`` extra
(``pip install mm-asset-rag[clip]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ImageEmbeddingUnavailable(RuntimeError):
    """Raised when image embedding is requested without the [clip] extra installed."""


class ImageEmbedder:
    """CLIP-style image + text encoder via sentence-transformers."""

    modality = "image"

    def __init__(self, model_name: str | None = None) -> None:
        from ..settings import get_settings

        s = get_settings()
        self.model_name = model_name or s.clip_model
        self._model = None
        # Fail fast at construction so callers get a clear error.
        self._check_available()

    @property
    def name(self) -> str:
        return self.model_name

    @staticmethod
    def is_available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _check_available() -> None:
        if not ImageEmbedder.is_available():
            raise ImageEmbeddingUnavailable(
                "Image embedding requires the [clip] extra: "
                "`pip install mm-asset-rag[clip]`"
            )

    def dim(self) -> int:
        if getattr(self, "_dim", None) is not None:
            return self._dim
        self._dim = len(self.embed_text("probe"))
        return self._dim

    def embed(self, content: Any) -> list[float]:
        if isinstance(content, (str, Path)):
            return self.embed_text(str(content))
        return self.embed_image(Path(content))

    def embed_text(self, text: str) -> list[float]:
        model = self._load_model()
        return [float(v) for v in model.encode(text, normalize_embeddings=True).tolist()]

    def embed_image(self, image_path: Path) -> list[float]:
        from PIL import Image

        model = self._load_model()
        image = Image.open(image_path).convert("RGB")
        return [float(v) for v in model.encode(image, normalize_embeddings=True).tolist()]

    def embed_batch(self, contents: list[Any]) -> list[list[float]]:
        return [self.embed(c) for c in contents]

    def _load_model(self):
        if self._model is None:
            self._check_available()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


# Backward-compat alias. New code should use ``ImageEmbedder``.
ImageEmbeddingProvider = ImageEmbedder