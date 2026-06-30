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
                "Image embedding requires the [clip] extra: `pip install mm-asset-rag[clip]`"
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

    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple text strings in one model call.

        sentence-transformers' CLIP ``encode`` accepts a list of strings
        and amortises model overhead; this is significantly faster than
        calling :meth:`embed_text` in a loop for >2 strings.
        """
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return [[float(v) for v in row.tolist()] for row in vectors]

    def embed_image_batch(self, image_paths: list[Path]) -> list[list[float]]:
        """Encode multiple images in one model call.

        Like :meth:`embed_text_batch` but for image inputs. Images are
        loaded eagerly and PIL-converted to RGB before the batched
        ``model.encode`` call. The single model invocation amortises
        the per-image CPU/GPU setup that the per-image ``embed_image``
        loop repeats.
        """
        if not image_paths:
            return []
        from PIL import Image

        model = self._load_model()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        vectors = model.encode(images, normalize_embeddings=True, batch_size=32)
        return [[float(v) for v in row.tolist()] for row in vectors]

    def embed_batch(self, contents: list[Any]) -> list[list[float]]:
        """Mixed batch — text and Path items in one call.

        Splits the input by type, calls the appropriate batched encode
        for each, then reassembles in the original order. Falls back
        to the per-item :meth:`embed` for unrecognised types so the
        contract ``len(embed_batch(xs)) == len(xs)`` always holds.
        """
        if not contents:
            return []
        texts: list[str] = []
        text_indices: list[int] = []
        images: list[Path] = []
        image_indices: list[int] = []
        for i, c in enumerate(contents):
            if isinstance(c, str):
                texts.append(c)
                text_indices.append(i)
            elif isinstance(c, Path):
                images.append(c)
                image_indices.append(i)
            else:
                # Unrecognised type — fall back to single-item encode.
                return [self.embed(c) for c in contents]
        text_vecs = self.embed_text_batch(texts) if texts else []
        image_vecs = self.embed_image_batch(images) if images else []
        out: list[list[float] | None] = [None] * len(contents)
        for idx, vec in zip(text_indices, text_vecs):
            out[idx] = vec
        for idx, vec in zip(image_indices, image_vecs):
            out[idx] = vec
        # ``None`` slots indicate a fall-through; should not happen with
        # the dispatch above but assert defensively.
        return [v if v is not None else [] for v in out]

    def _load_model(self):
        if self._model is None:
            self._check_available()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


# Backward-compat alias. New code should use ``ImageEmbedder``.
ImageEmbeddingProvider = ImageEmbedder
