"""HTTP-based providers for embedding, image embedding, LLM, and OCR/VLM.

Every provider talks to a real, configured backend. There is no offline
fallback: missing configuration raises so misconfiguration is caught
early instead of producing silently-wrong embeddings.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from PIL import Image

# Load .env exactly once at import time so the rest of the module sees
# whatever the user configured. Tests can ``monkeypatch.delenv`` without
# us re-reading the file on every call.
load_dotenv()


class EmbeddingConfigError(RuntimeError):
    """Raised when an embedding request is made without valid configuration."""


def configure_embedding() -> str:
    """Configure LlamaIndex ``Settings.embed_model`` from environment.

    Reads ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``OPENAI_MODEL`` (or
    ``EMBEDDING_*`` overrides). Raises :class:`EmbeddingConfigError` when
    any required variable is missing.

    Returns a human-readable description of what was set.
    """
    api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("EMBEDDING_MODEL") or os.environ.get("OPENAI_MODEL")

    if not (api_key and base_url and model):
        raise EmbeddingConfigError(
            "Embedding requires OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL "
            "(or EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL overrides). "
            "Configure them in your environment or .env file."
        )

    Settings.embed_model = OpenAIEmbedding(
        model_name=model,
        api_key=api_key,
        api_base=base_url,
    )
    return f"OpenAIEmbedding({model} via {base_url})"


class EmbeddingProvider:
    """OpenAI-compatible text embedding client with retry/backoff."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.model = os.environ.get("EMBEDDING_MODEL") or os.environ.get("OPENAI_MODEL")

        if not (self.api_key and self.base_url and self.model):
            raise EmbeddingConfigError(
                "EmbeddingProvider requires OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL."
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        max_chars = int(os.environ.get("EMBEDDING_MAX_INPUT_CHARS", "8192"))
        truncated = [t if len(t) <= max_chars else t[:max_chars] for t in texts]
        batch_size = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "5")))
        vectors: list[list[float]] = []
        for offset in range(0, len(truncated), batch_size):
            vectors.extend(
                self._embed_remote_batch(truncated[offset : offset + batch_size])
            )
            interval = float(os.environ.get("EMBEDDING_REQUEST_INTERVAL", "0.25"))
            if interval > 0:
                time.sleep(interval)
        return vectors

    def _embed_remote_batch(self, texts: list[str]) -> list[list[float]]:
        retry_count = int(os.environ.get("EMBEDDING_RETRY_COUNT", "5"))
        timeout = float(os.environ.get("EMBEDDING_TIMEOUT", "120"))
        last_error: Exception | None = None
        for attempt in range(retry_count):
            response = requests.post(
                self.base_url.rstrip("/") + "/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
                timeout=timeout,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else min(5 * (attempt + 1), 30)
                time.sleep(wait_seconds)
                continue
            try:
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 20))
                continue
            data = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
            return [[float(value) for value in item["embedding"]] for item in data]
        if last_error:
            raise last_error
        raise RuntimeError("Embedding request failed after retries, probably due to rate limit")

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class ImageEmbeddingUnavailable(RuntimeError):
    """Raised when image embedding is requested without the [clip] extra installed."""


class ImageEmbeddingProvider:
    """CLIP-based image + text encoder via sentence-transformers.

    Requires the optional ``[clip]`` extra (``pip install mm-asset-rag[clip]``).
    Without it, every call raises :class:`ImageEmbeddingUnavailable` so that
    the caller can decide whether to skip the image route or warn the user.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("CLIP_MODEL", "clip-ViT-B-32")
        self._model = None
        # Try eagerly to give a clear error message at construction time.
        self._check_available()

    @staticmethod
    def is_available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _check_available() -> None:
        if not ImageEmbeddingProvider.is_available():
            raise ImageEmbeddingUnavailable(
                "Image embedding requires the [clip] extra: "
                "`pip install mm-asset-rag[clip]`"
            )

    def embed_image(self, image_path: Path) -> list[float]:
        model = self._load_model()
        image = Image.open(image_path).convert("RGB")
        return [float(v) for v in model.encode(image, normalize_embeddings=True).tolist()]

    def embed_text(self, text: str) -> list[float]:
        model = self._load_model()
        return [float(v) for v in model.encode(text, normalize_embeddings=True).tolist()]

    def _load_model(self):
        if self._model is None:
            self._check_available()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
