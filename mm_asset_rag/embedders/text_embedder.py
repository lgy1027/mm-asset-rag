"""Required remote OpenAI-compatible text embedding client."""

from __future__ import annotations

import time
from typing import Any

from ..openai_adapters import OpenAIEmbeddingAdapter
from ..openai_compatible import require_connection


class EmbeddingConfigError(RuntimeError):
    """Raised when the mandatory remote embedding configuration is incomplete."""


class TextEmbedder:
    modality = "text"

    def __init__(
        self, *, api_key=None, base_url=None, model=None, settings=None, **overrides
    ) -> None:
        from ..settings import get_settings

        s = settings or get_settings()
        key, url, configured_model = s.text_embedding_creds
        self.api_key = api_key or key
        self.base_url = base_url or url
        self.model = model or configured_model
        self.batch_size = overrides.get("batch_size") or s.embedding_batch_size
        self.request_interval = overrides.get("request_interval", s.embedding_request_interval)
        self.retry_count = overrides.get("retry_count") or s.embedding_retry_count
        self.timeout = overrides.get("timeout") or s.embedding_timeout
        self.max_input_chars = overrides.get("max_input_chars") or s.embedding_max_input_chars
        if not (self.api_key and self.base_url and self.model):
            raise EmbeddingConfigError(
                "Remote embedding requires OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_BASE_URL, "
                "and EMBEDDING_MODEL (or explicit EMBEDDING_* overrides)."
            )
        self.connection = require_connection(self.base_url, self.api_key)
        self.adapter = OpenAIEmbeddingAdapter(
            connection=self.connection,
            model=self.model,
            timeout=self.timeout,
            retry_count=self.retry_count,
        )

    @property
    def name(self) -> str:
        return self.model

    def dim(self) -> int:
        from ..settings import get_settings

        configured = get_settings().embedding_dim
        if configured is not None:
            return configured
        if not hasattr(self, "_dim"):
            self._dim = len(self.embed("probe"))
        return self._dim

    def embed(self, content: Any) -> list[float]:
        return self.embed_batch([content])[0]

    def embed_batch(self, contents: list[Any]) -> list[list[float]]:
        vectors: list[list[float]] = []
        texts = [str(item)[: self.max_input_chars] for item in contents]
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._remote_batch(texts[start : start + self.batch_size]))
            if self.request_interval:
                time.sleep(self.request_interval)
        return vectors

    def _remote_batch(self, texts: list[str]) -> list[list[float]]:
        return self.adapter.embed_batch(texts)


def build_default_text_embedder() -> TextEmbedder:
    from ..settings import get_settings

    return TextEmbedder(settings=get_settings())
