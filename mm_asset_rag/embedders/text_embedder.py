"""OpenAI-compatible text embedder.

Replaces the prior ``EmbeddingProvider`` in ``mm_asset_rag.providers`` after
the embedders/ subpackage split. Implements the :class:`Embedder`
Protocol so it can be registered via :func:`register_embedder`.

Configuration is read from the centralized ``Settings`` instance, falling
back to ``OPENAI_*`` env vars for ``api_key`` / ``base_url`` / ``model``.
"""

from __future__ import annotations

import time
from typing import Any

import requests


class EmbeddingConfigError(RuntimeError):
    """Raised when an embedding request is made without valid configuration."""


class TextEmbedder:
    """OpenAI-compatible text embedding client with retry / backoff.

    Implements the :class:`mm_asset_rag.protocols.Embedder` Protocol:
    ``modality == "text"``, ``name == model``.
    """

    modality = "text"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        request_interval: float | None = None,
        retry_count: int | None = None,
        timeout: float | None = None,
        max_input_chars: int | None = None,
        settings: object | None = None,
    ) -> None:
        """Construct a TextEmbedder.

        All keyword arguments are explicit overrides; ``settings`` is the
        centralized ``Settings`` instance to read from. Defaults to
        ``get_settings()``. Pass ``Settings(_env_file=None)`` (or similar)
        for tests that want to bypass the on-disk ``.env`` file.
        """
        from ..settings import Settings, get_settings

        s = settings or get_settings()
        if s is None:  # pragma: no cover — Settings() never returns None
            s = Settings()
        creds = s.text_embedding_creds
        self.api_key = api_key or creds[0]
        self.base_url = base_url or creds[1]
        self.model = model or creds[2] or "text-embedding-3-small"

        self.batch_size = batch_size or s.embedding_batch_size
        self.request_interval = (
            request_interval if request_interval is not None else s.embedding_request_interval
        )
        self.retry_count = retry_count or s.embedding_retry_count
        self.timeout = timeout or s.embedding_timeout
        self.max_input_chars = max_input_chars or s.embedding_max_input_chars

        if not (self.api_key and self.base_url and self.model):
            raise EmbeddingConfigError(
                "TextEmbedder requires api_key, base_url, and model. "
                "Set OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL (or the "
                "EMBEDDING_* overrides) in your environment or .env file."
            )

    @property
    def name(self) -> str:
        return self.model

    def dim(self) -> int:
        # Lazily probe dimension by embedding a tiny probe string.
        if getattr(self, "_dim", None) is not None:
            return self._dim
        self._dim = len(self.embed("probe"))
        return self._dim

    def embed(self, content: Any) -> list[float]:
        return self.embed_batch([content])[0]

    def embed_batch(self, contents: list[Any]) -> list[list[float]]:
        texts = [str(c) for c in contents]
        truncated = [t if len(t) <= self.max_input_chars else t[: self.max_input_chars] for t in texts]
        vectors: list[list[float]] = []
        for offset in range(0, len(truncated), self.batch_size):
            vectors.extend(self._remote_batch(truncated[offset : offset + self.batch_size]))
            if self.request_interval > 0:
                time.sleep(self.request_interval)
        return vectors

    def _remote_batch(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            response = requests.post(
                self.base_url.rstrip("/") + "/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(5 * (attempt + 1), 30)
                time.sleep(wait)
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
        raise RuntimeError(
            f"Embedding request failed after {self.retry_count} retries (likely rate-limited)"
        )


# Backward-compat alias. New code should use ``TextEmbedder``.
EmbeddingProvider = TextEmbedder