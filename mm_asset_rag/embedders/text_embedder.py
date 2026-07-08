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
        truncated = [
            t if len(t) <= self.max_input_chars else t[: self.max_input_chars] for t in texts
        ]
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


class SentenceTransformerTextEmbedder:
    """Local sentence-transformers backend for multilingual / cross-language corpora.

    Falls back to ``TextEmbedder`` for the OpenAI-compatible path —
    switch by setting ``EMBEDDING_BACKEND=sentence_transformers`` and
    ``EMBEDDING_MODEL=BAAI/bge-m3`` (or another HF model id). The
    same Protocol contract holds: ``modality == "text"``,
    ``name == model``, ``dim()`` lazily probes a tiny encode.
    """

    modality = "text"

    def __init__(
        self,
        *,
        model: str | None = None,
        batch_size: int | None = None,
        max_input_chars: int | None = None,
        settings: object | None = None,
    ) -> None:
        from ..settings import get_settings

        s = settings or get_settings()
        self.model = model or s.embedding_model
        if not self.model:
            raise EmbeddingConfigError(
                "SentenceTransformerTextEmbedder requires EMBEDDING_MODEL set to a "
                "HuggingFace model id (e.g. BAAI/bge-m3, intfloat/multilingual-e5-large)."
            )
        self.batch_size = batch_size or s.embedding_batch_size
        self.max_input_chars = max_input_chars or s.embedding_max_input_chars
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingConfigError(
                    "sentence-transformers is not installed. "
                    "Install with `pip install mm-asset-rag[clip]` "
                    "or use EMBEDDING_BACKEND=openai instead."
                ) from exc
            self._model = SentenceTransformer(self.model)
        return self._model

    @property
    def name(self) -> str:
        return self.model

    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("probe"))
        return self._dim

    def embed(self, content: Any) -> list[float]:
        return self.embed_batch([content])[0]

    def embed_batch(self, contents: list[Any]) -> list[list[float]]:
        if not contents:
            return []
        texts = [str(c) for c in contents]
        truncated = [
            t if len(t) <= self.max_input_chars else t[: self.max_input_chars] for t in texts
        ]
        model = self._load()
        vectors = model.encode(
            truncated,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(v) for v in row.tolist()] for row in vectors]

    # ─── Optional sparse / ColBERT capabilities ──────────────────────────
    # These are *model-dependent*: only bge-m3 (and a few others) expose
    # lexical sparse vectors and late-interaction ColBERT vectors through
    # sentence-transformers' ``encode(return_sparse=True,
    # return_colbert_vecs=True)``. ``qdrant_backend`` probes with
    # ``getattr(embedder, "embed_text_sparse", None)`` so the default
    # OpenAI-compatible :class:`TextEmbedder` (which does not implement
    # these) is unaffected — its text collection schema stays dense + BM25
    # + BM25-zh, no reindex.

    def _supports_sparse_colbert(self) -> bool:
        """True iff the configured model is known to support sparse + ColBERT.

        Currently only bge-m3 models expose both. We check by name rather
        than probing at runtime because probing requires a model load
        (and a non-supporting model raises on the kwargs).
        """
        return "bge-m3" in (self.model or "").lower()

    def embed_text_sparse(self, text: str) -> object | None:
        """Return a Qdrant ``SparseVector``-compatible dict, or ``None``.

        ``None`` signals "this embedder does not support sparse vectors"
        — the caller must skip the sparse prefetch / vector field. The
        return value is a plain ``dict`` with ``indices`` / ``values``
        lists (qdrant-client accepts this shape) so this module does not
        need to import ``qdrant_client.models`` at module top-level.
        """
        if not self._supports_sparse_colbert():
            return None
        model = self._load()
        truncated = text if len(text) <= self.max_input_chars else text[: self.max_input_chars]
        result = model.encode(
            [truncated],
            batch_size=1,
            return_sparse=True,
            show_progress_bar=False,
        )
        # sentence-transformers returns a dict-like for sparse:
        # {"sparse": {"indices": [...], "values": [...]}} when
        # ``return_sparse=True``. The exact container varies by
        # version; normalise to a plain dict for qdrant-client.
        sparse = None
        if isinstance(result, dict) or (hasattr(result, "get") and callable(result.get)):
            sparse = result.get("sparse")  # type: ignore[union-attr]
        if sparse is None:
            return None
        if isinstance(sparse, list):
            sparse = sparse[0] if sparse else None
        if sparse is None:
            return None
        indices = list(sparse.get("indices", []))
        values = list(sparse.get("values", []))
        if not indices:
            return None
        return {"indices": [int(i) for i in indices], "values": [float(v) for v in values]}

    def embed_text_colbert(self, text: str) -> list[list[float]] | None:
        """Return ColBERT late-interaction token vectors, or ``None``.

        ``None`` signals "not supported by this embedder". The caller
        must skip the ColBERT multi-vector prefetch / field. The return
        is a list of token vectors (each a list[float]).
        """
        if not self._supports_sparse_colbert():
            return None
        model = self._load()
        truncated = text if len(text) <= self.max_input_chars else text[: self.max_input_chars]
        result = model.encode(
            [truncated],
            batch_size=1,
            return_colbert_vecs=True,
            show_progress_bar=False,
        )
        colbert = None
        if isinstance(result, dict) or (hasattr(result, "get") and callable(result.get)):
            colbert = result.get("colbert_vecs")  # type: ignore[union-attr]
        if colbert is None:
            return None
        if isinstance(colbert, list) and colbert:
            first = colbert[0]
            if first is None:
                return None
            return [[float(v) for v in row] for row in first]
        return None


def build_default_text_embedder() -> TextEmbedder | SentenceTransformerTextEmbedder:
    """Factory: pick the right embedder for ``Settings.embedding_backend``.

    The default registry key (``("text", "default")``) routes through
    this factory so users can switch providers without code changes.
    """
    from ..settings import get_settings

    s = get_settings()
    if s.embedding_backend == "sentence_transformers":
        return SentenceTransformerTextEmbedder(settings=s)
    return TextEmbedder(settings=s)
