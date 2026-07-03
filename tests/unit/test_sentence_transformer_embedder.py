"""Tests for ``SentenceTransformerTextEmbedder`` and the factory selector.

We don't actually load a HF model in tests — that would pull hundreds
of MB on first run. Instead we replace ``_load`` with a stub that
returns deterministic vectors and exercise the protocol surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from mm_asset_rag.embedders.text_embedder import (
    EmbeddingConfigError,
    SentenceTransformerTextEmbedder,
    build_default_text_embedder,
)
from mm_asset_rag.settings import Settings


class _StubSentenceTransformer:
    """Minimal stand-in for the HF model: returns fixed-dim unit vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        out = []
        for i, _t in enumerate(texts):
            v = np.zeros(self.dim, dtype=np.float32)
            v[i % self.dim] = 1.0
            out.append(v)
        return np.stack(out)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_sentence_transformer_requires_model(monkeypatch) -> None:
    for key in ("EMBEDDING_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    with pytest.raises(EmbeddingConfigError, match="EMBEDDING_MODEL"):
        SentenceTransformerTextEmbedder(settings=s)


def test_sentence_transformer_embed_returns_vectors(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-model")
    s = Settings(_env_file=None)
    emb = SentenceTransformerTextEmbedder(settings=s)
    # Inject the stub loader so we don't touch the network.
    emb._model = _StubSentenceTransformer(dim=8)
    emb._dim = 8
    out = emb.embed_batch(["hello", "world"])
    assert len(out) == 2
    assert all(len(v) == 8 for v in out)
    # Same input set should produce unit-norm vectors (normalize_embeddings=True in real model).
    for v in out:
        assert all(isinstance(x, float) for x in v)


def test_sentence_transformer_dim_cached(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-model")
    s = Settings(_env_file=None)
    emb = SentenceTransformerTextEmbedder(settings=s)
    emb._model = _StubSentenceTransformer(dim=16)
    emb._dim = 16
    assert emb.dim() == 16
    # dim() should hit the cache and not call the model again.
    emb._model = None  # would raise if dim() tried to load
    assert emb.dim() == 16


def test_factory_picks_sentence_transformers_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "sentence_transformers")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-model")
    Settings(_env_file=None)  # verify Settings can be constructed in isolation
    emb = build_default_text_embedder()
    assert isinstance(emb, SentenceTransformerTextEmbedder)
    assert emb.model == "fake-model"


def test_factory_picks_openai_by_default(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "text-embedding-3-small")
    Settings(_env_file=None)
    emb = build_default_text_embedder()
    # Concrete type is the OpenAI one — TextEmbedder (not the HF stub).
    from mm_asset_rag.embedders.text_embedder import TextEmbedder

    assert isinstance(emb, TextEmbedder)
    assert not isinstance(emb, SentenceTransformerTextEmbedder)
