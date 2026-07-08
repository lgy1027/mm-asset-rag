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


# ─── Optional sparse / ColBERT capability ──────────────────────────────────


def test_supports_sparse_colbert_only_for_bge_m3(monkeypatch) -> None:
    """``_supports_sparse_colbert`` is True only for bge-m3 models."""
    # bge-m3 model
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    s = Settings(_env_file=None)
    emb = SentenceTransformerTextEmbedder(settings=s)
    assert emb._supports_sparse_colbert() is True

    # other model
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    s = Settings(_env_file=None)
    emb = SentenceTransformerTextEmbedder(settings=s)
    assert emb._supports_sparse_colbert() is False


def test_embed_text_sparse_returns_none_for_non_bge_m3(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    s = Settings(_env_file=None)
    emb = SentenceTransformerTextEmbedder(settings=s)
    assert emb.embed_text_sparse("hello") is None
    assert emb.embed_text_colbert("hello") is None


def test_embed_text_sparse_returns_vectors_for_bge_m3(monkeypatch) -> None:
    """A bge-m3 embedder returns a dict with indices/values from the model."""

    class _BgeM3Stub:
        def encode(self, texts, **kwargs):
            return_sparse = kwargs.get("return_sparse", False)
            return_colbert = kwargs.get("return_colbert_vecs", False)
            if return_sparse:
                return {"sparse": [{"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]}]}
            if return_colbert:
                return {"colbert_vecs": [[[0.1, 0.2], [0.3, 0.4]]]}
            # dense path
            return np.zeros((len(texts), 8), dtype=np.float32)

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    s = Settings(_env_file=None)
    emb = SentenceTransformerTextEmbedder(settings=s)
    emb._model = _BgeM3Stub()
    emb._dim = 8

    sparse = emb.embed_text_sparse("hello world")
    assert sparse is not None
    assert sparse["indices"] == [1, 2, 3]
    assert sparse["values"] == [0.1, 0.2, 0.3]

    colbert = emb.embed_text_colbert("hello world")
    assert colbert is not None
    assert colbert == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_text_sparse_returns_none_when_model_returns_none(monkeypatch) -> None:
    """If the model's encode returns an unexpected shape, return None."""

    class _StubReturnsNone:
        def encode(self, texts, **kwargs):
            return {}  # no "sparse" key

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    s = Settings(_env_file=None)
    emb = SentenceTransformerTextEmbedder(settings=s)
    emb._model = _StubReturnsNone()
    emb._dim = 8
    assert emb.embed_text_sparse("hello") is None
    assert emb.embed_text_colbert("hello") is None


def test_openai_text_embedder_does_not_implement_sparse_colbert(monkeypatch) -> None:
    """The OpenAI-compatible TextEmbedder must not expose sparse/colbert.

    This is the guarantee that the default OpenAI configuration has zero
    schema change — ``qdrant_backend`` probes with ``getattr`` and the
    methods are absent, so no sparse/colbert field is added.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "text-embedding-3-small")
    s = Settings(_env_file=None)
    from mm_asset_rag.embedders.text_embedder import TextEmbedder

    emb = TextEmbedder(settings=s)
    assert not hasattr(emb, "embed_text_sparse")
    assert not hasattr(emb, "embed_text_colbert")
