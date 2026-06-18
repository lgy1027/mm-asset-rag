"""Tests for mm_asset_rag.providers.

External HTTP calls (OpenAI) are intercepted via ``responses`` so these
tests stay offline.
"""

from __future__ import annotations

import math

import responses

from mm_asset_rag.providers import EmbeddingProvider


def test_mock_embedding_deterministic() -> None:
    monkeypatch_env = {
        "EMBEDDING_PROVIDER": "mock",
        "MOCK_EMBEDDING_DIM": "16",
    }
    for key, value in monkeypatch_env.items():
        import os

        os.environ[key] = value

    provider = EmbeddingProvider()
    v1 = provider.embed_text("hello world")
    v2 = provider.embed_text("hello world")
    assert v1 == v2
    assert len(v1) == 16
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-6


@responses.activate
def test_openai_compatible_embedding_success() -> None:
    import os

    os.environ.update(
        {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "test-key",
            "EMBEDDING_BASE_URL": "https://api.example.com/v1",
            "EMBEDDING_MODEL": "text-embed-3-small",
        }
    )

    responses.post(
        "https://api.example.com/v1/embeddings",
        json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
        status=200,
    )

    provider = EmbeddingProvider()
    vectors = provider.embed_texts(["hello"])
    assert vectors == [[0.1, 0.2, 0.3]]


@responses.activate
def test_embedding_request_retries_on_429() -> None:
    import os

    os.environ.update(
        {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "test-key",
            "EMBEDDING_BASE_URL": "https://api.example.com/v1",
            "EMBEDDING_MODEL": "text-embed-3-small",
            "EMBEDDING_RETRY_COUNT": "3",
            "EMBEDDING_REQUEST_INTERVAL": "0",
        }
    )

    responses.post(
        "https://api.example.com/v1/embeddings",
        status=429,
        headers={"Retry-After": "0"},
    )
    responses.post(
        "https://api.example.com/v1/embeddings",
        json={"data": [{"index": 0, "embedding": [0.5]}]},
        status=200,
    )

    provider = EmbeddingProvider()
    assert provider.embed_texts(["x"]) == [[0.5]]
    assert len(responses.calls) == 2


def test_configure_embedding_falls_back_to_mock(monkeypatch) -> None:
    from mm_asset_rag import providers

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")

    name = providers.configure_embedding()
    assert "MockEmbedding" in name
