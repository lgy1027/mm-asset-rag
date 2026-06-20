"""Tests for mm_asset_rag.providers.

External HTTP calls (OpenAI-compatible endpoints, including local Ollama)
are intercepted via ``responses`` so the tests stay offline at the
network level — but every code path uses a real provider instance,
no mocks.
"""

from __future__ import annotations

import pytest
import responses

from mm_asset_rag.providers import (
    EmbeddingConfigError,
    EmbeddingProvider,
    configure_embedding,
)


def test_provider_requires_configuration(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(EmbeddingConfigError, match="requires"):
        EmbeddingProvider()


def test_configure_embedding_requires_configuration(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(EmbeddingConfigError, match="requires"):
        configure_embedding()


@responses.activate
def test_provider_embed_texts_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "text-embed-3-small")

    responses.post(
        "https://api.example.com/v1/embeddings",
        json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
        status=200,
    )

    provider = EmbeddingProvider()
    assert provider.embed_texts(["hello"]) == [[0.1, 0.2, 0.3]]


@responses.activate
def test_provider_embed_texts_retries_on_429(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "text-embed-3-small")
    monkeypatch.setenv("EMBEDDING_RETRY_COUNT", "3")
    monkeypatch.setenv("EMBEDDING_REQUEST_INTERVAL", "0")

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


@responses.activate
def test_provider_honors_batch_size(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "text-embed-3-small")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")
    monkeypatch.setenv("EMBEDDING_REQUEST_INTERVAL", "0")

    def _callback(request):
        body = request.body
        # Extract the "input" list length from the JSON body
        import json

        n_inputs = len(json.loads(body)["input"])
        return (
            200,
            {},
            json.dumps(
                {
                    "data": [
                        {"index": i, "embedding": [float(i), float(n_inputs)]}
                        for i in range(n_inputs)
                    ]
                }
            ),
        )

    responses.add_callback(
        responses.POST,
        "https://api.example.com/v1/embeddings",
        callback=_callback,
    )

    provider = EmbeddingProvider()
    vectors = provider.embed_texts(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    # 5 inputs / batch_size 2 = 3 batches (2, 2, 1)
    assert len(responses.calls) == 3
