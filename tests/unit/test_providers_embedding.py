"""Tests for ``mm_asset_rag.embedders.text_embedder.TextEmbedder``.

External HTTP calls (OpenAI-compatible endpoints, including local Ollama)
are intercepted via ``responses`` so the tests stay offline at the
network level — but every code path uses a real embedder instance,
no mocks.
"""

from __future__ import annotations

import pytest
import responses

from mm_asset_rag.embedders.text_embedder import EmbeddingConfigError, TextEmbedder
from mm_asset_rag.settings import Settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Each test gets a fresh Settings singleton (no stale .env values)."""
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _isolated_settings(monkeypatch) -> Settings:
    """Strip embedding env vars and return a Settings with no .env fallback."""
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


def test_provider_requires_configuration(monkeypatch) -> None:
    s = _isolated_settings(monkeypatch)
    with pytest.raises(EmbeddingConfigError, match="requires"):
        TextEmbedder(settings=s)


def test_provider_alias_still_raises(monkeypatch) -> None:
    s = _isolated_settings(monkeypatch)
    with pytest.raises(EmbeddingConfigError, match="requires"):
        TextEmbedder(settings=s)


@responses.activate
def test_provider_embed_texts_success(monkeypatch) -> None:
    # ``Settings(_env_file=None)`` skips the on-disk .env but STILL reads
    # os.environ, so a host with EMBEDDING_BASE_URL set (e.g. local ollama)
    # would leak into ``creds[1]`` and the request would miss the mock.
    # Use the isolated-settings helper that strips the embedding env vars.
    s = _isolated_settings(monkeypatch)
    s.openai_api_key = "test-key"
    s.openai_base_url = "https://api.example.com/v1"
    s.openai_model = "text-embed-3-small"
    s.embedding_api_key = "test-key"
    s.embedding_base_url = "https://api.example.com/v1"
    s.embedding_model = "text-embed-3-small"

    responses.post(
        "https://api.example.com/v1/embeddings",
        json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
        status=200,
    )

    provider = TextEmbedder(settings=s)
    assert provider.embed_batch(["hello"]) == [[0.1, 0.2, 0.3]]


@responses.activate
def test_provider_embed_texts_retries_on_429(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_RETRY_COUNT", "3")
    monkeypatch.setenv("EMBEDDING_REQUEST_INTERVAL", "0")
    s = _isolated_settings(monkeypatch)
    s.openai_api_key = "test-key"
    s.openai_base_url = "https://api.example.com/v1"
    s.openai_model = "text-embed-3-small"
    s.embedding_api_key = "test-key"
    s.embedding_base_url = "https://api.example.com/v1"
    s.embedding_model = "text-embed-3-small"
    s.embedding_retry_count = 3
    s.embedding_request_interval = 0

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

    provider = TextEmbedder(settings=s)
    assert provider.embed_batch(["x"]) == [[0.5]]
    assert len(responses.calls) == 2


@responses.activate
def test_provider_honors_batch_size(monkeypatch) -> None:
    s = _isolated_settings(monkeypatch)
    s.openai_api_key = "test-key"
    s.openai_base_url = "https://api.example.com/v1"
    s.openai_model = "text-embed-3-small"
    s.embedding_api_key = "test-key"
    s.embedding_base_url = "https://api.example.com/v1"
    s.embedding_model = "text-embed-3-small"
    s.embedding_batch_size = 2
    s.embedding_request_interval = 0

    def _callback(request):
        body = request.body
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

    provider = TextEmbedder(settings=s)
    vectors = provider.embed_batch(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    # 5 inputs / batch_size 2 = 3 batches (2, 2, 1)
    assert len(responses.calls) == 3
