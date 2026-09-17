"""Behavioral tests for OpenAI-compatible capability adapters."""

from __future__ import annotations

import requests

from mm_asset_rag.core.llm_transport import LlmRateLimiter
from mm_asset_rag.core.openai_compatible import OpenAICompatibleConnection


class _Response:
    def __init__(self, status: int, body: dict | None = None) -> None:
        self.status_code = status
        self._body = body or {}
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> dict:
        return self._body


def test_chat_adapter_sends_chat_completion_to_its_capability_endpoint() -> None:
    from mm_asset_rag.core.openai_adapters import OpenAIChatAdapter

    calls: list[tuple[str, dict]] = []

    def post(url: str, **kwargs):
        calls.append((url, kwargs))
        return _Response(200)

    adapter = OpenAIChatAdapter(
        connection=OpenAICompatibleConnection("https://api.example.test/v1", "secret"),
        model="chat-model",
        timeout=7,
        max_retries=0,
        limiter=LlmRateLimiter(0),
        post=post,
    )

    response = adapter.complete([{"role": "user", "content": "hello"}], max_tokens=42)

    assert response.status_code == 200
    assert calls == [
        (
            "https://api.example.test/v1/chat/completions",
            {
                "headers": {
                    "Authorization": "Bearer secret",
                    "Content-Type": "application/json",
                },
                "json": {
                    "model": "chat-model",
                    "temperature": 0.1,
                    "stream": False,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 42,
                },
                "timeout": 7,
                "stream": False,
            },
        )
    ]


def test_embedding_adapter_sends_inputs_to_embeddings_endpoint_in_response_order() -> None:
    from mm_asset_rag.core.openai_adapters import OpenAIEmbeddingAdapter

    calls: list[tuple[str, dict]] = []

    def post(url: str, **kwargs):
        calls.append((url, kwargs))
        return _Response(
            200,
            {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]},
        )

    adapter = OpenAIEmbeddingAdapter(
        connection=OpenAICompatibleConnection("https://api.example.test/v1", "secret"),
        model="embedding-model",
        timeout=7,
        retry_count=1,
        post=post,
    )

    assert adapter.embed_batch(["first", "second"]) == [[1.0], [2.0]]
    assert calls == [
        (
            "https://api.example.test/v1/embeddings",
            {
                "headers": {
                    "Authorization": "Bearer secret",
                    "Content-Type": "application/json",
                },
                "json": {"model": "embedding-model", "input": ["first", "second"]},
                "timeout": 7,
            },
        )
    ]
