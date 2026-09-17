"""Capability-specific adapters for OpenAI-compatible remote APIs."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from mm_asset_rag.core import provider_security

from .openai_compatible import OpenAICompatibleConnection


class LlmTransportError(RuntimeError):
    """Raised when a chat request cannot be completed after allowed retries."""


class LlmRateLimiter:
    """Thread-safe minimum-interval limiter; zero disables pacing for tests."""

    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = self._clock()
            wait = max(0.0, self._next_start - now)
            if wait:
                self._sleep(wait)
                now = self._clock()
            self._next_start = max(now, self._next_start) + self._interval


def _retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed >= 0 else None


def _is_retryable_chat_error(exc: BaseException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or 500 <= exc.response.status_code < 600
    return False


class OpenAIChatAdapter:
    """Adapter for the OpenAI-compatible ``/chat/completions`` capability."""

    def __init__(
        self,
        *,
        connection: OpenAICompatibleConnection,
        model: str,
        timeout: float,
        max_retries: int,
        retry_backoff_seconds: float = 1.0,
        limiter: LlmRateLimiter | None = None,
        post: Callable[..., requests.Response] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.connection = connection
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.limiter = limiter or LlmRateLimiter(0)
        self._post = post
        self._sleep = sleep

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> requests.Response:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "stream": stream,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        provider_security.warn_insecure_base_url(self.connection.base_url)
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self.limiter.acquire()
                response = self._post(
                    self.connection.endpoint("chat/completions"),
                    headers={
                        "Authorization": f"Bearer {self.connection.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
                response.raise_for_status()
                return response
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable_chat_error(exc):
                    break
                delay = (
                    _retry_after(exc.response)
                    if isinstance(exc, requests.HTTPError) and exc.response
                    else None
                )
                if delay is None:
                    delay = self.retry_backoff_seconds * (2**attempt) + random.uniform(0.0, 0.1)
                if delay:
                    self._sleep(delay)
        raise LlmTransportError("chat completion failed after retry policy") from last_error


class OpenAIEmbeddingAdapter:
    """Adapter for the OpenAI-compatible ``/embeddings`` capability."""

    def __init__(
        self,
        *,
        connection: OpenAICompatibleConnection,
        model: str,
        timeout: float,
        retry_count: int,
        post: Callable[..., requests.Response] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.connection = connection
        self.model = model
        self.timeout = timeout
        self.retry_count = max(1, retry_count)
        self._post = post
        self._sleep = sleep

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                response = self._post(
                    self.connection.endpoint("embeddings"),
                    headers={
                        "Authorization": f"Bearer {self.connection.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": texts},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return [
                    [float(value) for value in item["embedding"]]
                    for item in sorted(
                        response.json()["data"], key=lambda item: item.get("index", 0)
                    )
                ]
            except Exception as exc:
                error = exc
                if attempt + 1 < self.retry_count:
                    self._sleep(min(2**attempt, 20))
        raise error or RuntimeError("Embedding request failed")
