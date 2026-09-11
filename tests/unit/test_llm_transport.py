"""Offline tests for the shared OpenAI-compatible LLM transport."""

from __future__ import annotations

import pytest
import requests

from mm_asset_rag.llm_transport import LlmRateLimiter, LlmTransportError, post_chat_completion


class _Response:
    def __init__(self, status: int, *, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


def test_limiter_spaces_request_starts_with_injected_clock() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = LlmRateLimiter(5, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    assert sleeps == [12.0]


def test_post_retries_429_retry_after_then_returns_response() -> None:
    calls: list[dict] = []
    sleeps: list[float] = []
    responses = [_Response(429, headers={"Retry-After": "3"}), _Response(200)]

    def post(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    result = post_chat_completion(
        "https://example.test/v1",
        "key",
        "model",
        [{"role": "user", "content": "hello"}],
        timeout=1,
        max_retries=1,
        retry_backoff_seconds=0,
        limiter=LlmRateLimiter(0),
        post=post,
        sleep=sleeps.append,
    )
    assert result.status_code == 200
    assert len(calls) == 2
    assert sleeps == [3.0]


def test_post_does_not_retry_401() -> None:
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(401)

    with pytest.raises(LlmTransportError):
        post_chat_completion(
            "https://example.test/v1",
            "key",
            "model",
            [],
            timeout=1,
            max_retries=2,
            limiter=LlmRateLimiter(0),
            post=post,
        )
    assert calls == 1


def test_post_retries_timeout_then_returns_response() -> None:
    responses: list[object] = [requests.Timeout("slow"), _Response(200)]

    def post(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    result = post_chat_completion(
        "https://example.test/v1",
        "key",
        "model",
        [],
        timeout=1,
        max_retries=1,
        retry_backoff_seconds=0,
        limiter=LlmRateLimiter(0),
        post=post,
        sleep=lambda _: None,
    )
    assert result.status_code == 200
