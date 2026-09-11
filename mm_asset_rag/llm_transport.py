"""Paced, retry-safe transport for OpenAI-compatible chat completions."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from . import provider_security
from .settings import get_settings


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


_limiter_lock = threading.Lock()
_limiter: LlmRateLimiter | None = None
_limiter_rate: int | None = None


def get_llm_rate_limiter() -> LlmRateLimiter:
    """Return the process-wide limiter for the current typed settings."""
    global _limiter, _limiter_rate
    rate = max(0, int(get_settings().llm_requests_per_minute))
    with _limiter_lock:
        if _limiter is None or _limiter_rate != rate:
            _limiter = LlmRateLimiter(rate)
            _limiter_rate = rate
        return _limiter


def _retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed >= 0 else None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or 500 <= exc.response.status_code < 600
    return False


def post_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: float,
    stream: bool = False,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
    limiter: LlmRateLimiter | None = None,
    post: Callable[..., requests.Response] = requests.post,
    sleep: Callable[[float], None] = time.sleep,
) -> requests.Response:
    """POST one completion, pacing every attempt and retrying transient failures."""
    settings = get_settings()
    retries = settings.llm_max_retries if max_retries is None else max_retries
    backoff = (
        settings.llm_retry_backoff_seconds
        if retry_backoff_seconds is None
        else retry_backoff_seconds
    )
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "stream": stream,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    provider_security.warn_insecure_base_url(base_url)
    active_limiter = limiter or get_llm_rate_limiter()
    last_error: BaseException | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            active_limiter.acquire()
            response = post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
                stream=stream,
            )
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_error = exc
            if attempt >= max(0, retries) or not _is_retryable(exc):
                break
            delay = (
                _retry_after(exc.response)
                if isinstance(exc, requests.HTTPError) and exc.response
                else None
            )
            if delay is None:
                delay = max(0.0, float(backoff)) * (2**attempt) + random.uniform(0.0, 0.1)
            if delay:
                sleep(delay)
    raise LlmTransportError("chat completion failed after retry policy") from last_error
