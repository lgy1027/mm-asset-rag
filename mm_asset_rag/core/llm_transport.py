"""Paced, retry-safe transport for OpenAI-compatible chat completions."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from .observability import get_tracer
from .openai_adapters import LlmRateLimiter, LlmTransportError, OpenAIChatAdapter
from .openai_compatible import require_connection
from .settings import get_settings

__all__ = ["LlmRateLimiter", "LlmTransportError", "get_llm_rate_limiter", "post_chat_completion"]


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
    adapter = OpenAIChatAdapter(
        connection=require_connection(base_url, api_key),
        model=model,
        timeout=timeout,
        max_retries=retries,
        retry_backoff_seconds=backoff,
        limiter=limiter or get_llm_rate_limiter(),
        post=post,
        sleep=sleep,
    )
    tracer = get_tracer()
    with tracer.start_generation(
        "llm.chat_completion",
        model=model,
        input=messages,
        model_parameters={
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        metadata={
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": base_url,
        },
    ) as span:
        response = adapter.complete(
            messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            span.set_attribute("status_code", status_code)
        if not stream:
            span.update(usage=_extract_usage(response), output=_extract_output(response))
        return response


def _extract_output(response: requests.Response, *, max_chars: int = 2000) -> str | None:
    """First completion choice text from a non-streaming body, for tracing.

    Truncated to ``max_chars`` — full transcripts live in the application,
    telemetry only needs enough to audit what the model was asked and
    answered. ``None`` when unavailable; never raises.
    """
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    text = str(content)
    return text[:max_chars]


def _extract_usage(response: requests.Response) -> dict[str, int] | None:
    """Pull ``usage`` token counts out of a non-streaming completion body.

    Returns Langfuse-style ``{"input": n, "output": n, "total": n}`` keys, or
    ``None`` when the provider sent no usage block (or the body is not JSON).
    Never raises — usage is telemetry, not transport.
    """
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    # ``in`` checks, not ``or``: a legit 0 token count must not fall through
    # to the alternate key name and get dropped from the report.
    prompt = usage["prompt_tokens"] if "prompt_tokens" in usage else usage.get("input_tokens")
    completion = (
        usage["completion_tokens"] if "completion_tokens" in usage else usage.get("output_tokens")
    )
    total = usage.get("total_tokens")
    out: dict[str, int] = {}
    if isinstance(prompt, int):
        out["input"] = prompt
    if isinstance(completion, int):
        out["output"] = completion
    if isinstance(total, int):
        out["total"] = total
    return out or None
