"""Shared connection primitives for remote OpenAI-compatible providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAICompatibleConnection:
    """A resolved remote provider connection shared by model capabilities."""

    base_url: str
    api_key: str

    def endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


def require_connection(base_url: str | None, api_key: str | None) -> OpenAICompatibleConnection:
    """Return a validated connection instead of permitting implicit fallbacks."""
    if not base_url or not api_key:
        raise ValueError("OpenAI-compatible base URL and API key are required")
    return OpenAICompatibleConnection(base_url=base_url, api_key=api_key)
