"""Authentication and trusted-host helpers for the HTTP API."""

from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from .settings import get_settings

_DEFAULT_TRUSTED_HOSTS = ["127.0.0.1", "localhost", "[::1]"]


def _resolve_trusted_hosts() -> list[str]:
    raw = get_settings().mmrag_trusted_hosts
    if raw is None or raw.strip() == "":
        return _DEFAULT_TRUSTED_HOSTS
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    return hosts or _DEFAULT_TRUSTED_HOSTS


_TOKEN_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_BEARER_HEADER = APIKeyHeader(name="Authorization", auto_error=False, scheme_name="bearer")


def _extract_bearer(value: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer <t>`` header."""
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def require_token(
    x_api_key: str | None = Depends(_TOKEN_HEADER),
    authorization: str | None = Depends(_BEARER_HEADER),
) -> None:
    """Reject requests missing the configured API token."""
    expected = get_settings().mmrag_api_token
    if not expected or not expected.strip():
        return
    provided = x_api_key or _extract_bearer(authorization)
    if not provided or not _const_time_eq(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API token (set Authorization: Bearer <token> or X-API-Key)",
            headers={"WWW-Authenticate": 'Bearer realm="mmrag"'},
        )


def _const_time_eq(a: str, b: str) -> bool:
    """Constant-time string compare to avoid a token timing oracle."""
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())
