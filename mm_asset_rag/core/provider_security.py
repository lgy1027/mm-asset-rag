"""Security warnings shared by outbound provider clients."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# base_url values already warned about in this process — avoids log spam
# when many chunks / captions hit the same insecure endpoint.
_warned_insecure_base_urls: set[str] = set()

# Hosts considered loopback — http:// to these is fine (local ollama, etc.).
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def warn_insecure_base_url(base_url: str) -> None:
    """Warn once when ``base_url`` is plain HTTP to a non-loopback host.

    Only warns — never raises or blocks the request — so existing local
    http://ollama deployments keep working. A non-loopback http:// endpoint
    would send ``Authorization: Bearer <api_key>`` in cleartext over the
    network; we surface that risk once per base_url so deployments can move
    to HTTPS without breaking callers that intentionally use plain HTTP.
    """
    if not base_url:
        return
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return
    if parsed.scheme != "http":
        return
    host = (parsed.hostname or "").lower()
    if not host or host in _LOOPBACK_HOSTS:
        return
    if base_url in _warned_insecure_base_urls:
        return
    _warned_insecure_base_urls.add(base_url)
    log.warning(
        "LLM/VLM base_url %s 使用明文 HTTP 且非本机回环地址,"
        " Authorization Bearer api_key 将以明文 over HTTP 传输,"
        " 建议改用 HTTPS 或保持本机 (127.0.0.1/localhost/::1) 部署。",
        base_url,
    )
