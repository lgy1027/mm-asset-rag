"""Streaming support shared by the HTTP API's NDJSON endpoints."""

from __future__ import annotations

import asyncio
import queue
import re
import threading

from .settings import get_settings

_STREAM_ERR_MAX_CHARS = 240
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_HOST_QUOTED_RE = re.compile(r"\bhost='([^']+)'")
_HOSTPORT_RE = re.compile(r"\b([a-z0-9][a-z0-9.-]*):(\d{2,5})\b")


def _provider_hosts() -> set[str]:
    """Hosts parsed from the configured LLM/VLM base URLs."""
    from urllib.parse import urlparse

    hosts: set[str] = set()
    try:
        s = get_settings()
        for base in (s.openai_base_url, s.vlm_base_url, s.embedding_base_url):
            if base:
                host = urlparse(base).hostname
                if host:
                    hosts.add(host)
    except Exception:
        pass
    return hosts


def _safe_stream_error(exc: BaseException) -> str:
    """Render ``exc`` for a streamed error event without leaking URLs/hosts."""
    msg = str(exc)
    msg = _URL_RE.sub("<url>", msg)
    msg = _HOST_QUOTED_RE.sub("host=<host>", msg)
    msg = _HOSTPORT_RE.sub("<host>:<port>", msg)
    for host in _provider_hosts():
        if host:
            msg = msg.replace(host, "<host>")
    first_line = msg.splitlines()[0] if msg else ""
    first_line = "".join(c for c in first_line if c >= " " or c == "\t")
    if len(first_line) > _STREAM_ERR_MAX_CHARS:
        first_line = first_line[:_STREAM_ERR_MAX_CHARS] + "…"
    return first_line or f"{type(exc).__name__}: <no message>"


_STREAM_DONE: object = object()


async def _iter_sync_in_thread(factory, *args, **kwargs) -> asyncio.Queue:
    """Bridge a sync generator into the event loop through a bounded queue."""
    out: queue.Queue = queue.Queue(maxsize=64)
    stop = threading.Event()

    def _put_terminal(item) -> None:
        while True:
            try:
                out.put(item, timeout=0.1)
                return
            except queue.Full:
                # An active consumer will eventually free a slot, so keep
                # retrying and preserve terminal ordering. On disconnect the
                # consumer sets ``stop``; abandon after this bounded attempt
                # instead of leaking a producer thread behind a full queue.
                if stop.is_set():
                    return

    def _worker():
        try:
            for item in factory(*args, **kwargs):
                if stop.is_set():
                    return
                while not stop.is_set():
                    try:
                        out.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    return
        except BaseException as exc:
            _put_terminal(exc)
        finally:
            _put_terminal(_STREAM_DONE)

    thread = threading.Thread(target=_worker, daemon=True, name="chat-stream-producer")
    thread.start()
    out.stop = stop  # type: ignore[attr-defined]
    out.thread = thread  # type: ignore[attr-defined]
    return out
