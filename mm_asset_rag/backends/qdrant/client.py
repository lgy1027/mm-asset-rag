"""Qdrant client lifecycle, caching, and local-file lock handling."""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
from pathlib import Path

from qdrant_client import QdrantClient

from ...paths import get_indexes_dir
from ...settings import get_settings


class QdrantLockHeldError(RuntimeError):
    """Raised when Qdrant local storage is already open by another live process.

    qdrant-client's local mode uses a process-local file lock at
    ``<indexes>/qdrant/.lock`` and refuses to open the same storage from a
    second process. The previous version of ``_clean_stale_lock``
    silently deleted the lock in all cases, which caused ``mmrag reindex``
    to hang when the API server (``uvicorn``) was still running.
    """


# Process-wide shared QdrantClient (local-file mode only). See
# ``get_qdrant_client`` for the rationale. Tests can call
# ``reset_qdrant_client_cache()`` to drop the cached instance between
# cases.
_QDRANT_CLIENT: QdrantClient | None = None
_QDRANT_CLIENT_KEY: str | None = None
_QDRANT_CLIENT_LOCK = threading.Lock()


def reset_qdrant_client_cache() -> None:
    """Drop the cached local QdrantClient. Test-only helper."""
    global _QDRANT_CLIENT, _QDRANT_CLIENT_KEY
    with _QDRANT_CLIENT_LOCK:
        if _QDRANT_CLIENT is not None:
            with contextlib.suppress(Exception):
                _QDRANT_CLIENT.close()
            _remove_owned_lock(_QDRANT_CLIENT_KEY)
        _QDRANT_CLIENT = None
        _QDRANT_CLIENT_KEY = None


def get_qdrant_client() -> QdrantClient:
    """Return a process-wide shared ``QdrantClient`` instance.

    Qdrant's local-file mode writes ``<storage>/.lock`` on open and
    refuses a second open from another instance. Without this cache
    each concurrent worker thread would instantiate its own client
    and they would race on the lock (or fail with
    ``Storage folder already accessed``). The cache is keyed by the
    storage location so the same path always returns the same client
    but switching to ``QDRANT_URL`` (server mode) does not share state
    with a stale local client.
    """
    global _QDRANT_CLIENT, _QDRANT_CLIENT_KEY
    settings = get_settings()
    if settings.qdrant_url:
        # Remote mode: each call returns its own client. Qdrant
        # server handles concurrency; the in-process cache would
        # just hold a connection alive longer than necessary.
        # If we previously cached a *local* client (deployer flipped
        # ``QDRANT_URL`` on at runtime), close it so its local-file
        # ``.lock`` and the underlying storage fd are released —
        # otherwise the lock stays held for the rest of the process
        # even though we no longer use that client.
        with _QDRANT_CLIENT_LOCK:
            if _QDRANT_CLIENT is not None:
                with contextlib.suppress(Exception):
                    _QDRANT_CLIENT.close()
                _remove_owned_lock(_QDRANT_CLIENT_KEY)
                _QDRANT_CLIENT = None
                _QDRANT_CLIENT_KEY = None
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            # httpx's default is 5s; a busy Qdrant server can take
            # 7s+ to clean up a non-empty collection (drain the
            # optimizer, drop snapshots, release segments) and a
            # ``client.delete_collection`` that times out client-side
            # leaves the collection live on the server while the
            # caller raises — so the next reindex step fails to
            # recreate it. 30s is well above observed cleanup time
            # without making healthy calls hang.
            timeout=30,
        )
    qdrant_path = get_indexes_dir() / "qdrant"
    key = str(qdrant_path)
    with _QDRANT_CLIENT_LOCK:
        if key != _QDRANT_CLIENT_KEY or _QDRANT_CLIENT is None:
            qdrant_path.mkdir(parents=True, exist_ok=True)
            _clean_stale_lock(qdrant_path)
            _QDRANT_CLIENT = QdrantClient(path=key)
            _QDRANT_CLIENT_KEY = key
        return _QDRANT_CLIENT


def _remove_owned_lock(client_key: str | None) -> None:
    """Remove a local lock after this process closes its cached client."""
    if not client_key:
        return
    lock = Path(client_key) / ".lock"
    with contextlib.suppress(OSError):
        lock.unlink()


def _clean_stale_lock(qdrant_path: Path) -> None:
    """Remove a stale ``.lock`` from a previous crashed session, but only
    when the lock is *not* held by a live process.

    qdrant-client's local mode writes ``.lock`` on open and removes it on
    ``close()``. If the process is killed before close() runs (SIGKILL, OOM,
    abrupt interpreter exit), the .lock is left behind and the next startup
    fails with ``Storage folder X is already accessed by another instance of
    Qdrant client``.

    If the lock is held by a *live* process (e.g. an ``uvicorn`` API server
    is still running), we refuse to remove it — qdrant-client in the second
    process would otherwise hang on the lock. Instead we raise
    :class:`QdrantLockHeldError` so the caller (e.g. ``mmrag reindex``) can
    surface a clear "stop the API server first" message.

    If the holder process **cannot be determined** (``lsof`` missing, timed
    out, or returned non-zero), we also raise rather than guess — silently
    unlinking a lock held by a live process would let two processes write
    the same local storage and corrupt the index. Resolve by stopping other
    processes, switching to ``QDRANT_URL`` (server mode), or removing the
    lock by hand once certain no process holds it.

    Safe for single-process use; switch to ``QDRANT_URL`` (server mode) for
    concurrent access.
    """
    lock = qdrant_path / ".lock"
    if not lock.exists():
        return
    state, holder_pid = _probe_lock_holder(lock)
    if state == "unknown":
        # lsof could not answer (missing / timed out / errored). Deleting the
        # lock here would be unsafe: another live process (e.g. an API server)
        # may be holding it, and blindly unlinking would let a second process
        # open the same local storage concurrently and corrupt the index.
        # Surface the uncertainty and let the user resolve it (stop other
        # processes, switch to QDRANT_URL, or remove the lock manually once
        # certain nothing holds it).
        raise QdrantLockHeldError(
            f"Qdrant local storage at {qdrant_path} has a .lock but the holder "
            f"process could not be determined (lsof missing or failed). Stop "
            f"any process that may hold it (or set QDRANT_URL to use server "
            f"mode), then retry. To override, remove {lock} manually once you "
            f"are certain no process is using it."
        )
    if state == "held" and holder_pid is not None and _pid_alive(holder_pid):
        raise QdrantLockHeldError(
            f"Qdrant local storage at {qdrant_path} is already open by "
            f"process {holder_pid} (probably the API server / another CLI). "
            f"Stop that process first, or set QDRANT_URL to use Qdrant server mode."
        )
    # state == "free" (lsof confirmed no holder), or "held" but the holder
    # PID is no longer alive — both are safe to unlink.
    try:
        lock.unlink()
        print(f"[qdrant] removed stale .lock from previous session: {lock.name}")
    except OSError as exc:
        print(f"[qdrant] warning: could not unlink {lock}: {exc}")


def _probe_lock_holder(lock: Path) -> tuple[str, int | None]:
    """Probe who holds ``lock``.

    Returns one of three explicit states so the caller never has to guess
    what a ``None`` pid means:

    * ``("held", pid)``    — lsof named a live holder; ``pid`` is its PID.
    * ``("free", None)``   — lsof ran to completion but found no process
      holding the lock, i.e. this is a genuinely stale lock from a crashed
      session; safe to unlink.
    * ``("unknown", None)``— lsof is missing, timed out, or errored before
      it could answer; the holder cannot be determined, so the caller must
      **not** unlink (a live process may still hold it).

    The distinction between ``"free"`` and ``"unknown"`` rests on lsof's
    exit code: ``returncode == 0`` or a "no match" non-zero exit (1 on both
    Linux and macOS when nothing holds the file) means lsof answered. Only
    the ``FileNotFoundError`` / timeout / other ``OSError`` paths are
    ``"unknown"``.
    """
    try:
        result = subprocess.run(
            ["lsof", "-F", "p", str(lock)],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ("unknown", None)
    # lsof exits 0 when it lists open files, and non-zero (commonly 1) when
    # nothing matches the path. Both mean lsof *ran* and answered — a
    # returncode != 0 with no "p" line is "free", not "unknown".
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            try:
                return ("held", int(line[1:]))
            except ValueError:
                continue
    return ("free", None)


def _lock_holder_pid(lock: Path) -> int | None:
    """Return the PID holding ``lock``, or None if it can't be determined.

    Thin compatibility shim over :func:`_probe_lock_holder`. Returns the
    holder PID for ``"held"``, and ``None`` for both ``"free"`` and
    ``"unknown"`` — callers that need to distinguish those two (e.g.
    :func:`_clean_stale_lock`) should use :func:`_probe_lock_holder`
    instead, since deleting on ``"unknown"`` is unsafe.
    """
    state, pid = _probe_lock_holder(lock)
    if state == "held":
        return pid
    return None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with this PID is running on this system."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
