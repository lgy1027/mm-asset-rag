"""Tests for ``mm_asset_rag.backends.qdrant_backend`` lock handling.

The qdrant local-mode ``.lock`` file is process-local, so two
``mm-asset-rag`` processes can't open the same storage at once.
``_clean_stale_lock`` distinguishes dead locks (from a previous crashed
session) from live locks (another process is still running) so the
caller can either unlink or raise a clear error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.backends.qdrant_backend import (
    QdrantLockHeldError,
    _clean_stale_lock,
    _lock_holder_pid,
    _pid_alive,
    _probe_lock_holder,
)

# ─── Unit tests for the helper primitives ───────────────────────────────


def test_pid_alive_returns_true_for_self():
    import os

    assert _pid_alive(os.getpid()) is True


def test_pid_alive_returns_false_for_nonexistent_pid():
    # A PID like 2^31 is almost certainly not in use.
    assert _pid_alive(2_147_483_647) is False


def test_lock_holder_pid_for_nonexistent_file(tmp_path: Path):
    missing = tmp_path / "does-not-exist.lock"
    assert _lock_holder_pid(missing) is None


def test_lock_holder_pid_for_unlocked_file(tmp_path: Path):
    """A file with no process holding it returns None (or empty from lsof)."""
    unheld = tmp_path / "free.lock"
    unheld.write_text("")
    # lsof may return either None (no pids) or empty list. Both are "safe".
    holder = _lock_holder_pid(unheld)
    assert holder is None or isinstance(holder, int)


# ─── Integration tests for _clean_stale_lock ────────────────────────────


def test_clean_stale_lock_no_op_when_lock_absent(tmp_path: Path):
    """No .lock present → no error, no action."""
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    # Should not raise.
    _clean_stale_lock(qdrant_path)
    assert not (qdrant_path / ".lock").exists()


def test_clean_stale_lock_removes_dead_lock(tmp_path: Path):
    """A .lock from a process that's no longer alive gets unlinked."""
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    lock = qdrant_path / ".lock"
    # Touch a fake lock file. ``lsof`` should return no pids (since the
    # process that created it never existed), so the cleanup proceeds.
    lock.write_text("stale")
    _clean_stale_lock(qdrant_path)
    assert not lock.exists()


def test_clean_stale_lock_raises_when_held_by_live_process(tmp_path: Path):
    """A .lock held by an open file descriptor of *this* process trips the guard.

    ``_lock_holder_pid`` uses ``lsof``, which sees the current process when
    the lock file is still open. ``_pid_alive`` is true for our own PID, so
    ``_clean_stale_lock`` should raise ``QdrantLockHeldError``.
    """
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    lock = qdrant_path / ".lock"
    # Keep the file open in this process so ``lsof`` sees our PID holding
    # the descriptor. Closing it inside the test would defeat the check.
    fp = open(lock, "w")  # noqa: SIM115
    try:
        with pytest.raises(QdrantLockHeldError) as excinfo:
            _clean_stale_lock(qdrant_path)
        # The error message should mention the holder PID and a hint about
        # stopping the process / switching to server mode.
        assert str(excinfo.value).startswith("Qdrant local storage at")
        assert "Stop that process first" in str(excinfo.value)
    finally:
        fp.close()
        # After the test closes the fd, _clean_stale_lock should succeed.
        if lock.exists():
            _clean_stale_lock(qdrant_path)


def test_clean_stale_lock_raises_when_holder_unknown(tmp_path: Path, monkeypatch):
    """When lsof cannot answer (missing / failed) the lock is NOT removed.

    Blindly unlinking a lock whose holder is unknown risks deleting a lock
    a live process still holds, letting two processes write the same local
    storage. The guard must raise instead of guessing.
    """
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    lock = qdrant_path / ".lock"
    lock.write_text("unknown")

    def _unknown_probe(_lock):
        return ("unknown", None)

    monkeypatch.setattr("mm_asset_rag.backends.qdrant_backend._probe_lock_holder", _unknown_probe)
    with pytest.raises(QdrantLockHeldError) as excinfo:
        _clean_stale_lock(qdrant_path)
    assert "could not be determined" in str(excinfo.value)
    # Crucially, the lock file is left in place.
    assert lock.exists()


def test_clean_stale_lock_removes_when_lsof_confirms_free(tmp_path: Path, monkeypatch):
    """A "free" probe (lsof ran, no holder) is unlinked, same as before."""
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    lock = qdrant_path / ".lock"
    lock.write_text("stale")

    def _free_probe(_lock):
        return ("free", None)

    monkeypatch.setattr("mm_asset_rag.backends.qdrant_backend._probe_lock_holder", _free_probe)
    _clean_stale_lock(qdrant_path)
    assert not lock.exists()


def test_probe_lock_holder_states(tmp_path: Path):
    """The three-state probe returns ("free", None) for an unheld lock."""
    unheld = tmp_path / "free.lock"
    unheld.write_text("")
    state, pid = _probe_lock_holder(unheld)
    # On a system with lsof this is "free"; on one without it is "unknown".
    assert state in ("free", "unknown")
    assert pid is None
