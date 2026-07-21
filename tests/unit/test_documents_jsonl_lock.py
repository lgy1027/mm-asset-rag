"""Tests for ``documents_jsonl_lock`` (cross-process documents.jsonl guard).

The lock prevents the data-loss race where a rewrite
(``_remove_asset_rows``: read → tmp → ``os.replace``) swaps the file out
while an appender still holds the old fd — the appender would keep writing
to the now-unlinked inode and those chunk rows would vanish (recovered
only by a later reindex). We exercise the lock directly: a concurrent
append + rewrite must leave the appended rows in the *new* file.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from mm_asset_rag.document_store import documents_jsonl_lock
from mm_asset_rag.paths import get_documents_jsonl


def _append_row(path: Path, asset_id: str) -> None:
    with documents_jsonl_lock(path), path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "x", "metadata": {"asset_id": asset_id}}) + "\n")


def _remove_rows(path: Path, asset_ids: set[str]) -> int:
    """Mirror ``_remove_asset_rows_from_documents_jsonl`` under the lock.

    ``os.replace`` is inside the lock — matching the production fix.
    Without it (replace outside the lock), a concurrent appender could
    grab the lock right after the rewrite releases it, open("a") the old
    inode, and we'd os.replace-swap the file out from under it, losing
    its writes to the unlinked inode.
    """
    removed = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with documents_jsonl_lock(path):
        src = path.open("r", encoding="utf-8")
        try:
            dst = tmp_path.open("w", encoding="utf-8")
            try:
                for line in src:
                    stripped = line.strip()
                    if not stripped:
                        dst.write(line)
                        continue
                    obj = json.loads(stripped)
                    if str(obj.get("metadata", {}).get("asset_id", "")) in asset_ids:
                        removed += 1
                        continue
                    dst.write(line)
            finally:
                dst.close()
        finally:
            src.close()
        os.replace(tmp_path, path)
    return removed


def test_lock_serialises_concurrent_append_and_rewrite(tmp_home: Path) -> None:
    """A rewrite (read → tmp → os.replace) concurrent with an append must
    not drop the appended row into the swapped-out inode.

    Deterministic timing: the appender starts by acquiring the lock
    *first* and holds it; the rewriter then blocks on the lock until the
    appender releases. The appender appends, releases; the rewriter
    acquires, does read→tmp→replace under the lock. Both writes land in
    the final file. Regression: if os.replace moves outside the lock,
    the rewriter's replace would swap the file out *after* releasing the
    lock — but here the appender has already finished (no concurrent fd),
    so this test alone can't catch replace-outside-lock. That case is
    covered by ``test_lock_blocks_second_acquire_until_released`` (a
    no-op lock would let both in simultaneously and lose data).
    """
    docs = get_documents_jsonl()
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(
        json.dumps({"text": "keep", "metadata": {"asset_id": "keep"}}) + "\n",
        encoding="utf-8",
    )

    errors: list[BaseException] = []
    appender_done = threading.Event()

    def appender():
        try:
            with documents_jsonl_lock(docs), docs.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"text": "x", "metadata": {"asset_id": "appended"}}) + "\n")
        except BaseException as exc:
            errors.append(exc)
        finally:
            appender_done.set()

    def rewriter():
        try:
            # Wait for the appender to finish so the rewriter's replace
            # runs strictly after; under the lock both are serialised.
            appender_done.wait(timeout=5)
            _remove_rows(docs, {"nonexistent"})
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=appender)
    t2 = threading.Thread(target=rewriter)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    rows = [json.loads(ln) for ln in docs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    asset_ids = [r["metadata"]["asset_id"] for r in rows]
    assert "keep" in asset_ids
    assert "appended" in asset_ids, f"appended row lost in rewrite race: {asset_ids}"


def test_lock_blocks_second_acquire_until_released(tmp_home: Path) -> None:
    """A second lock acquisition must block until the first releases — the
    hard guarantee the os.replace race fix relies on. If this ever regresses
    (lock becomes a no-op), the append-during-rewrite race silently returns:
    both writers enter the critical section at once, and the rewriter's
    os.replace swaps the file out while the appender's fd still points at
    the old inode → appends vanish."""
    docs = get_documents_jsonl()
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("", encoding="utf-8")

    first_acquired = threading.Event()
    second_got_lock = threading.Event()
    release_first = threading.Event()
    timing: list[str] = []

    def first():
        with documents_jsonl_lock(docs):
            first_acquired.set()
            # Hold until the second thread has (attempted to) acquire.
            release_first.wait(timeout=5)

    def second():
        first_acquired.wait(timeout=5)
        timing.append("second_started")
        with documents_jsonl_lock(docs):
            timing.append("second_got_lock")
            second_got_lock.set()

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    # The second thread must NOT get the lock while the first holds it.
    assert not second_got_lock.wait(timeout=1.0), "second acquired lock while first held it"
    release_first.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert "second_got_lock" in timing, f"second never acquired after release: {timing}"
