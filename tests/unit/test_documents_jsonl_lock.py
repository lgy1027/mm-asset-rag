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
import threading
import time
from pathlib import Path

from mm_asset_rag.document_store import documents_jsonl_lock
from mm_asset_rag.paths import get_documents_jsonl


def _append_row(path: Path, asset_id: str) -> None:
    with documents_jsonl_lock(path), path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "x", "metadata": {"asset_id": asset_id}}) + "\n")


def _remove_rows(path: Path, asset_ids: set[str]) -> int:
    """Mirror ``_remove_asset_rows_from_documents_jsonl`` under the lock."""
    removed = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with documents_jsonl_lock(path):
        src = path.open("r", encoding="utf-8")
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
            src.close()
            dst.close()
    import os

    os.replace(tmp_path, path)
    return removed


def test_lock_serialises_concurrent_append_and_rewrite(tmp_home: Path) -> None:
    """A rewrite (os.replace) concurrent with an append must not drop the
    appended row into the swapped-out inode. Before the lock, the append
    vanished; after, the lock serialises the two so the rewrite either
    sees the append (it lands in the tmp → new file) or runs before it
    (the append goes to the new file). Either way the row survives."""
    docs = get_documents_jsonl()
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(
        json.dumps({"text": "keep", "metadata": {"asset_id": "keep"}}) + "\n",
        encoding="utf-8",
    )

    appended = threading.Event()
    errors: list[BaseException] = []

    def appender():
        try:
            # Stall slightly so the rewrite and append actually overlap.
            time.sleep(0.02)
            _append_row(docs, "appended")
        except BaseException as exc:
            errors.append(exc)
        finally:
            appended.set()

    def rewriter():
        try:
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
    # The appended row must survive — not lost to the os.replace race.
    rows = [json.loads(ln) for ln in docs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    asset_ids = [r["metadata"]["asset_id"] for r in rows]
    assert "appended" in asset_ids, f"appended row lost in rewrite race: {asset_ids}"
    assert "keep" in asset_ids


def test_lock_blocks_second_acquire_until_released(tmp_home: Path) -> None:
    """A second lock acquisition must block until the first releases — the
    hard guarantee the os.replace race fix relies on. If this ever regresses
    (lock becomes a no-op), the append-during-rewrite race silently returns."""
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
