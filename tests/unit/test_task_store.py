from __future__ import annotations

import sqlite3

import pytest

from mm_asset_rag.task_store import TaskStore


def test_load_initializes_empty_database_schema(tmp_path) -> None:
    store = TaskStore(tmp_path)

    assert store.load() == []

    with sqlite3.connect(store.db_path()) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone()


def test_load_raises_for_corrupt_database(tmp_path) -> None:
    db_path = tmp_path / "tasks.db"
    db_path.write_bytes(b"not sqlite")

    with pytest.raises(RuntimeError, match="task store"):
        TaskStore(tmp_path).load()
