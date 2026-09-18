from __future__ import annotations

import sqlite3

import pytest

from mm_asset_rag.ingest.task_store import TaskStore


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


def test_load_ignores_fields_from_older_schema_versions(tmp_path):
    """Records persisted by an older schema (e.g. carrying
    ``version_statuses``) must not crash task listing/retry."""
    import sqlite3

    db = tmp_path / "tasks.db"
    payload = {
        "task_id": "legacy-1",
        "kind": "ingest",
        "status": "done",
        "version_statuses": {"doc-a": "indexed"},
        "unknown_future_field": 123,
    }
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE tasks (task_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL)"
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            ("legacy-1", __import__("json").dumps(payload), 1.0),
        )

    store = TaskStore.__new__(TaskStore)
    store._data_dir = tmp_path
    records = store.load()
    assert [r.task_id for r in records] == ["legacy-1"]
    assert records[0].status == "done"
