"""Persistent storage for background task records.

SQLite remains the source of truth.  The legacy ``tasks.jsonl`` migration
and the best-effort ``tasks.jsonl.last`` debug tail live here because they
are persistence details; document-version files and ``documents.jsonl`` deliberately do
not.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import get_data_dir


@dataclass
class TaskRecord:
    task_id: str
    kind: str  # "parse" or "ingest"
    status: str = "pending"  # pending | running | done | partial | failed | interrupted
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    current: str = ""
    error: str | None = None
    uploaded_files: list[str] = field(default_factory=list)
    parse_options: dict[str, object] = field(default_factory=dict)
    source: str = "upload"
    origin_task_id: str | None = None
    force: bool = False
    failed_only: bool = False
    version_statuses: dict[str, str] = field(default_factory=dict)


def task_from_dict(obj: dict[str, object]) -> TaskRecord:
    """Build a ``TaskRecord`` while tolerating fields absent from old rows."""
    kwargs: dict[str, object] = {}
    for field_name in TaskRecord.__dataclass_fields__:
        if field_name in obj:
            kwargs[field_name] = obj[field_name]
    return TaskRecord(**kwargs)  # type: ignore[arg-type]


class TaskStore:
    """Own SQLite schema, migration, serialization, and task CRUD."""

    _PERSIST_LOCK = threading.Lock()

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir

    def load(self) -> list[TaskRecord]:
        """Load all persisted task records, migrating legacy JSONL once."""
        db_path = self.db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._maybe_legacy_migrate(db_path)
        try:
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = list(conn.execute("SELECT payload FROM tasks"))
        except sqlite3.DatabaseError as exc:
            print(f"[tasks] warning: could not open history db: {exc}")
            return []

        records: list[TaskRecord] = []
        for row in rows:
            try:
                obj = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
            task_id = obj.get("task_id") if isinstance(obj, dict) else None
            if isinstance(task_id, str) and task_id:
                records.append(task_from_dict(obj))
        return records

    def save(self, record: TaskRecord) -> None:
        """Atomically persist ``record`` and append its debug-tail snapshot."""
        try:
            db_path = self.db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(asdict(record), ensure_ascii=False)
            with self._PERSIST_LOCK, sqlite3.connect(str(db_path)) as conn:
                conn.isolation_level = None
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS tasks ("
                    "task_id TEXT PRIMARY KEY, "
                    "payload TEXT NOT NULL, "
                    "updated_at REAL NOT NULL"
                    ")"
                )
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS tasks_updated_at_idx ON tasks (updated_at DESC)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO tasks (task_id, payload, updated_at) VALUES (?, ?, ?)",
                    (record.task_id, payload, time.time()),
                )
        except (sqlite3.DatabaseError, OSError) as exc:
            record.error = (record.error or "") + f"; persist failed: {exc}"
            print(f"[tasks] warning: could not persist {record.task_id}: {exc}")
            return

        try:
            jsonl_path = self.jsonl_path()
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
            with self._PERSIST_LOCK, jsonl_path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(line)
                file_obj.flush()
                os.fsync(file_obj.fileno())
        except OSError as exc:
            print(f"[tasks] warning: jsonl tail append failed for {record.task_id}: {exc}")

    def list(self) -> list[TaskRecord]:
        """Return persisted tasks ordered by descending update time."""
        db_path = self.db_path()
        if not db_path.exists():
            return []
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute("SELECT payload FROM tasks ORDER BY updated_at DESC").fetchall()
        except sqlite3.DatabaseError as exc:
            print(f"[tasks] warning: list_tasks db read failed: {exc}")
            return []

        records: list[TaskRecord] = []
        for (payload,) in rows:
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(task_from_dict(obj))
        return records

    def delete(self, task_id: str) -> bool:
        """Delete one SQLite task row and report whether it existed."""
        db_path = self.db_path()
        if not db_path.exists():
            return False
        try:
            with self._PERSIST_LOCK, sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                return cursor.rowcount > 0
        except (sqlite3.DatabaseError, OSError) as exc:
            print(f"[tasks] warning: could not delete {task_id}: {exc}")
            return False

    def db_path(self) -> Path:
        return self._root() / "tasks.db"

    def jsonl_path(self) -> Path:
        return self._root() / "tasks.jsonl.last"

    def legacy_path(self) -> Path:
        return self._root() / "tasks.jsonl"

    def _root(self) -> Path:
        return self._data_dir if self._data_dir is not None else get_data_dir()

    def _maybe_legacy_migrate(self, db_path: Path) -> None:
        if db_path.exists():
            return
        legacy = self.legacy_path()
        if not legacy.exists():
            return
        with self._PERSIST_LOCK, sqlite3.connect(str(db_path)) as conn:
            conn.isolation_level = None
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                "task_id TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, "
                "updated_at REAL NOT NULL"
                ")"
            )
            imported = 0
            with legacy.open("r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    task_id = obj.get("task_id") if isinstance(obj, dict) else None
                    if not isinstance(task_id, str) or not task_id:
                        continue
                    conn.execute(
                        "INSERT OR REPLACE INTO tasks (task_id, payload, updated_at)"
                        " VALUES (?, ?, ?)",
                        (task_id, json.dumps(obj, ensure_ascii=False), time.time()),
                    )
                    imported += 1
        legacy.rename(legacy.with_name(legacy.name + ".migrated"))
        print(f"[tasks] migrated {imported} record(s) from legacy {legacy}")
