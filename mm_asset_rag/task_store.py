"""SQLite storage for background task records."""

from __future__ import annotations

import json
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


class TaskStore:
    """Own task serialization and CRUD."""

    _PERSIST_LOCK = threading.Lock()

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir

    def load(self) -> list[TaskRecord]:
        """Load all persisted task records."""
        db_path = self.db_path()
        if not db_path.exists():
            return []
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
                records.append(TaskRecord(**obj))
        return records

    def save(self, record: TaskRecord) -> None:
        """Atomically persist ``record``."""
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
                records.append(TaskRecord(**obj))
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

    def _root(self) -> Path:
        return self._data_dir if self._data_dir is not None else get_data_dir()
