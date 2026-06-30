import contextlib
import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .paths import get_assets_dir, get_manifest_path


@dataclass(frozen=True)
class Asset:
    asset_id: str
    title: str
    source_type: str
    relative_path: str
    source_url: str
    tags: list[str]
    asset_dir: Path = field(default_factory=get_assets_dir)

    @property
    def file_path(self) -> Path:
        return self.asset_dir / self.relative_path


def load_assets(limit: int = 0, manifest_path: Path | None = None) -> list[Asset]:
    path = manifest_path or get_manifest_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = [
        Asset(
            asset_id=str(item["id"]),
            title=str(item["title"]),
            source_type=str(item["type"]),
            relative_path=str(item["path"]).replace("\\", "/"),
            source_url=str(item.get("source_url", "")),
            tags=[str(tag) for tag in item.get("tags", [])],
        )
        for item in payload["records"]
    ]
    if limit > 0:
        return assets[:limit]
    return assets


# ─── Safe manifest writers ────────────────────────────────────────────────
# The bundled data set is small (< 1k records, < 100 KB on disk) so a
# plain JSON file is the right primary store: human-readable, git-trackable,
# no schema migration, no DB connection to manage. We add three pieces of
# production discipline on top so the file is robust under concurrent
# writers and crashes:
#
#   * ``safe_write_manifest`` — atomic temp-file + ``os.replace`` so a
#     crash mid-write never leaves a half-written file on disk.
#   * backup ``.bak`` rotation — best-effort copy of the previous file
#     before each replace, so an accidentally clobbered record can be
#     recovered by hand without a git revert.
#   * ``locked_manifest_session`` — context manager that wraps a
#     read-modify-write cycle under an exclusive ``fcntl.flock`` so two
#     concurrent writers (``POST /upload`` racing ``mmrag reindex``, or two
#     CI jobs) cannot lose each other's records.


def safe_write_manifest(
    manifest_path: Path,
    payload: dict,
    *,
    backup: bool = True,
) -> None:
    """Atomically replace ``manifest_path`` with serialised ``payload``.

    The temp-file + ``os.replace`` sequence is atomic on POSIX: a reader
    at any point sees either the previous contents or the new contents,
    never a half-written file. The ``.bak`` rotation runs *before* the
    replace; it is best-effort and never raises.

    Args:
        manifest_path: destination path; created (with parents) if missing.
        payload: dict with at least ``"records"`` (list) and ``"total"`` (int).
        backup: if True, copy the existing file to ``<name>.json.bak`` before
                the atomic replace. Skipped on the very first write.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if backup and manifest_path.exists():
        try:
            shutil.copy2(
                manifest_path,
                manifest_path.with_suffix(manifest_path.suffix + ".bak"),
            )
        except OSError:
            # Best-effort: losing a backup is much less bad than losing
            # the manifest. Swallow and continue with the atomic replace.
            pass

    fd, tmp_path = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=manifest_path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, manifest_path)  # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


@contextmanager
def locked_manifest_session(
    manifest_path: Path,
    *,
    backup: bool = True,
) -> Iterator[dict]:
    """Read ``manifest_path`` for in-place mutation, atomically write it back.

    The whole read-modify-write cycle runs under an exclusive
    ``fcntl.flock`` on a sidecar ``.lock`` file so concurrent writers
    don't interleave and lose records. On platforms without ``fcntl``
    (Windows, restricted containers) the lock is a best-effort no-op —
    the atomic replace still prevents half-written files.

    Usage::

        with locked_manifest_session(manifest_path) as payload:
            payload["records"].append(new_record)
            payload["total"] = len(payload["records"])
        # payload has been written atomically on context exit.

    If the body raises, the file is left unchanged (the temp file is
    cleaned up but the existing manifest is not overwritten).
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_fd = open(lock_path, "w")
    try:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        except (NameError, OSError):
            # fcntl unavailable (Windows, restricted env). The atomic
            # replace still saves us from half-written files; only
            # concurrent-writer safety is lost.
            pass

        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            payload = {"name": "", "total": 0, "records": []}
        yield payload

        safe_write_manifest(manifest_path, payload, backup=backup)
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except (NameError, OSError):
            pass
        lock_fd.close()
        try:
            lock_path.unlink()
        except OSError:
            pass
