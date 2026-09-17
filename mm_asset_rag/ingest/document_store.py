import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from ..core.knowledge_models import Chunk
from ..core.paths import get_documents_jsonl


@contextmanager
def _advisory_lock(path: Path, *, exclusive: bool):
    """Cross-process ``fcntl`` advisory lock on a sidecar ``.lock`` file.

    ``exclusive=True`` → ``LOCK_EX`` (writers serialise against each other
    and against readers); ``exclusive=False`` → ``LOCK_SH`` (readers share,
    block a concurrent writer from exposing a half-written line).

    Best-effort on platforms without ``fcntl`` (e.g. Windows): no-op lock,
    matching the pre-fix behaviour, rather than crashing the caller. The
    lock fd is closed in ``finally`` so a held lock can't leak past an
    exception in the critical section.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    acquired = False
    try:
        import fcntl

        lock_fd = lock_path.open("w")
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_fd.fileno(), mode)
        acquired = True
    except (ImportError, OSError):
        # No fcntl (Windows) or lock file not openable — degrade to no
        # cross-process lock. Same as pre-fix behaviour.
        pass
    try:
        yield
    finally:
        if acquired and lock_fd is not None:
            try:
                import fcntl

                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            lock_fd.close()


@contextmanager
def documents_jsonl_lock(path: Path | None = None):
    """Exclusive cross-process advisory lock guarding ``documents.jsonl``
    writers.

    Two write shapes touch this file and must not overlap, or data is
    lost:

    * the ingest workflow appends chunk rows for a parsed current document
      (``target.open("a")``).
    * document/version lifecycle cleanup does a read → tmp →
      ``os.replace`` rewrite for exact version rows (including force retry).

    If the rewrite's ``os.replace`` swaps the file out while an appender
    still holds the old fd, the appender keeps writing to the now-unlinked
    inode and those chunk rows vanish (recovered only by a later reindex).
    A process-wide threading lock can't help when the two writers are in
    different processes (the API server appending while a CLI retry
    rewrites). This OS-level advisory lock serialises both.
    """
    target = path or get_documents_jsonl()
    with _advisory_lock(target, exclusive=True):
        yield


def write_documents(documents: list[Chunk], path: Path | None = None) -> None:
    """Persist only complete document-version chunk rows."""
    target = path or get_documents_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Validate before touching the existing store, then replace it atomically
    # under the same cross-process lock used by append/rewrite callers.
    if any(not isinstance(document, Chunk) for document in documents):
        raise TypeError("documents.jsonl accepts Chunk records only")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with documents_jsonl_lock(target):
        try:
            with temporary.open("w", encoding="utf-8") as file_obj:
                for document in documents:
                    file_obj.write(json.dumps(document.to_record(), ensure_ascii=False) + "\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def append_documents(documents: list[Chunk], path: Path | None = None) -> None:
    """Append validated v2 chunks without exposing legacy DTO rows."""
    target = path or get_documents_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(not isinstance(document, Chunk) for document in documents):
        raise TypeError("documents.jsonl accepts Chunk records only")
    with documents_jsonl_lock(target), target.open("a", encoding="utf-8") as file_obj:
        for document in documents:
            file_obj.write(json.dumps(document.to_record(), ensure_ascii=False) + "\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())


def read_documents(path: Path | None = None) -> list[Chunk]:
    """Read complete v2 ``Chunk`` rows from the JSONL store.

    Holds a shared advisory lock so a concurrent appender can't expose a
    half-written line mid-parse. A corrupted row (truncated write, partial
    flush, OOM kill mid-append) is skipped with a warning rather than
    aborting the whole index build — matching :func:`asset_index.load_records`
    and the exact document-version row cleanup path. The next reindex
    rebuilds from the surviving rows; losing one chunk is preferable to failing the
    whole index build.
    """
    target = path or get_documents_jsonl()
    if not target.exists():
        raise RuntimeError(f"Document JSONL not found: {target}")
    documents: list[Chunk] = []
    with _advisory_lock(target, exclusive=False), target.open("r", encoding="utf-8") as file_obj:
        for lineno, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                documents.append(Chunk.from_record(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(
                    f"[document_store] skip malformed row {lineno} in {target.name}: {exc}",
                    file=sys.stderr,
                )
                continue
    return documents
