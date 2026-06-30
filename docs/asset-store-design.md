# Asset store design

The bundled sample set — and every user-supplied asset directory the
project accepts — is described by a single `asset_manifest.json` file.
This document records **why** we picked a plain JSON file, the
production-grade discipline layered on top, and when to migrate to a
database.

## Decision: file-based manifest, with three hardening layers

A file is the right primary store for this project because the
bundled set is **small** (< 1,000 records, < 100 KB on disk) and we
get four wins for free that a database does not give us:

- **No ops overhead** — no DB to spin up, no schema migrations, no
  connection pool to monitor.
- **Version-controlled** — `git diff` on a manifest is a readable,
  reviewable change. A 200-record manifest diff is easier to skim
  than a SQL migration.
- **Portable** — `git clone` ships the data definition; the user is
  not asked to seed a database before the first `mmrag parse`.
- **Human-editable** — adding a record by hand (or via
  `scripts/build_manifest.py` / `scripts/expand_corpus.py`) is a
  trivial JSON edit, not a SQL `INSERT`.

The price of those wins is that a raw `write_text` is fragile under
crashes and concurrent writers, so we layer three pieces of
production discipline on top:

1. **Atomic replace** — `tempfile.mkstemp` + `os.replace` means a crash
   mid-write never leaves a half-written file on disk. A reader at
   any point sees either the previous version or the new version,
   never the bytes in between.
2. **`.bak` rotation** — every successful write copies the previous
   file to `asset_manifest.json.bak` first. A user who clobbers a
   record can recover by hand without a `git revert`.
3. **`fcntl.flock` session** — `locked_manifest_session()` wraps the
   read-modify-write cycle under an exclusive flock. Two writers
   cannot interleave and drop each other's records (the realistic
   scenarios: `POST /upload` racing `mmrag reindex`, or two CI jobs).

## How the layers compose

```
caller
  │
  │ with locked_manifest_session(manifest_path) as payload:
  ▼
  ┌──────────────────────────────────────────┐
  │ flock(<manifest>.lock, LOCK_EX)         │  ← concurrent writer safety
  │ payload = json.loads(manifest_path)     │  ← read
  │ yield payload                           │  ← caller mutates
  │ safe_write_manifest(manifest_path,      │  ← atomic + .bak
  │                       payload)          │
  │ flock(... LOCK_UN)                      │
  │ .lock.unlink()                          │
  └──────────────────────────────────────────┘
```

## When to migrate to SQLite

| Scale | Recommendation |
| --- | --- |
| < 1,000 records | **File (this design)** |
| 1k - 100k records | SQLite, single-file `assets` table, indexed on `id` + `tags` |
| > 100k records OR multi-service writers | PostgreSQL |

The migration is mechanical: the JSON record schema is flat and
maps 1:1 onto a SQLite row. The interface in
`mm_asset_rag.assets` is already abstracted — `load_assets()` is
the only public reader, `safe_write_manifest` /
`locked_manifest_session` the only public writers. Replacing the
file backend with a SQLite backend is a one-file change.

## Threat model: when does each layer matter?

| Failure mode | What saves us |
| --- | --- |
| `mmrag parse` killed mid-write | Atomic replace — reader sees the previous file |
| User accidentally deletes a record | `.bak` — recover by `cp asset_manifest.json.bak asset_manifest.json` |
| Two `POST /upload` requests race | `flock` — second one waits, then sees the first one's record |
| Disk full mid-write | `tempfile` lives in the same dir → `os.replace` fails atomically, original intact |
| Windows / restricted container (no `fcntl`) | Best-effort no-op lock — atomic replace still prevents half-written files |

## Public API

```python
from mm_asset_rag.assets import (
    load_assets,                  # reader; one-shot
    safe_write_manifest,          # low-level atomic + .bak writer
    locked_manifest_session,      # high-level read-modify-write under flock
)

# Typical caller pattern:
with locked_manifest_session(manifest_path) as payload:
    payload["records"].append(new_record)
    payload["total"] = len(payload["records"])
# written atomically on context exit

# Direct write (e.g. scripts/build_manifest.py builds a fresh payload):
safe_write_manifest(manifest_path, payload, backup=True)
```

## What lives where

| Layer | Where |
| --- | --- |
| Reader (`load_assets`) | `mm_asset_rag/assets.py:load_assets` |
| Atomic writer | `mm_asset_rag/assets.py:safe_write_manifest` |
| Locked session | `mm_asset_rag/assets.py:locked_manifest_session` |
| Tests | `tests/unit/test_assets.py` (atomic replace, backup, exception path, lock cleanup) |
| Callers | `scripts/expand_corpus.py`, `scripts/build_manifest.py` |

## Out of scope

- **Concurrent reads** — the manifest is small enough that
  `json.loads` is fast even at 1,000 records; we don't need a
  per-reader cache. If profiling shows it matters, a
  `functools.lru_cache` around `load_assets` is the natural next step.
- **Schema versioning** — the manifest format is implicit in
  `Asset` dataclass fields. When we add a v2 field, the loader
  defaults missing keys to `""` / `[]`, so old manifests keep
  loading. A formal `$schema` URL would be the next step if external
  tooling starts reading the manifest.
- **Multi-region writes** — `flock` is single-host. A cross-host
  manifest needs a database with cross-host locking (e.g.
  PostgreSQL `SELECT … FOR UPDATE`) or a CRDT layer.
