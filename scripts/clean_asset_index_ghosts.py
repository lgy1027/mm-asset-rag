#!/usr/bin/env python3
"""Remove ghost rows from ``asset_index.jsonl``.

``asset_index.jsonl`` is append-only (``latest row wins`` on read). Over
repeated eval runs the same logical document was re-uploaded under new
random ``preview_id`` suffixes, so the file accumulated hundreds of
active rows whose underlying ``assets/`` file no longer exists — the
only thing still pointing at the real 135-document corpus is
``documents.jsonl`` (the parsed-chunk store the retriever actually
reads).

This script keeps every ``asset_index.jsonl`` row whose ``asset_id`` also
appears in ``documents.jsonl``'s ``metadata.asset_id`` set (and drops
active rows that don't), preserving each kept row's ``sha256`` /
``relative_path`` / etc. verbatim. Tombstones (``deleted=True``) are kept
as-is so delete history isn't rewritten.

Idempotent: re-running is a no-op once ghosts are gone. Backs up the
original file to ``asset_index.jsonl.bak`` before rewriting.

Usage::

    python scripts/clean_asset_index_ghosts.py            # default home
    python scripts/clean_asset_index_ghosts.py /path/to/home
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _data_home() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    env = __import__("os").environ.get("MM_ASSET_RAG_HOME")
    return Path(env).expanduser() if env else Path.home() / ".mm_asset_rag"


def _live_asset_ids(documents_jsonl: Path) -> set[str]:
    ids: set[str] = set()
    if not documents_jsonl.exists():
        return ids
    with documents_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = row.get("metadata", {}).get("asset_id")
            if aid:
                ids.add(aid)
    return ids


def main() -> int:
    home = _data_home()
    index_path = home / "asset_index.jsonl"
    documents_path = home / "documents.jsonl"
    if not index_path.exists():
        print(f"no asset_index.jsonl at {index_path}", file=sys.stderr)
        return 1

    live = _live_asset_ids(documents_path)
    print(f"documents.jsonl references {len(live)} live asset_ids")

    kept: list[str] = []
    active_total = 0
    ghost_active = 0
    tombstones = 0
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("deleted"):
                tombstones += 1
                kept.append(line)
                continue
            active_total += 1
            if row.get("asset_id") in live:
                kept.append(line)
            else:
                ghost_active += 1

    backup = index_path.with_suffix(index_path.suffix + ".bak")
    backup.write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
    index_path.write_text("".join(kept), encoding="utf-8")

    print(
        f"asset_index.jsonl: {active_total} active rows -> "
        f"{active_total - ghost_active} kept, {ghost_active} ghosts dropped, "
        f"{tombstones} tombstones preserved"
    )
    print(f"backup written to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
