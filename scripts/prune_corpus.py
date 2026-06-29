"""Mirror of ``expand_corpus.py``: drop assets that are no longer in the manifest.

The companion to :mod:`scripts.expand_corpus`. Deleting a record from
``asset_manifest.json`` does NOT, on its own, remove the corresponding
parsed markdown or ``documents.jsonl`` rows — they become orphans that
waste disk + memory + slow down searches. This script reconciles the
on-disk state with the manifest.

What it removes (only with ``--yes``; default is dry-run):

- ``$MM_ASSET_RAG_HOME/parsed/<orphan_id>/`` — markdown pages produced
  by an earlier ``mmrag parse``. Pure cache; recreatable from the PDF.
- ``$MM_ASSET_RAG_HOME/documents.jsonl`` rows whose ``asset_id`` is
  not in the manifest. Pure cache; recreatable by re-running
  ``mmrag parse``.
- ``$MM_ASSET_RAG_HOME/tasks.jsonl`` rows older than
  ``--keep-tasks-days`` (default 30). Pure history.

What it **never** touches:

- The original PDF / image files under
  ``examples/data/chapter11_assets/{pdfs,images}/`` (the source of truth).
- The Qdrant collection. ``mmrag reindex`` is always drop+rebuild, so
  pruning the source ``documents.jsonl`` is enough — re-running it
  afterwards rebuilds the index from the surviving chunks only. This
  sidesteps the single-process Qdrant file lock that would otherwise
  require the API server to be stopped.
- ``asset_manifest.json`` itself (caller's responsibility).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def _manifest_ids(manifest_path: Path) -> set[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(r["id"]) for r in payload.get("records", [])}


def _dry_label() -> str:
    return "DRY-RUN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--assets-dir",
        default="examples/data/chapter11_assets",
        help="Root of the bundled asset directory (for the manifest).",
    )
    parser.add_argument(
        "--home",
        default=os.environ.get("MM_ASSET_RAG_HOME", str(Path.home() / ".mm_asset_rag")),
        help="MM_ASSET_RAG_HOME override.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--keep-tasks-days",
        type=int,
        default=30,
        help="Drop tasks.jsonl rows older than this many days. "
             "Set to 0 to disable task pruning.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.assets_dir) / "asset_manifest.json"
    home = Path(args.home)
    parsed_dir = home / "parsed"
    docs_path = home / "documents.jsonl"
    tasks_path = home / "tasks.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    active_ids = _manifest_ids(manifest_path)
    mode = "EXECUTE" if args.yes else _dry_label()

    # ─── 1. Orphan parsed dirs ───────────────────────────────────────────
    orphan_dirs: list[Path] = []
    if parsed_dir.exists():
        for child in sorted(parsed_dir.iterdir()):
            if child.is_dir() and child.name not in active_ids:
                orphan_dirs.append(child)

    # ─── 2. Orphan documents.jsonl chunks ────────────────────────────────
    orphan_chunk_count = 0
    orphan_chunk_asset_ids: set[str] = set()
    surviving_docs: list[str] = []
    if docs_path.exists():
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                surviving_docs.append(line)
                continue
            aid = doc.get("metadata", {}).get("asset_id", "")
            if aid and aid not in active_ids:
                orphan_chunk_count += 1
                orphan_chunk_asset_ids.add(aid)
            else:
                surviving_docs.append(line)

    # ─── 3. Stale task rows ──────────────────────────────────────────────
    stale_task_count = 0
    surviving_tasks: list[str] = []
    cutoff: datetime | None = None
    if args.keep_tasks_days > 0 and tasks_path.exists():
        cutoff = datetime.now() - timedelta(days=args.keep_tasks_days)
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                surviving_tasks.append(line)
                continue
            ts = rec.get("created_at") or rec.get("timestamp") or ""
            try:
                when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                # No timestamp or unparseable — keep.
                surviving_tasks.append(line)
                continue
            if when < cutoff:
                stale_task_count += 1
            else:
                surviving_tasks.append(line)

    # ─── Print the plan ──────────────────────────────────────────────────
    print(f"== prune_corpus ({mode}) ==")
    print(f"  manifest:           {manifest_path} ({len(active_ids)} active assets)")
    print(f"  parsed dirs orphans: {len(orphan_dirs)}")
    for d in orphan_dirs:
        print(f"    - {d.name}")
    print(f"  documents.jsonl orphans: {orphan_chunk_count} chunks "
          f"across {len(orphan_chunk_asset_ids)} assets")
    print(f"  qdrant:                 not touched here — run `mmrag reindex` afterwards")
    if cutoff is not None:
        print(f"  tasks.jsonl:          drop {stale_task_count} rows older than {cutoff.date().isoformat()}")

    if not args.yes:
        print("\nRe-run with --yes to apply.")
        return

    # ─── Apply ───────────────────────────────────────────────────────────
    deleted_dirs = 0
    for d in orphan_dirs:
        try:
            shutil.rmtree(d)
            deleted_dirs += 1
        except OSError as exc:
            print(f"  rm {d} failed: {exc}")
    print(f"  removed {deleted_dirs} parsed dirs")

    if docs_path.exists() and orphan_chunk_count:
        docs_path.write_text(
            "\n".join(surviving_docs) + ("\n" if surviving_docs else ""),
            encoding="utf-8",
        )
        print(f"  rewrote {docs_path} ({len(surviving_docs)} chunks kept)")

    if cutoff is not None and stale_task_count:
        tasks_path.write_text(
            "\n".join(surviving_tasks) + ("\n" if surviving_tasks else ""),
            encoding="utf-8",
        )
        print(f"  rewrote {tasks_path} ({len(surviving_tasks)} rows kept, {stale_task_count} dropped)")

    print("\nDone. Run `mmrag reindex` to rebuild the Qdrant text collection cleanly.")


if __name__ == "__main__":
    main()