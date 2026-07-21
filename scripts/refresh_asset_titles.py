#!/usr/bin/env python3
"""Refresh ``asset_title`` on already-indexed PDFs without re-parsing.

The eval set's expected ids use paper-spec canonical titles (e.g.
``"Learning Transferable Visual Models From Natural Language Supervision"``)
but with ``AUTO_META_ENABLED=false`` the indexed ``asset_title`` is just the
filename stem (``"Clip"`` for ``clip.pdf``), so the matcher never lines the
two up. Re-parsing re-embeds every chunk (costly, and title doesn't enter the
embedding anyway), so instead we:

1. call ``auto_meta_pdf_first_page`` per PDF to get the LLM-derived title,
2. ``upsert_entry`` the new ``asset_title`` into ``asset_index.jsonl``,
3. ``set_payload`` the new ``asset_title`` onto every Qdrant text point whose
   payload ``asset_id`` matches — no vector recompute.

Backs up ``asset_index.jsonl`` to ``.bak`` first. Safe to re-run.

Usage::

    python scripts/refresh_asset_titles.py            # default home
    python scripts/refresh_asset_titles.py /path/to/home
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force auto_meta on for this process regardless of .env, then clear the
# settings cache so the override takes. auto_meta_pdf_first_page early-returns
# when settings.auto_meta_enabled is false. gemma4 is slow under concurrency
# (11-13s per first-page render), so bump the timeout well past the default 30s
# and run single-threaded to avoid ollama read timeouts stacking up.
os.environ["AUTO_META_ENABLED"] = "true"
os.environ.setdefault("AUTO_META_TIMEOUT", "120")
os.environ.setdefault("AUTO_META_MAX_CONCURRENCY", "1")


def _data_home() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    env = os.environ.get("MM_ASSET_RAG_HOME")
    return Path(env).expanduser() if env else Path.home() / ".mm_asset_rag"


def main() -> int:
    # Defer imports until after the env override is set, so get_settings()
    # picks up AUTO_META_ENABLED=true.
    from mm_asset_rag import asset_index, auto_meta
    from mm_asset_rag.backends import qdrant_backend
    from mm_asset_rag.paths import get_assets_dir
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    if not settings.auto_meta_enabled:
        print("AUTO_META_ENABLED still false; cannot generate titles", file=sys.stderr, flush=True)
        return 1

    home = _data_home()
    index_path = home / "asset_index.jsonl"
    backup = index_path.with_suffix(index_path.suffix + ".bak")
    backup.write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"backed up asset_index.jsonl -> {backup}", flush=True)

    entries = asset_index.list_active()
    pdf_entries = [e for e in entries if e.source_type == "pdf" and e.asset_id]
    print(f"{len(entries)} active entries, {len(pdf_entries)} PDFs to refresh", flush=True)

    assets_dir = get_assets_dir()
    concurrency = max(1, settings.auto_meta_max_concurrency)

    def refresh_one(entry: asset_index.AssetIndexEntry) -> tuple[str, str | None]:
        rel = entry.relative_path
        pdf_path = assets_dir / rel if not Path(rel).is_absolute() else Path(rel)
        if not pdf_path.exists():
            return entry.asset_id, None
        # Retry: ollama occasionally read-times-out under load; the title is
        # cheap to regenerate and a transient timeout shouldn't drop the asset.
        last: str | None = None
        for _ in range(3):
            am = auto_meta.auto_meta_pdf_first_page(pdf_path)
            if am is not None and am.title:
                last = am.title.strip()
                break
        return entry.asset_id, last

    updated: dict[str, str] = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(refresh_one, e): e for e in pdf_entries}
        for i, fut in enumerate(as_completed(futures), 1):
            aid, title = fut.result()
            if title:
                updated[aid] = title
            else:
                failed += 1
            if i % 10 == 0 or i == len(pdf_entries):
                print(
                    f"  {i}/{len(pdf_entries)} done, {len(updated)} titled, {failed} skipped",
                    flush=True,
                )

    if not updated:
        print("no titles generated; nothing to update", file=sys.stderr, flush=True)
        return 1

    # 1) asset_index.jsonl: rewrite asset_title on each updated row.
    lines: list[str] = []
    touched = 0
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                lines.append(line)
                continue
            row = json.loads(line)
            if row.get("asset_id") in updated and not row.get("deleted"):
                row["asset_title"] = updated[row["asset_id"]]
                touched += 1
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    index_path.write_text("".join(lines), encoding="utf-8")
    print(f"asset_index.jsonl: {touched} rows retitled", flush=True)

    # 2) Qdrant text collection: set_payload asset_title per asset_id.
    client = qdrant_backend.get_qdrant_client()
    # Resolve the real collection name. It suffixes by vector dim
    # (e.g. mmrag_text_1024d). Prefer the dim-suffixed live collection over a
    # bare base name; among multiple suffixed leftovers pick the most
    # populated (the active index). "last match wins" previously risked
    # set_payload into a stale dim-suffixed leftover from an old embedding.
    base = qdrant_backend.TEXT_COLLECTION_BASE
    coll = None
    best_key = (-1, -1)  # (is_suffixed(0/1), point_count)
    for c in client.get_collections().collections:
        name = c.name
        if not name.startswith(base):
            continue
        is_suffixed = 1 if name != base else 0
        try:
            count = client.count(collection_name=name, exact=True).count
        except Exception:  # pragma: no cover
            count = 0
        if (is_suffixed, count) > best_key:
            coll = name
            best_key = (is_suffixed, count)
    if coll is None:
        print("no qdrant text collection found; skipping set_payload", file=sys.stderr, flush=True)
        return 0
    print(f"qdrant collection: {coll}", flush=True)
    from qdrant_client import models as qmodels

    pt_updated = 0
    for aid, title in updated.items():
        client.set_payload(
            collection_name=coll,
            payload={"asset_title": title},
            points=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="asset_id",
                        match=qmodels.MatchValue(value=aid),
                    )
                ]
            ),
            wait=True,
        )
        pt_updated += 1
    print(f"qdrant set_payload: {pt_updated} asset_ids updated", flush=True)

    print(f"\ndone: {len(updated)} PDFs retitled, {failed} skipped (no LLM title)", flush=True)
    print(f"backup at {backup}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
