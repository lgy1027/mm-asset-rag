"""Tests for ``mm_asset_rag.asset_index``."""

from __future__ import annotations

import json
from pathlib import Path

from mm_asset_rag.asset_index import (
    AssetIndexEntry,
    find_active_by_asset_id,
    find_by_sha256,
    latest_by_asset_id,
    list_active,
    load_entries,
    mark_deleted,
    upsert_entry,
)


def _write_entry(path: Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields) + "\n")


def test_load_entries_tolerates_corrupt_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text("\nnot json\n", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "asset_id": "a",
                "sha256": "h1",
                "source_type": "image",
                "relative_path": "images/a.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entries = load_entries(path)
    assert len(entries) == 1
    assert entries[0].asset_id == "a"


def test_load_entries_missing_file(tmp_path: Path) -> None:
    assert load_entries(tmp_path / "missing.jsonl") == []


def test_upsert_and_find(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a", sha256="hash", source_type="image", relative_path="images/a.png"
        ),
        path=path,
    )
    assert find_by_sha256("hash", path=path).asset_id == "a"
    assert find_active_by_asset_id("a", path=path).asset_id == "a"


def test_mark_deleted_toggles_state(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a", sha256="hash", source_type="image", relative_path="images/a.png"
        ),
        path=path,
    )
    assert mark_deleted("a", path=path, at=42.0) is True
    # Already deleted: further mark_deleted is a no-op.
    assert mark_deleted("a", path=path, at=43.0) is False
    assert find_by_sha256("hash", path=path) is None
    assert find_active_by_asset_id("a", path=path) is None


def test_latest_by_asset_id_folds_history(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    _write_entry(
        path,
        asset_id="a",
        sha256="h1",
        source_type="image",
        relative_path="images/a.png",
    )
    _write_entry(
        path,
        asset_id="a",
        sha256="h2",
        source_type="image",
        relative_path="images/a2.png",
    )
    latest = latest_by_asset_id(path=path)
    assert latest["a"].sha256 == "h2"


def test_list_active_skips_deleted(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a", sha256="h1", source_type="image", relative_path="images/a.png"
        ),
        path=path,
    )
    upsert_entry(
        AssetIndexEntry(
            asset_id="b", sha256="h2", source_type="image", relative_path="images/b.png"
        ),
        path=path,
    )
    mark_deleted("a", path=path, at=1.0)
    active = list_active(path=path)
    assert [e.asset_id for e in active] == ["b"]


def test_entry_round_trip_preserves_tags(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a",
            sha256="hash",
            source_type="image",
            relative_path="images/a.png",
            tags=["beach", "sunset"],
        ),
        path=path,
    )
    entries = load_entries(path)
    assert entries[0].tags == ["beach", "sunset"]


def test_entry_loads_default_tags_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text(
        __import__("json").dumps(
            {
                "asset_id": "legacy",
                "sha256": "h",
                "source_type": "image",
                "relative_path": "images/legacy.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entries = load_entries(path)
    assert entries[0].tags == []
