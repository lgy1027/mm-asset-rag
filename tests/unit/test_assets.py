"""Tests for mm_asset_rag.assets.

Reads the bundled real manifest at ``examples/data/chapter11_assets``,
and exercises the atomic + lock-protected writer helpers that replace
the bare ``write_text`` calls in ``scripts/expand_corpus.py`` and
``scripts/build_manifest.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mm_asset_rag.assets import (
    Asset,
    load_assets,
    locked_manifest_session,
    safe_write_manifest,
)


def test_load_assets_reads_manifest(examples_home: Path) -> None:
    assets = load_assets()
    # The bundled sample set grows as new PDFs and Picsum images are added;
    # we assert the previously-known lower bound plus the presence of the
    # fixtures the rest of the suite depends on.
    assert len(assets) >= 30
    # The RAG paper is still tagged as such in PDF_ENTRIES and the loader
    # derives the id from the filename stem, so we look it up by suffix.
    rag = next(a for a in assets if a.asset_id.startswith("retrieval_augmented"))
    assert rag.source_type == "pdf"
    assert rag.title.startswith("Retrieval-Augmented")
    suzanne = next(a for a in assets if "suzanne1" in a.asset_id)
    assert suzanne.source_type == "image"


def test_load_assets_respects_limit(examples_home: Path) -> None:
    assert len(load_assets(limit=3)) == 3
    full = load_assets(limit=0)
    assert len(full) >= 30


def test_load_assets_with_explicit_manifest_path(examples_home: Path) -> None:
    manifest = examples_home / "assets" / "asset_manifest.json"
    assets = load_assets(manifest_path=manifest)
    assert len(assets) >= 30
    # Relative paths in the manifest use backslashes (Windows-style); the
    # loader normalizes them so file_path joins work on POSIX.
    rag = next(a for a in assets if a.asset_id.startswith("retrieval_augmented"))
    assert rag.file_path.exists()
    assert rag.file_path.suffix == ".pdf"


def test_asset_file_path_uses_explicit_asset_dir(tmp_path: Path) -> None:
    asset = Asset(
        asset_id="x",
        title="",
        source_type="pdf",
        relative_path="doc.pdf",
        source_url="",
        tags=[],
        asset_dir=tmp_path,
    )
    assert asset.file_path == tmp_path / "doc.pdf"


# ─── Manifest safe-writer tests ─────────────────────────────────────────


def _initial_payload() -> dict:
    return {
        "name": "test",
        "total": 1,
        "records": [{"id": "alpha", "title": "Alpha", "type": "pdf", "path": "p/a.pdf"}],
    }


def test_safe_write_manifest_creates_new_file(tmp_path: Path) -> None:
    """First write creates the file (no .bak since there was nothing to back up)."""
    target = tmp_path / "manifest.json"
    safe_write_manifest(target, _initial_payload(), backup=True)
    assert target.exists()
    assert not (target.with_suffix(target.suffix + ".bak")).exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["total"] == 1
    assert payload["records"][0]["id"] == "alpha"


def test_safe_write_manifest_creates_backup_on_overwrite(tmp_path: Path) -> None:
    """Second write leaves a .bak with the previous contents."""
    target = tmp_path / "manifest.json"
    safe_write_manifest(target, _initial_payload(), backup=True)
    new_payload = _initial_payload()
    new_payload["records"].append(
        {"id": "beta", "title": "Beta", "type": "pdf", "path": "p/b.pdf"}
    )
    new_payload["total"] = 2
    safe_write_manifest(target, new_payload, backup=True)

    assert json.loads(target.read_text(encoding="utf-8"))["total"] == 2
    bak = target.with_suffix(target.suffix + ".bak")
    assert bak.exists()
    assert json.loads(bak.read_text(encoding="utf-8"))["total"] == 1


def test_safe_write_manifest_skip_backup(tmp_path: Path) -> None:
    """``backup=False`` is honoured for the first-and-second-write case."""
    target = tmp_path / "manifest.json"
    safe_write_manifest(target, _initial_payload(), backup=True)
    safe_write_manifest(target, _initial_payload(), backup=False)
    assert not (target.with_suffix(target.suffix + ".bak")).exists()


def test_safe_write_manifest_does_not_leave_temp_files_on_success(tmp_path: Path) -> None:
    """Atomic replace cleans up the temp file on success."""
    target = tmp_path / "manifest.json"
    safe_write_manifest(target, _initial_payload())
    leftovers = list(tmp_path.glob("manifest.json.*.tmp"))
    assert leftovers == []


def test_safe_write_manifest_atomic_replace_uses_tempfile(tmp_path: Path, monkeypatch) -> None:
    """The write goes through tempfile.mkstemp + os.replace."""
    target = tmp_path / "manifest.json"
    seen: dict[str, int] = {"mkstemp": 0, "replace": 0}

    real_mkstemp = __import__("tempfile").mkstemp
    real_replace = os.replace

    def fake_mkstemp(*args, **kwargs):
        seen["mkstemp"] += 1
        return real_mkstemp(*args, **kwargs)

    def fake_replace(*args, **kwargs):
        seen["replace"] += 1
        return real_replace(*args, **kwargs)

    monkeypatch.setattr("mm_asset_rag.assets.tempfile.mkstemp", fake_mkstemp)
    monkeypatch.setattr("mm_asset_rag.assets.os.replace", fake_replace)

    safe_write_manifest(target, _initial_payload())
    assert seen["mkstemp"] == 1
    assert seen["replace"] == 1


def test_locked_manifest_session_persists_mutation(tmp_path: Path) -> None:
    """In-place mutations inside the context manager are written on exit."""
    target = tmp_path / "manifest.json"
    safe_write_manifest(target, _initial_payload(), backup=False)

    with locked_manifest_session(target) as payload:
        payload["records"].append(
            {"id": "beta", "title": "Beta", "type": "pdf", "path": "p/b.pdf"}
        )
        payload["total"] = 2

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["total"] == 2
    assert {r["id"] for r in on_disk["records"]} == {"alpha", "beta"}


def test_locked_manifest_session_leaves_file_unchanged_on_exception(
    tmp_path: Path,
) -> None:
    """A body that raises leaves the manifest unchanged (atomic abort)."""
    target = tmp_path / "manifest.json"
    safe_write_manifest(target, _initial_payload(), backup=False)

    with pytest.raises(RuntimeError, match="boom"):
        with locked_manifest_session(target) as payload:
            payload["records"].append(
                {"id": "beta", "title": "Beta", "type": "pdf", "path": "p/b.pdf"}
            )
            payload["total"] = 99
            raise RuntimeError("boom")

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    # Manifest was not overwritten — body raise aborts the atomic write.
    assert on_disk["total"] == 1
    assert {r["id"] for r in on_disk["records"]} == {"alpha"}


def test_locked_manifest_session_seeds_empty_manifest(tmp_path: Path) -> None:
    """First-time write of a non-existent manifest gets a sensible seed."""
    target = tmp_path / "manifest.json"
    assert not target.exists()

    with locked_manifest_session(target) as payload:
        payload["records"].append(
            {"id": "first", "title": "First", "type": "pdf", "path": "p/f.pdf"}
        )
        payload["total"] = 1

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["total"] == 1
    assert on_disk["records"][0]["id"] == "first"


def test_locked_manifest_session_cleans_up_lock_file(tmp_path: Path) -> None:
    """The .lock sidecar is removed on context exit (success and failure)."""
    target = tmp_path / "manifest.json"
    safe_write_manifest(target, _initial_payload(), backup=False)
    with locked_manifest_session(target) as payload:
        # Inside the context the lock file exists.
        assert (target.with_suffix(target.suffix + ".lock")).exists()
    # After exit the lock file is gone.
    assert not (target.with_suffix(target.suffix + ".lock")).exists()
