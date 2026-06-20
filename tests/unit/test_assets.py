"""Tests for mm_asset_rag.assets.

Reads the bundled real manifest at ``examples/data/chapter11_assets``.
"""

from __future__ import annotations

from pathlib import Path

from mm_asset_rag.assets import Asset, load_assets


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
