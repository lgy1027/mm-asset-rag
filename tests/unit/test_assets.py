"""Tests for mm_asset_rag.assets.

Reads the bundled real manifest at ``examples/data/chapter11_assets``.
"""

from __future__ import annotations

from pathlib import Path

from mm_asset_rag.assets import Asset, load_assets


def test_load_assets_reads_manifest(examples_home: Path) -> None:
    assets = load_assets()
    assert len(assets) == 30
    by_id = {asset.asset_id: asset for asset in assets}
    assert by_id["pdf_rag"].source_type == "pdf"
    assert by_id["pdf_rag"].title.startswith("Retrieval-Augmented")
    assert by_id["img_01_opencv-sample-data-blender-suzanne1-jpg"].source_type == "image"


def test_load_assets_respects_limit(examples_home: Path) -> None:
    assert len(load_assets(limit=3)) == 3
    assert len(load_assets(limit=0)) == 30


def test_load_assets_with_explicit_manifest_path(examples_home: Path) -> None:
    manifest = examples_home / "assets" / "asset_manifest.json"
    assets = load_assets(manifest_path=manifest)
    assert len(assets) == 30
    # Relative paths in the manifest use backslashes (Windows-style); the
    # loader normalizes them so file_path joins work on POSIX.
    pdf_rag = next(a for a in assets if a.asset_id == "pdf_rag")
    assert pdf_rag.file_path.exists()
    assert pdf_rag.file_path.suffix == ".pdf"


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
