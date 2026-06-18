"""Tests for mm_asset_rag.assets."""

from __future__ import annotations

from pathlib import Path

from mm_asset_rag.assets import Asset, load_assets


def test_load_assets_reads_manifest(populated_home: Path) -> None:
    assets = load_assets()
    assert len(assets) == 2
    by_id = {asset.asset_id: asset for asset in assets}
    assert by_id["pdf_sample"].source_type == "pdf"
    assert by_id["img_sample"].source_type == "image"
    assert by_id["pdf_sample"].file_path == populated_home / "assets" / "sample.pdf"
    assert by_id["img_sample"].tags == ["test", "sample"]


def test_load_assets_respects_limit(populated_home: Path) -> None:
    assert len(load_assets(limit=1)) == 1


def test_load_assets_with_explicit_manifest_path(populated_home: Path) -> None:
    manifest = populated_home / "assets" / "asset_manifest.json"
    assets = load_assets(manifest_path=manifest)
    assert len(assets) == 2


def test_asset_file_path_uses_explicit_asset_dir(tmp_path) -> None:
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
