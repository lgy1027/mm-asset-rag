"""Packaging integrity tests — guard the published artifact's shape.

These don't run against installed code; they build the sdist/wheel locally
(via ``uv build``) and assert the artifact carries the right metadata, so a
release can't ship with a stale README (the v0.1.0 regression: tag was cut
before the "install from PyPI" README fix) or missing entry points.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


def _build(tmp_path: Path) -> Path:
    """Build sdist + wheel into tmp_path/dist; return the dist dir."""
    repo = Path(__file__).resolve().parents[2]
    dist = tmp_path / "dist"
    if not shutil.which("uv"):
        pytest.skip("uv not installed")
    subprocess.run(
        ["uv", "build", "--out-dir", str(dist)],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return dist


@pytest.fixture(scope="module")
def built_dist(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("pkg"))


def test_sdist_readme_has_pypi_install(built_dist: Path) -> None:
    """The sdist long-description (shown on the PyPI project page) must tell
    users to ``pip install mm-asset-rag`` — not the pre-release "install from
    source" wording. Catches the v0.1.0 regression where the tag was cut
    before the README fix landed."""
    (sdist,) = built_dist.glob("*.tar.gz")
    with tarfile.open(sdist) as t:
        # PKG-INFO lives at "<pkg>-<ver>/PKG-INFO".
        info = next(t.extractfile(n) for n in t.getnames() if n.endswith("/PKG-INFO"))
        info = info.read().decode("utf-8", "replace")
    body = info.split("\n\n", 1)[-1]
    assert "pip install mm-asset-rag" in body, "sdist README missing PyPI install line"
    assert "not yet published" not in body, "sdist README still has pre-release wording"


def test_wheel_has_entry_points(built_dist: Path) -> None:
    """The wheel must declare both console scripts so ``mmrag`` and
    ``mmrag-api`` land on PATH after ``pip install``."""
    (wheel,) = built_dist.glob("*.whl")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    entry = next(n for n in names if n.endswith("entry_points.txt"))
    text = zipfile.ZipFile(wheel).read(entry).decode()
    assert "mmrag = mm_asset_rag.cli:main" in text
    assert "mmrag-api = mm_asset_rag.api:run" in text


def test_wheel_includes_web_ui(built_dist: Path) -> None:
    """The bundled web UI (index.html) must be in the wheel — it's served by
    ``GET /`` and a missing file breaks the web UI silently."""
    (wheel,) = built_dist.glob("*.whl")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    assert any(n.endswith("web/index.html") for n in names), "wheel missing web/index.html"


def test_version_matches_source(built_dist: Path) -> None:
    """The built wheel's metadata version must equal the in-source __version__,
    so a release tag can't drift from what the code reports."""
    from mm_asset_rag import __version__

    (wheel,) = built_dist.glob("*.whl")
    with zipfile.ZipFile(wheel) as z:
        meta = next(n for n in z.namelist() if n.endswith("METADATA"))
        text = z.read(meta).decode()
    line = next(ln for ln in text.splitlines() if ln.startswith("Version:"))
    built = line.split(":", 1)[1].strip()
    assert built == __version__, f"wheel version {built} != source {__version__}"
    _ = sys  # silence linter on unused import in some configs
