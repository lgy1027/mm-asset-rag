"""Shared pytest fixtures.

All tests in this suite are designed to run offline. External services
(OpenAI, Qdrant server, sentence-transformers downloads) are mocked.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_ASSETS_DIR = FIXTURES_DIR / "sample_assets"
SAMPLE_MANIFEST = FIXTURES_DIR / "sample_manifest.json"


@pytest.fixture
def tmp_home(tmp_path, monkeypatch) -> Path:
    """Point MM_ASSET_RAG_HOME at a fresh tmp directory."""
    home = tmp_path / "mm_asset_rag_home"
    home.mkdir()
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(home))
    return home


@pytest.fixture
def populated_home(tmp_home) -> Path:
    """Copy sample assets + manifest into the tmp data directory."""
    assets_dir = tmp_home / "assets"
    assets_dir.mkdir()
    for src in SAMPLE_ASSETS_DIR.iterdir():
        shutil.copy2(src, assets_dir / src.name)
    shutil.copy2(SAMPLE_MANIFEST, assets_dir / "asset_manifest.json")
    return tmp_home


@pytest.fixture
def fake_qdrant_client(monkeypatch) -> MagicMock:
    """Replace QdrantClient with a MagicMock that returns empty results.

    Tests that need richer behaviour can configure the mock explicitly.
    """
    mock = MagicMock()
    mock.collection_exists.return_value = False
    mock.query_points.return_value = MagicMock(points=[])
    mock.scroll.return_value = ([], None)
    monkeypatch.setattr("mm_asset_rag.qdrant_store.QdrantClient", lambda *a, **kw: mock)
    return mock


@pytest.fixture
def fixed_vector(monkeypatch) -> None:
    """Pin the embedding providers to a fixed 4-dim mock vector.

    Replaces both EmbeddingProvider.embed_text and ImageEmbeddingProvider
    methods with deterministic outputs so test assertions can match exactly.
    """
    import mm_asset_rag.providers as providers

    def _text(self, text: str) -> list[float]:
        digest = sum(ord(c) for c in text) % 100
        return [float(digest), 0.1, 0.2, 0.3]

    def _image(self, path: Path) -> list[float]:
        return [0.4, 0.5, 0.6, 0.7]

    monkeypatch.setattr(providers.EmbeddingProvider, "embed_text", _text)
    monkeypatch.setattr(
        providers.EmbeddingProvider,
        "embed_texts",
        lambda self, texts: [_text(self, t) for t in texts],
    )
    monkeypatch.setattr(providers.ImageEmbeddingProvider, "embed_text", _text)
    monkeypatch.setattr(providers.ImageEmbeddingProvider, "embed_image", _image)
