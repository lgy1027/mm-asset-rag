"""Shared pytest fixtures.

All tests use the real ``examples/data/chapter11_assets/`` sample set
that ships with the repository. There are no mock embeddings or fake
fixtures — the test suite exercises the same paths as production.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DATA_DIR = REPO_ROOT / "examples" / "data" / "chapter11_assets"


def _have_examples() -> bool:
    return EXAMPLES_DATA_DIR.is_dir() and (EXAMPLES_DATA_DIR / "asset_manifest.json").is_file()


@pytest.fixture
def tmp_home(tmp_path, monkeypatch) -> Path:
    """Point MM_ASSET_RAG_HOME at a fresh tmp directory (no manifest/seed)."""
    home = tmp_path / "mm_asset_rag_home"
    home.mkdir()
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(home))
    return home


@pytest.fixture
def examples_home(tmp_path, monkeypatch) -> Path:
    """Copy ``examples/data/chapter11_assets`` into a tmp home and point at it.

    Each test gets a fresh copy so file mutations (cached parsed output,
    Qdrant local files, captions) don't leak across tests. The fixture
    is skipped automatically if the bundled sample data is missing.
    """
    if not _have_examples():
        pytest.skip(f"sample data not found at {EXAMPLES_DATA_DIR}")

    home = tmp_path / "mm_asset_rag_home"
    assets = home / "assets"
    assets.mkdir(parents=True)
    shutil.copytree(EXAMPLES_DATA_DIR, assets, dirs_exist_ok=True)
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(home))
    return home


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


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Drop the ``Settings`` singleton around each test.

    Several tests set ``MM_ASSET_RAG_HOME`` (or other env vars) in a
    fixture. Because :func:`get_settings` is ``lru_cache``-wrapped, the
    first call freezes whatever env was visible at that point. Clearing
    the cache around each test means the next ``get_settings()`` call
    rebuilds from the current env, including monkeypatch overrides.
    """
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fixed_vector(monkeypatch) -> None:
    """Pin both embedding providers to a fixed deterministic vector.

    Lets the merge / normalize logic in ``retrieval`` be exercised without
    hitting a real embedding backend.
    """
    import mm_asset_rag.embedders.text_embedder as text_emb
    import mm_asset_rag.embedders.image_embedder as image_emb

    def _text(self, text: str) -> list[float]:
        digest = sum(ord(c) for c in text) % 100
        return [float(digest), 0.1, 0.2, 0.3]

    def _image(self, _path: Path) -> list[float]:
        return [0.4, 0.5, 0.6, 0.7]

    monkeypatch.setattr(
        text_emb.TextEmbedder,
        "embed",
        lambda self, content: _text(self, str(content)),
    )
    monkeypatch.setattr(
        text_emb.TextEmbedder,
        "embed_batch",
        lambda self, contents: [_text(self, str(c)) for c in contents],
    )
    monkeypatch.setattr(
        image_emb.ImageEmbedder, "embed_text", _text
    )  # ImageEmbedder keeps embed_text / embed_image methods
    monkeypatch.setattr(image_emb.ImageEmbedder, "embed_image", _image)
