"""Shared pytest fixtures.

The project no longer ships a manifest-backed sample corpus. Unit tests
build the assets they need inside ``tmp_home`` and keep embedding / Qdrant
calls mocked unless a test explicitly opts into them.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_home(tmp_path, monkeypatch) -> Path:
    """Point MM_ASSET_RAG_HOME at a fresh tmp directory."""
    home = tmp_path / "mm_asset_rag_home"
    home.mkdir()
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(home))
    # Unit tests should never accidentally hit a real VLM endpoint.
    monkeypatch.setenv("AUTO_META_ENABLED", "false")
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
    monkeypatch.setattr("mm_asset_rag.backends.qdrant_backend.QdrantClient", lambda *a, **kw: mock)
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
    import mm_asset_rag.embedders.image_embedder as image_emb
    import mm_asset_rag.embedders.text_embedder as text_emb

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
    monkeypatch.setattr(image_emb.ImageEmbedder, "embed_text", _text)
    monkeypatch.setattr(image_emb.ImageEmbedder, "embed_image", _image)
