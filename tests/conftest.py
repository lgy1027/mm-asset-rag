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
def _isolate_env_file(monkeypatch):
    """Keep the project-root ``.env`` out of the test process.

    Two channels load ``.env`` and both must be blocked, or a developer's
    local ``.env`` (git-ignored, so CI is fine) overrides code defaults
    (``AUTO_META_ENABLED`` / ``RERANKER_ENABLED`` / ``ENRICH_CHUNK_WITH_KEYWORDS``
    / ``PDF_EXTRACT_IMAGES`` / ``CONTEXTUAL_ENABLED``) and turns tests that
    assert those defaults red:

    1. :func:`mm_asset_rag.config.load_env` calls ``python-dotenv``'s
       ``load_dotenv()``, which *populates* ``os.environ`` from ``.env``.
       ``get_service()`` calls ``load_env()`` on first use, so any test
       that touches the ingest service leaks the ``.env`` values into the
       process env for every later test.
    2. pydantic-settings' ``Settings.model_config["env_file"] == ".env"``
       re-reads the file on each ``get_settings()`` rebuild.

    We block both: no-op ``load_env`` so ``os.environ`` is never seeded
    from ``.env``, and ``env_file=None`` so pydantic doesn't re-read it.
    The ``_service`` singleton is also reset so the next ``get_service()``
    rebuilds under the isolated env (it may have been created earlier with
    ``.env``-tainted settings). Tests that genuinely want a setting
    override set the env var explicitly via ``monkeypatch.setenv``.
    """
    from mm_asset_rag import config
    from mm_asset_rag.settings import Settings, get_settings

    # Patch the underlying load_dotenv on the config module so every
    # ``load_env()`` caller — including ``service.get_service()`` which
    # captured ``load_env`` at import time (``from .config import
    # load_env``) and would otherwise bypass a ``config.load_env``
    # monkeypatch — becomes a no-op. This stops ``.env`` values from
    # being seeded into ``os.environ`` for the rest of the process.
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(config, "load_env", lambda: None)
    original_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    # Drop a process-wide IngestService built under any prior (possibly
    # .env-tainted) settings so the next get_service() rebuilds clean.
    import mm_asset_rag.service as service_mod

    monkeypatch.setattr(service_mod, "_service", None)
    yield
    Settings.model_config["env_file"] = original_env_file
    get_settings.cache_clear()


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
