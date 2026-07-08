"""Tests for ``_create_collection`` schema-mismatch detection.

These verify the fail-fast path added so that upgrading a Settings flag
(``bm25_zh_enabled``) without reindexing surfaces a clear error instead
of silently writing partial vectors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mm_asset_rag.backends import qdrant_backend as qb
from mm_asset_rag.settings import Settings


def _make_client(
    *,
    exists: bool,
    existing_sparse_names: list[str] | None,
    existing_vector_names: list[str] | None = None,
) -> MagicMock:
    """Build a mock QdrantClient that reports the given vector config.

    ``existing_vector_names`` lists the named dense / multi-vectors the
    collection carries (e.g. ``["dense"]``). Defaults to ``["dense"]`` so
    the schema check sees the standard dense field.
    """
    client = MagicMock()
    client.collection_exists.return_value = exists
    if exists and existing_sparse_names is not None:
        info = MagicMock()
        info.config.params.sparse_vectors = {name: MagicMock() for name in existing_sparse_names}
        info.config.params.vectors = {
            name: MagicMock() for name in (existing_vector_names or ["dense"])
        }
        client.get_collection.return_value = info
    return client


def _settings(**overrides) -> Settings:
    """Fresh Settings with no env-var contamination, plus overrides."""
    base = Settings(_env_file=None, bm25_zh_enabled=True)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_create_collection_matches_existing_schema(monkeypatch) -> None:
    """No-op when collection already has the expected sparse vectors."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings())
    client = _make_client(
        exists=True,
        existing_sparse_names=[qb.SPARSE_VECTOR_NAME, qb._settings().bm25_zh_vector_name]
        if False
        else [qb.SPARSE_VECTOR_NAME, "bm25_zh"],
    )
    # No raise.
    qb._create_collection(client, "text", vector_size=2560, sparse=True)
    client.create_collection.assert_not_called()


def test_create_collection_fails_when_bm25_zh_missing(monkeypatch) -> None:
    """Old collection (only `bm25`) fails fast when settings expect `bm25_zh`."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings(bm25_zh_enabled=True))
    client = _make_client(
        exists=True,
        existing_sparse_names=[qb.SPARSE_VECTOR_NAME],  # no bm25_zh
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        qb._create_collection(client, "text", vector_size=2560, sparse=True)
    client.create_collection.assert_not_called()


def test_create_collection_fails_when_bm25_zh_unexpected(monkeypatch) -> None:
    """Settings disabled bm25_zh but the collection still has it — also a mismatch."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings(bm25_zh_enabled=False))
    client = _make_client(
        exists=True,
        existing_sparse_names=[qb.SPARSE_VECTOR_NAME, "bm25_zh"],  # leftover
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        qb._create_collection(client, "text", vector_size=2560, sparse=True)


def test_create_collection_skips_check_when_not_sparse(monkeypatch) -> None:
    """Image collection (dense only) never goes through the sparse check."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings())
    client = _make_client(exists=True, existing_sparse_names=None)
    qb._create_collection(client, "image", vector_size=512)
    # No raise; no schema inspection needed for a dense-only collection.
    client.get_collection.assert_not_called()


def test_create_collection_recreate_path_drops_then_rebuilds(monkeypatch) -> None:
    """``recreate=True`` short-circuits the schema check (it drops first)."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings(bm25_zh_enabled=True))
    client = MagicMock()
    client.collection_exists.return_value = True
    qb._create_collection(client, "text", vector_size=2560, sparse=True, recreate=True)
    client.delete_collection.assert_called_once()
    client.get_collection.assert_not_called()  # no schema check needed


# ─── embed_sparse / embed_colbert schema checks ───────────────────────────


def test_create_collection_matches_when_embed_sparse_enabled(monkeypatch) -> None:
    """When ``embed_sparse=True`` the existing collection must carry ``embed_sparse``."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings())
    client = _make_client(
        exists=True,
        existing_sparse_names=[qb.SPARSE_VECTOR_NAME, "bm25_zh", qb.EMBED_SPARSE_VECTOR_NAME],
    )
    # No raise.
    qb._create_collection(client, "text", vector_size=1024, sparse=True, embed_sparse=True)
    client.create_collection.assert_not_called()


def test_create_collection_fails_when_embed_sparse_missing(monkeypatch) -> None:
    """Collection built without ``embed_sparse`` fails fast when settings expect it."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings())
    client = _make_client(
        exists=True,
        existing_sparse_names=[qb.SPARSE_VECTOR_NAME, "bm25_zh"],  # no embed_sparse
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        qb._create_collection(client, "text", vector_size=1024, sparse=True, embed_sparse=True)


def test_create_collection_matches_when_embed_colbert_enabled(monkeypatch) -> None:
    """When ``embed_colbert=True`` the existing collection must carry ``embed_colbert``."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings())
    client = _make_client(
        exists=True,
        existing_sparse_names=[qb.SPARSE_VECTOR_NAME, "bm25_zh"],
        existing_vector_names=["dense", qb.EMBED_COLBERT_VECTOR_NAME],
    )
    # No raise.
    qb._create_collection(
        client,
        "text",
        vector_size=1024,
        sparse=True,
        embed_colbert=True,
        colbert_dim=1024,
    )
    client.create_collection.assert_not_called()


def test_create_collection_fails_when_embed_colbert_missing(monkeypatch) -> None:
    """Collection built without ``embed_colbert`` fails fast when settings expect it."""
    monkeypatch.setattr(qb, "get_settings", lambda: _settings())
    client = _make_client(
        exists=True,
        existing_sparse_names=[qb.SPARSE_VECTOR_NAME, "bm25_zh"],
        existing_vector_names=["dense"],  # no embed_colbert
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        qb._create_collection(
            client,
            "text",
            vector_size=1024,
            sparse=True,
            embed_colbert=True,
            colbert_dim=1024,
        )
