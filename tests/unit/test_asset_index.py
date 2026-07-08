"""Tests for ``mm_asset_rag.asset_index``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.asset_index import (
    AssetIndexEntry,
    find_active_by_asset_id,
    find_by_semantic,
    find_by_sha256,
    latest_by_asset_id,
    list_active,
    load_entries,
    mark_deleted,
    record_asset_embedding,
    upsert_entry,
)


def _write_entry(path: Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields) + "\n")


def test_load_entries_tolerates_corrupt_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text("\nnot json\n", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "asset_id": "a",
                "sha256": "h1",
                "source_type": "image",
                "relative_path": "images/a.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entries = load_entries(path)
    assert len(entries) == 1
    assert entries[0].asset_id == "a"


def test_load_entries_missing_file(tmp_path: Path) -> None:
    assert load_entries(tmp_path / "missing.jsonl") == []


def test_upsert_and_find(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a", sha256="hash", source_type="image", relative_path="images/a.png"
        ),
        path=path,
    )
    assert find_by_sha256("hash", path=path).asset_id == "a"
    assert find_active_by_asset_id("a", path=path).asset_id == "a"


def test_mark_deleted_toggles_state(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a", sha256="hash", source_type="image", relative_path="images/a.png"
        ),
        path=path,
    )
    assert mark_deleted("a", path=path, at=42.0) is True
    # Already deleted: further mark_deleted is a no-op.
    assert mark_deleted("a", path=path, at=43.0) is False
    assert find_by_sha256("hash", path=path) is None
    assert find_active_by_asset_id("a", path=path) is None


def test_latest_by_asset_id_folds_history(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    _write_entry(
        path,
        asset_id="a",
        sha256="h1",
        source_type="image",
        relative_path="images/a.png",
    )
    _write_entry(
        path,
        asset_id="a",
        sha256="h2",
        source_type="image",
        relative_path="images/a2.png",
    )
    latest = latest_by_asset_id(path=path)
    assert latest["a"].sha256 == "h2"


def test_list_active_skips_deleted(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a", sha256="h1", source_type="image", relative_path="images/a.png"
        ),
        path=path,
    )
    upsert_entry(
        AssetIndexEntry(
            asset_id="b", sha256="h2", source_type="image", relative_path="images/b.png"
        ),
        path=path,
    )
    mark_deleted("a", path=path, at=1.0)
    active = list_active(path=path)
    assert [e.asset_id for e in active] == ["b"]


def test_entry_round_trip_preserves_tags(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a",
            sha256="hash",
            source_type="image",
            relative_path="images/a.png",
            tags=["beach", "sunset"],
        ),
        path=path,
    )
    entries = load_entries(path)
    assert entries[0].tags == ["beach", "sunset"]


def test_entry_loads_default_tags_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text(
        __import__("json").dumps(
            {
                "asset_id": "legacy",
                "sha256": "h",
                "source_type": "image",
                "relative_path": "images/legacy.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entries = load_entries(path)
    assert entries[0].tags == []


# ── Semantic dedup (LlamaIndex-style) ────────────────────────────────────


def test_find_by_semantic_returns_none_when_empty(tmp_path: Path) -> None:
    assert find_by_semantic([], embeddings_path=tmp_path / "emb.jsonl") is None
    assert find_by_semantic([0.1, 0.2], embeddings_path=tmp_path / "missing.jsonl") is None


def test_record_and_find_by_semantic(tmp_path: Path) -> None:
    emb_path = tmp_path / "emb.jsonl"
    record_asset_embedding(
        "Alexnet_0c1c2b23",
        "Alexnet",
        [1.0, 0.0, 0.0],
        model="test",
        embeddings_path=emb_path,
    )
    # Near-identical direction → cosine ~1.0 > default 0.92.
    found = find_by_semantic([0.99, 0.0, 0.0], embeddings_path=emb_path)
    assert found == "Alexnet_0c1c2b23"
    # Orthogonal vector → cosine 0, below threshold.
    assert find_by_semantic([0.0, 1.0, 0.0], embeddings_path=emb_path) is None


def test_find_by_semantic_respects_custom_threshold(tmp_path: Path) -> None:
    emb_path = tmp_path / "emb.jsonl"
    record_asset_embedding("a_hash1", "a", [1.0, 0.0], embeddings_path=emb_path)
    # cosine ~0.707 between [1,0] and [1,1].
    assert find_by_semantic([1.0, 1.0], threshold=0.6, embeddings_path=emb_path) == "a_hash1"
    assert find_by_semantic([1.0, 1.0], threshold=0.9, embeddings_path=emb_path) is None


def test_find_by_semantic_excludes_own_sha(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    emb_path = tmp_path / "emb.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a_11111111",
            sha256="sha_a",
            source_type="pdf",
            relative_path="pdfs/a.pdf",
            asset_title="a",
        ),
        path=path,
        semantic_embedding=[1.0, 0.0],
        embeddings_path=emb_path,
    )
    # Same content sha already indexed — exclude_sha256 should skip it.
    found = find_by_semantic(
        [1.0, 0.0],
        exclude_sha256="sha_a",
        path=path,
        embeddings_path=emb_path,
    )
    assert found is None


def test_upsert_entry_semantic_dedup_reuses_existing(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    emb_path = tmp_path / "emb.jsonl"
    first = upsert_entry(
        AssetIndexEntry(
            asset_id="Alexnet_0c1c2b23",
            sha256="sha_a",
            source_type="pdf",
            relative_path="pdfs/a.pdf",
            asset_title="Alexnet",
        ),
        path=path,
        semantic_embedding=[1.0, 0.0, 0.0],
        semantic_model="test",
        embeddings_path=emb_path,
    )
    assert first == "Alexnet_0c1c2b23"
    # Near-duplicate content (different sha, near-identical embedding)
    # should be detected as a semantic duplicate and reuse the existing id.
    reused = upsert_entry(
        AssetIndexEntry(
            asset_id="Alexnet_copy_22222222",
            sha256="sha_b",
            source_type="pdf",
            relative_path="pdfs/a_copy.pdf",
            asset_title="Alexnet",
        ),
        path=path,
        semantic_embedding=[0.99, 0.0, 0.0],
        embeddings_path=emb_path,
    )
    assert reused == "Alexnet_0c1c2b23"
    # The duplicate row was *not* written to the index.
    entries = load_entries(path)
    assert all(e.asset_id == "Alexnet_0c1c2b23" for e in entries)
    assert len(entries) == 1


def test_upsert_entry_backward_compatible_without_semantic(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    emb_path = tmp_path / "emb.jsonl"
    # No semantic_embedding → behaves exactly like the legacy call:
    # writes the row, no embedding recorded.
    returned = upsert_entry(
        AssetIndexEntry(
            asset_id="plain_33333333",
            sha256="sha_p",
            source_type="image",
            relative_path="images/p.png",
        ),
        path=path,
    )
    assert returned == "plain_33333333"
    assert find_by_sha256("sha_p", path=path).asset_id == "plain_33333333"
    assert not emb_path.exists()


def test_mark_deleted_tombstones_embedding(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    emb_path = tmp_path / "emb.jsonl"
    upsert_entry(
        AssetIndexEntry(
            asset_id="a_44444444",
            sha256="sha_a",
            source_type="pdf",
            relative_path="pdfs/a.pdf",
            asset_title="a",
        ),
        path=path,
        semantic_embedding=[1.0, 0.0],
        embeddings_path=emb_path,
    )
    assert find_by_semantic([1.0, 0.0], embeddings_path=emb_path) == "a_44444444"
    assert mark_deleted("a_44444444", path=path, embeddings_path=emb_path, at=99.0) is True
    # Tombstone must shadow the embedding — no semantic match after delete.
    assert find_by_semantic([1.0, 0.0], embeddings_path=emb_path) is None


def test_dedup_threshold_env_override(monkeypatch, tmp_path: Path) -> None:
    from mm_asset_rag.settings import get_settings

    emb_path = tmp_path / "emb.jsonl"
    record_asset_embedding("a_55555555", "a", [1.0, 0.0], embeddings_path=emb_path)
    # cosine([1,0],[1,1]) ≈ 0.707. Default 0.92 → miss.
    assert find_by_semantic([1.0, 1.0], embeddings_path=emb_path) is None
    # Lower threshold via env → hit. ``get_settings`` is lru_cached, so the
    # cache must be cleared for the new env value to take effect mid-test.
    monkeypatch.setenv("DEDUP_SEMANTIC_THRESHOLD", "0.5")
    get_settings.cache_clear()
    assert find_by_semantic([1.0, 1.0], embeddings_path=emb_path) == "a_55555555"


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    """Vectors of different dimension return 0.0 rather than a truncated dot
    product, so an embedding-model change (dim change) between ingests cannot
    misclassify an asset as a near-duplicate via a meaningless short-vector
    projection."""
    from mm_asset_rag.asset_index import _cosine_similarity

    assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
