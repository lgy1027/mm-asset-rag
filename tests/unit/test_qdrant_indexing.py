"""Tests for the encoder-fingerprint checks in ``build_qdrant_image_index``.

The fingerprint sidecar records which encoder produced the image index;
ingest must refuse to mix vector spaces (fail fast before writing) and
record provenance after a successful write. Stubs return synthetic
vectors — no model downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.backends.qdrant import indexing as qdrant_indexing
from mm_asset_rag.core.knowledge_models import AccessPolicy, Asset, Chunk, Document, Source
from mm_asset_rag.embedders.fingerprint import (
    ImageEncoderMismatchError,
    compute_image_fingerprint,
    load_image_fingerprint,
    save_image_fingerprint,
)


def _unit_vector(dim: int) -> list[float]:
    """Deterministic synthetic unit vector (first component 1.0)."""
    return [1.0] + [0.0] * (dim - 1)


class _StubImageEmbedder:
    """Deterministic stub: embeds any image to a fixed synthetic vector."""

    _model_name = "stub-model-a"

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector if vector is not None else _unit_vector(8)
        self.embed_calls = 0

    def embed_image(self, image_path: Path) -> list[float] | None:
        self.embed_calls += 1
        return list(self._vector)

    def embed_image_batch(self, paths: list[Path]) -> list[list[float]]:
        return [list(self._vector) for _ in paths]


class _StubImageEmbedderB(_StubImageEmbedder):
    """Same surface, different concrete class name and model."""

    _model_name = "stub-model-b"


def _image_chunk() -> Chunk:
    source = Source(source_id="upload:photo", uri="uploads/photo.png")
    policy = AccessPolicy(collection="engineering", allowed_principals=())
    document = Document("photo", "Photo", source, policy)
    asset = Asset("b" * 64, "image", "images/photo.png")
    return Chunk.create(
        document=document,
        asset=asset,
        ordinal=0,
        text="a photo",
        source=source,
        access_policy=policy,
    )


def _wire_image_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_qdrant_client,
    provider,
    fingerprint_path: Path,
) -> None:
    """Stub everything external so ``build_qdrant_image_index`` runs in-process."""
    monkeypatch.setattr(qdrant_indexing, "read_documents", lambda: [_image_chunk()])
    monkeypatch.setattr(qdrant_indexing, "get_default_image_embedder", lambda: provider)
    monkeypatch.setattr(qdrant_indexing, "get_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(qdrant_indexing, "get_qdrant_client", lambda: fake_qdrant_client)
    monkeypatch.setattr(qdrant_indexing, "image_collection", lambda dim: "multimodal_image_512d")
    monkeypatch.setattr(qdrant_indexing, "_create_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(qdrant_indexing, "_ensure_payload_indexes", lambda *args, **kwargs: None)
    monkeypatch.setattr(qdrant_indexing, "image_fingerprint_path", lambda: fingerprint_path)


def test_build_image_index_raises_on_fingerprint_mismatch_before_writing(
    monkeypatch, tmp_path, fake_qdrant_client
) -> None:
    """A sidecar from another encoder aborts ingest before any point is written."""
    fingerprint_path = tmp_path / "fp.json"
    save_image_fingerprint(fingerprint_path, compute_image_fingerprint(_StubImageEmbedderB()))
    provider = _StubImageEmbedder()
    _wire_image_build(monkeypatch, tmp_path, fake_qdrant_client, provider, fingerprint_path)

    with pytest.raises(ImageEncoderMismatchError) as excinfo:
        qdrant_indexing.build_qdrant_image_index()

    message = str(excinfo.value)
    assert "_StubImageEmbedderB" in message  # recorded encoder
    assert "_StubImageEmbedder" in message  # current encoder
    assert "stub-model-b" in message  # recorded model
    assert "force_recreate" in message  # rebuild instructions
    assert fake_qdrant_client.upsert.call_count == 0  # nothing written
    # Only the fingerprint canary was embedded — no corpus image was embedded.
    assert provider.embed_calls == 1


def test_build_image_index_raises_when_canary_perturbed_beyond_tolerance(
    monkeypatch, tmp_path, fake_qdrant_client
) -> None:
    """Same provider class with different weights (canary drift) also refuses."""
    fingerprint_path = tmp_path / "fp.json"
    save_image_fingerprint(fingerprint_path, compute_image_fingerprint(_StubImageEmbedder()))
    perturbed = _unit_vector(8)
    perturbed[1] = 0.1  # cosine drops well below 1 - 1e-3
    provider = _StubImageEmbedder(perturbed)
    _wire_image_build(monkeypatch, tmp_path, fake_qdrant_client, provider, fingerprint_path)

    with pytest.raises(ImageEncoderMismatchError, match="_StubImageEmbedder"):
        qdrant_indexing.build_qdrant_image_index()

    assert fake_qdrant_client.upsert.call_count == 0


def test_build_image_index_saves_fingerprint_after_successful_write(
    monkeypatch, tmp_path, fake_qdrant_client
) -> None:
    """No sidecar yet → ingest writes points then records provenance."""
    fingerprint_path = tmp_path / "fp.json"
    provider = _StubImageEmbedder()
    _wire_image_build(monkeypatch, tmp_path, fake_qdrant_client, provider, fingerprint_path)

    inserted, _summary = qdrant_indexing.build_qdrant_image_index()

    assert inserted == 1
    assert fake_qdrant_client.upsert.call_count == 1
    recorded = load_image_fingerprint(fingerprint_path)
    assert recorded is not None
    assert recorded == compute_image_fingerprint(provider)


def test_build_image_index_force_recreate_replaces_sidecar(
    monkeypatch, tmp_path, fake_qdrant_client
) -> None:
    """``force_recreate=True`` rebuilds from scratch and overwrites the sidecar."""
    fingerprint_path = tmp_path / "fp.json"
    save_image_fingerprint(fingerprint_path, compute_image_fingerprint(_StubImageEmbedderB()))
    provider = _StubImageEmbedder()
    _wire_image_build(monkeypatch, tmp_path, fake_qdrant_client, provider, fingerprint_path)

    inserted, _summary = qdrant_indexing.build_qdrant_image_index(force_recreate=True)

    assert inserted == 1
    recorded = load_image_fingerprint(fingerprint_path)
    assert recorded is not None
    assert recorded.provider == "_StubImageEmbedder"
    assert recorded == compute_image_fingerprint(provider)


def test_build_image_index_proceeds_when_sidecar_matches(
    monkeypatch, tmp_path, fake_qdrant_client
) -> None:
    """Matching sidecar → normal incremental ingest, nothing raises."""
    fingerprint_path = tmp_path / "fp.json"
    provider = _StubImageEmbedder()
    save_image_fingerprint(fingerprint_path, compute_image_fingerprint(provider))
    _wire_image_build(monkeypatch, tmp_path, fake_qdrant_client, provider, fingerprint_path)

    inserted, _summary = qdrant_indexing.build_qdrant_image_index()

    assert inserted == 1
    assert fake_qdrant_client.upsert.call_count == 1
