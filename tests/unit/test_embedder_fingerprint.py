"""Tests for ``mm_asset_rag.embedders.fingerprint``.

The encoder fingerprint catches the silent image-index/encoder mismatch
(clip vs cn_clip, same 512-dim collection, near-zero scores, no error).
Stubs here return synthetic fixed vectors — no model downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.embedders.fingerprint import (
    ImageEmbedderFingerprint,
    ImageEncoderMismatchError,
    compute_image_fingerprint,
    fingerprint_matches,
    image_fingerprint_path,
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
        self.seen_paths: list[Path] = []

    def embed_image(self, image_path: Path) -> list[float] | None:
        self.seen_paths.append(Path(image_path))
        return list(self._vector)


class _StubImageEmbedderB(_StubImageEmbedder):
    """Same surface, different concrete class name."""


class _StubNoModel(_StubImageEmbedder):
    """Provider with neither ``_model_name`` nor ``model_name``."""

    _model_name = None  # type: ignore[assignment]


# ── canary determinism ──────────────────────────────────────────────────────


def test_canary_is_deterministic_across_calls() -> None:
    first = compute_image_fingerprint(_StubImageEmbedder())
    second = compute_image_fingerprint(_StubImageEmbedder())
    assert first.canary == second.canary
    assert first == second


def test_canary_is_a_real_64px_png() -> None:
    captured: dict[str, object] = {}

    class _CapturingStub(_StubImageEmbedder):
        def embed_image(self, image_path: Path) -> list[float] | None:
            from PIL import Image

            with Image.open(image_path) as image:
                captured["size"] = image.size
            captured["png_magic"] = Path(image_path).read_bytes()[:8]
            return super().embed_image(image_path)

    compute_image_fingerprint(_CapturingStub())
    assert captured, "provider should be asked to embed the canary"
    assert captured["png_magic"] == b"\x89PNG\r\n\x1a\n"
    assert captured["size"] == (64, 64)


# ── fingerprint_matches ─────────────────────────────────────────────────────


def test_matches_identical_stub() -> None:
    recorded = compute_image_fingerprint(_StubImageEmbedder())
    assert fingerprint_matches(recorded, _StubImageEmbedder()) is True


def test_mismatch_when_provider_class_name_differs() -> None:
    recorded = compute_image_fingerprint(_StubImageEmbedder())
    assert fingerprint_matches(recorded, _StubImageEmbedderB()) is False


def test_mismatch_when_model_differs() -> None:
    recorded = compute_image_fingerprint(_StubImageEmbedder())
    current = _StubImageEmbedder()  # same class name, same canary vector
    current._model_name = "stub-model-b"
    assert fingerprint_matches(recorded, current) is False


def test_mismatch_when_dim_differs() -> None:
    recorded = compute_image_fingerprint(_StubImageEmbedder(_unit_vector(8)))
    assert fingerprint_matches(recorded, _StubImageEmbedder(_unit_vector(4))) is False


def test_mismatch_when_canary_perturbed_beyond_tolerance() -> None:
    perturbed = _unit_vector(8)
    perturbed[1] = 0.1  # cosine drops well below 1 - 1e-3
    recorded = compute_image_fingerprint(_StubImageEmbedder(_unit_vector(8)))
    assert fingerprint_matches(recorded, _StubImageEmbedder(perturbed)) is False


def test_matches_when_canary_perturbed_within_tolerance() -> None:
    perturbed = _unit_vector(8)
    perturbed[1] = 1e-4  # cosine ~= 1 - 5e-9, within 1 - 1e-3
    recorded = compute_image_fingerprint(_StubImageEmbedder(_unit_vector(8)))
    assert fingerprint_matches(recorded, _StubImageEmbedder(perturbed)) is True


def test_matches_when_both_model_names_absent() -> None:
    recorded = compute_image_fingerprint(_StubNoModel())
    assert recorded.model is None
    assert fingerprint_matches(recorded, _StubNoModel()) is True


# ── model-name probe ────────────────────────────────────────────────────────


def test_model_probed_from_model_name_attr() -> None:
    class _StubPublicModelName:
        model_name = "stub-public-model"

        def embed_image(self, image_path: Path) -> list[float] | None:
            return _unit_vector(4)

    fp = compute_image_fingerprint(_StubPublicModelName())
    assert fp.model == "stub-public-model"


# ── sidecar round-trip ──────────────────────────────────────────────────────


def test_sidecar_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "image_embedder_fingerprint.json"
    fp = compute_image_fingerprint(_StubImageEmbedder())
    save_image_fingerprint(path, fp)
    assert load_image_fingerprint(path) == fp


def test_load_missing_sidecar_returns_none(tmp_path: Path) -> None:
    assert load_image_fingerprint(tmp_path / "nope.json") is None


# ── misc contract ───────────────────────────────────────────────────────────


def test_mismatch_error_is_a_value_error() -> None:
    assert issubclass(ImageEncoderMismatchError, ValueError)


def test_fingerprint_is_frozen() -> None:
    fp = compute_image_fingerprint(_StubImageEmbedder())
    with pytest.raises(AttributeError):
        fp.provider = "other"  # type: ignore[misc]


def test_image_fingerprint_path_lives_under_indexes_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(tmp_path))
    from mm_asset_rag.core.paths import get_indexes_dir

    assert image_fingerprint_path() == get_indexes_dir() / "image_embedder_fingerprint.json"
    assert ImageEmbedderFingerprint.__dataclass_params__.frozen is True
