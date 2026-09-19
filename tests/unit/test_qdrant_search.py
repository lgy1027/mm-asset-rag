"""Tests for encoder-fingerprint degradation in the qdrant image search routes.

A recorded fingerprint that the current provider fails to reproduce means
the index holds vectors from a different encoder; both image routes must
warn once per process and return ``[]`` (same contract as
embedder-unavailable). Stubs return synthetic vectors — no model downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mm_asset_rag.backends.qdrant import search as qdrant_search
from mm_asset_rag.embedders.fingerprint import (
    ImageEmbedderFingerprint,
    compute_image_fingerprint,
    save_image_fingerprint,
)


def _unit_vector(dim: int) -> list[float]:
    return [1.0] + [0.0] * (dim - 1)


class _StubImageEmbedder:
    """Deterministic stub: embeds any image to a fixed synthetic vector."""

    _model_name = "stub-model-a"

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector if vector is not None else _unit_vector(8)
        self.canary_embeds = 0  # embed_image invocations (canary + query image)

    def embed_image(self, image_path: Path) -> list[float] | None:
        self.canary_embeds += 1
        return list(self._vector)

    def embed_text(self, text: str) -> list[float]:
        return list(self._vector)


class _StubImageEmbedderB(_StubImageEmbedder):
    """Same surface, different concrete class name and model."""

    _model_name = "stub-model-b"


class _EmptyCanaryStub(_StubImageEmbedder):
    """Provider whose canary embed comes back empty (fingerprint match raises)."""

    def embed_image(self, image_path: Path) -> list[float] | None:
        self.canary_embeds += 1
        return []


@pytest.fixture(autouse=True)
def _reset_fingerprint_state():
    """Warn-once flag and memo are process state; isolate it per test."""
    qdrant_search._IMAGE_FINGERPRINT_MEMO.clear()
    qdrant_search._IMAGE_FINGERPRINT_WARNED = False
    yield
    qdrant_search._IMAGE_FINGERPRINT_MEMO.clear()
    qdrant_search._IMAGE_FINGERPRINT_WARNED = False


def _wire_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_qdrant_client,
    provider,
    recorded=None,
) -> None:
    """Point the fingerprint sidecar at ``tmp_path`` and stub the qdrant client."""
    fingerprint_path = tmp_path / "fp.json"
    if recorded is not None:
        save_image_fingerprint(fingerprint_path, recorded)
    monkeypatch.setattr(qdrant_search, "image_fingerprint_path", lambda: fingerprint_path)
    monkeypatch.setattr(qdrant_search, "get_qdrant_client", lambda: fake_qdrant_client)
    monkeypatch.setattr(qdrant_search, "image_collection", lambda dim: "multimodal_image_512d")
    monkeypatch.setattr(qdrant_search, "get_default_image_embedder", lambda: provider)


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_text_to_image_degrades_loudly_on_fingerprint_mismatch(
    monkeypatch, tmp_path, fake_qdrant_client, caplog
) -> None:
    provider = _StubImageEmbedder()
    _wire_search(
        monkeypatch,
        tmp_path,
        fake_qdrant_client,
        provider,
        recorded=compute_image_fingerprint(_StubImageEmbedderB()),
    )

    with caplog.at_level(logging.WARNING):
        assert qdrant_search.qdrant_text_to_image_search("a photo") == []
        assert qdrant_search.qdrant_text_to_image_search("another photo") == []

    warnings = _warnings(caplog)
    assert len(warnings) == 1  # one warning per process, not per query
    message = str(warnings[0].getMessage())
    assert "_StubImageEmbedderB" in message  # recorded encoder
    assert "force_recreate" in message  # how to rebuild
    assert fake_qdrant_client.query_points.call_count == 0  # never queried qdrant


def test_image_to_image_degrades_loudly_on_fingerprint_mismatch(
    monkeypatch, tmp_path, fake_qdrant_client, caplog
) -> None:
    provider = _StubImageEmbedder()
    _wire_search(
        monkeypatch,
        tmp_path,
        fake_qdrant_client,
        provider,
        recorded=compute_image_fingerprint(_StubImageEmbedderB()),
    )

    with caplog.at_level(logging.WARNING):
        assert qdrant_search.qdrant_image_to_image_search(Path("q.png")) == []
        assert qdrant_search.qdrant_image_to_image_search(Path("q.png")) == []

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "_StubImageEmbedderB" in str(warnings[0].getMessage())
    assert fake_qdrant_client.query_points.call_count == 0


def test_search_routes_query_normally_when_no_sidecar(
    monkeypatch, tmp_path, fake_qdrant_client, caplog
) -> None:
    """Fresh instance (index predates fingerprinting or was never fingerprinted
    at search time) keeps the previous behaviour: query, no warning."""
    provider = _StubImageEmbedder()
    _wire_search(monkeypatch, tmp_path, fake_qdrant_client, provider)

    with caplog.at_level(logging.WARNING):
        assert qdrant_search.qdrant_text_to_image_search("a photo") == []
        assert qdrant_search.qdrant_image_to_image_search(Path("q.png")) == []

    assert fake_qdrant_client.query_points.call_count == 2
    assert _warnings(caplog) == []


def test_search_routes_query_normally_when_sidecar_matches(
    monkeypatch, tmp_path, fake_qdrant_client, caplog
) -> None:
    provider = _StubImageEmbedder()
    _wire_search(
        monkeypatch,
        tmp_path,
        fake_qdrant_client,
        provider,
        recorded=compute_image_fingerprint(provider),
    )

    with caplog.at_level(logging.WARNING):
        assert qdrant_search.qdrant_text_to_image_search("a photo") == []
        assert qdrant_search.qdrant_image_to_image_search(Path("q.png")) == []

    assert fake_qdrant_client.query_points.call_count == 2
    assert _warnings(caplog) == []


def test_empty_canary_embedding_is_a_mismatch_not_a_crash(
    monkeypatch, tmp_path, fake_qdrant_client, caplog
) -> None:
    """``fingerprint_matches`` raises ValueError on an empty canary; the route
    must treat that as a loud mismatch, not propagate."""
    provider = _EmptyCanaryStub()
    recorded = ImageEmbedderFingerprint(
        provider="_EmptyCanaryStub", model=None, dim=8, canary=tuple(_unit_vector(8))
    )
    _wire_search(monkeypatch, tmp_path, fake_qdrant_client, provider, recorded=recorded)

    with caplog.at_level(logging.WARNING):
        assert qdrant_search.qdrant_text_to_image_search("a photo") == []

    assert _warnings(caplog)
    assert fake_qdrant_client.query_points.call_count == 0


def test_fingerprint_check_is_memoized_per_process(
    monkeypatch, tmp_path, fake_qdrant_client
) -> None:
    """Repeated mismatched queries re-embed the canary at most once."""
    provider = _StubImageEmbedder()
    _wire_search(
        monkeypatch,
        tmp_path,
        fake_qdrant_client,
        provider,
        recorded=compute_image_fingerprint(_StubImageEmbedderB()),
    )

    for _ in range(3):
        assert qdrant_search.qdrant_text_to_image_search("a photo") == []

    assert provider.canary_embeds == 1
