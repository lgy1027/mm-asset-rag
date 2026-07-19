"""Tests for the per-channel RRF weight plumbing in ``_hybrid_text_query``.

The earlier ``_channel_limit`` workaround (scale prefetch limit
proportional to weight) is gone — Qdrant server 1.17+ accepts
``RrfQuery(rrf=Rrf(weights=[...]))`` natively. These tests pin
the dispatch logic: when all weights are 1.0, we use the simpler
``FusionQuery(fusion=Fusion.RRF)``; otherwise we use the weighted
``RrfQuery`` with positional weights.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from mm_asset_rag.backends import qdrant_backend
from mm_asset_rag.backends.qdrant_backend import RRF_K
from mm_asset_rag.settings import Settings


class _StubClient:
    """Minimal QdrantClient stand-in that records the last ``query_points`` call."""

    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def query_points(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        return type("R", (), {"points": []})()


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(**overrides) -> Settings:
    s = Settings(_env_file=None)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _sv(indices=(0,), values=(1.0,)) -> models.SparseVector:
    return models.SparseVector(indices=list(indices), values=list(values))


def test_uniform_weights_use_fusion_query(monkeypatch) -> None:
    """All 1.0 weights → ``FusionQuery(Fusion.RRF)`` so server defaults apply."""
    s = _settings()
    s.rrf_weight_dense = 1.0
    s.rrf_weight_bm25 = 1.0
    s.rrf_weight_bm25_zh = 1.0
    monkeypatch.setattr(qdrant_backend, "get_settings", lambda: s)

    client = _StubClient()
    qdrant_backend._hybrid_text_query(client, "coll", [0.1, 0.2], _sv(), None, top_k=5)
    assert isinstance(client.last_kwargs["query"], models.FusionQuery)


def test_non_uniform_weights_use_rrf_query(monkeypatch) -> None:
    """A non-1.0 weight triggers ``RrfQuery(rrf=Rrf(weights=[...]))``.

    No Chinese sparse vector is supplied, so the zh prefetch is absent —
    the weights list must therefore have one entry per *actual* prefetch
    (dense + bm25_en = 2), not a phantom third slot for a channel that
    isn't in the prefetch list.
    """
    s = _settings()
    s.rrf_weight_dense = 1.5
    s.rrf_weight_bm25 = 1.0
    s.rrf_weight_bm25_zh = 1.0
    monkeypatch.setattr(qdrant_backend, "get_settings", lambda: s)

    client = _StubClient()
    qdrant_backend._hybrid_text_query(client, "coll", [0.1, 0.2], _sv(), None, top_k=5)
    q = client.last_kwargs["query"]
    assert isinstance(q, models.RrfQuery)
    assert q.rrf.weights == [1.5, 1.0]
    assert q.rrf.k == RRF_K


def test_weights_length_matches_prefetch_count(monkeypatch) -> None:
    """Weights list length must equal the number of prefetches, always.

    Regression guard: previously the bm25_zh weight occupied a slot
    unconditionally even when its prefetch was skipped, producing a
    length mismatch (3 weights, 2 prefetches) that Qdrant would misapply.
    """
    s = _settings()
    s.rrf_weight_dense = 1.5  # non-uniform → forces RrfQuery path
    s.rrf_weight_bm25 = 1.0
    s.rrf_weight_bm25_zh = 2.0  # ignored: no zh prefetch in this call
    monkeypatch.setattr(qdrant_backend, "get_settings", lambda: s)

    client = _StubClient()
    qdrant_backend._hybrid_text_query(client, "coll", [0.1, 0.2], _sv(), None, top_k=5)
    q = client.last_kwargs["query"]
    assert isinstance(q, models.RrfQuery)
    # Only dense + bm25_en prefetches exist → exactly 2 weights.
    assert q.rrf.weights == [1.5, 1.0]
    assert len(q.rrf.weights) == len(client.last_kwargs["prefetch"])


def test_weights_are_positional_to_prefetches(monkeypatch) -> None:
    """[dense, bm25, bm25_zh] order — matches the prefetch list order."""
    s = _settings()
    s.rrf_weight_dense = 0.7
    s.rrf_weight_bm25 = 1.2
    s.rrf_weight_bm25_zh = 2.0
    monkeypatch.setattr(qdrant_backend, "get_settings", lambda: s)

    client = _StubClient()
    # Provide a Chinese sparse vector so all 3 prefetches participate.
    qdrant_backend._hybrid_text_query(
        client, "coll", [0.1, 0.2], _sv(), _sv(indices=(5, 6), values=(0.5, 0.5)), top_k=5
    )
    q = client.last_kwargs["query"]
    assert q.rrf.weights == [0.7, 1.2, 2.0]
    assert len(client.last_kwargs["prefetch"]) == 3


def test_rrf_k_matches_default() -> None:
    """``RRF_K = 60`` matches Qdrant server's default k constant."""
    assert RRF_K == 60
