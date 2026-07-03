"""Tests for ``_channel_limit`` and the per-channel RRF weight plumbing.

qdrant-client 1.18 doesn't expose per-prefetch RRF weights, so
``_hybrid_text_query`` approximates the bias by scaling each prefetch
``limit``. These tests pin the scaling contract so deployments
that tune ``Settings.rrf_weight_*`` see predictable behaviour.
"""

from __future__ import annotations

from mm_asset_rag.backends.qdrant_backend import _channel_limit


def test_weight_one_keeps_baseline() -> None:
    assert _channel_limit(1.0) == 20  # HYBRID_PREFETCH_LIMIT default


def test_weight_above_one_scales_up() -> None:
    assert _channel_limit(1.5) == 30
    assert _channel_limit(2.0) == 40
    assert _channel_limit(3.0) == 60


def test_weight_below_one_floors_at_baseline() -> None:
    # Below 1.0 weights should NOT shrink the candidate pool — that
    # would make a poorly-tuned weight destroy recall on a single
    # channel. Floor at HYBRID_PREFETCH_LIMIT.
    assert _channel_limit(0.5) == 20
    assert _channel_limit(0.1) == 20


def test_weight_caps_at_four_x() -> None:
    # Runaway weights get capped at 4x to keep latency bounded.
    assert _channel_limit(5.0) == 80
    assert _channel_limit(100.0) == 80
