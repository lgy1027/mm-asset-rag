"""Tests for ``mm_asset_rag.evaluation_v2``.

The v2 harness adds 50+ Chinese-primary multi-dimensional cases and a
prefix-tolerant matcher that has to survive ``_NNN_hash`` variants
of the same content. These tests pin down the matcher's contract
without running the full Qdrant-backed eval loop.
"""

from __future__ import annotations

from mm_asset_rag.evaluation_v2 import _expand, _match, _title_of


def test_title_of_strips_hash() -> None:
    assert _title_of("Caltech Airplanes 01_9fe67b3f") == "Caltech Airplanes 01"
    assert _title_of("联宝 ESG 年度报告_7df7f3f8") == "联宝 ESG 年度报告"
    # No hash → return as-is
    assert _title_of("no_hash_here") == "no_hash_here"
    # Tail not 8 hex chars → keep whole string
    assert _title_of("foo_longtail") == "foo_longtail"


def test_match_handles_hash_variants() -> None:
    """The v2 bug: a single bare id in ``expected`` did not match a
    different ``_NNN_hash`` of the same content in ``actual``. After
    the fix, ``_match`` compares titles (hash-stripped) and accepts
    any variant.
    """
    actuals = [
        "所有深度用 AI 编程的朋友，这篇 Codex 全景指南值得存好，架构生态横评和最佳实践一次讲透_c1cf02d1",
    ]
    expected = [
        "所有深度用 AI 编程的朋友，这篇 Codex 全景指南值得存好，架构生态横评和最佳实践一次讲透_0363cb35",
    ]
    assert _match(actuals, expected) == 1


def test_match_substring_bare_to_full() -> None:
    actuals = ["Learning Transferable Visual Models From Natural Language Supervision_79e328a2"]
    expected = ["Learning Transferable Visual Models From Natural Language Supervision"]
    assert _match(actuals, expected) == 1


def test_match_no_hit_returns_none() -> None:
    actuals = ["Caltech Panda 01_3443a5d5"]
    expected = ["Caltech Dolphin"]
    assert _match(actuals, expected) is None


def test_match_returns_first_hit_rank() -> None:
    actuals = [
        "Caltech Panda 01_3443a5d5",
        "Caltech Panda 02_x1234567",
        "Caltech Dolphin 01_bbd397c6",
    ]
    expected = ["Caltech Panda"]
    # "Caltech Panda 01" is a prefix-tolerant match for "Caltech Panda"
    # via substring containment — so rank 1 is the correct answer.
    assert _match(actuals, expected) == 1


def test_match_uses_title_stripping_to_avoid_hash_substring() -> None:
    """Random hex hash tokens should not be confused for title matches."""
    actuals = ["Caltech Panda 01_3443a5d5"]
    expected = ["a1b2c3d4"]  # bare hash, not a title
    # Hash is 8 chars but stripping on actual yields "Caltech Panda 01"
    # which does not contain "a1b2c3d4" (or vice versa).
    assert _match(actuals, expected) is None


def test_expand_returns_all_hash_variants() -> None:
    full = {
        "Codex_a1b2c3d4",
        "Codex_12345678",
        "Caltech Panda 01_3443a5d5",
    }
    assert _expand("Codex", full) == ["Codex_12345678", "Codex_a1b2c3d4"]
    # No match — fall back to bare id so the strict match still works
    # for cases that pass a full id directly.
    assert _expand("Nothing matches", {"x_y1234567"}) == ["Nothing matches"]


def test_match_empty_expected_returns_none() -> None:
    """Negative samples (expected=[]) should never be marked hit."""
    assert _match(["Picsum 240", "Caltech Panda 01_3443a5d5"], []) is None
