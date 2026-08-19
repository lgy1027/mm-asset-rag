"""Tests for ``mm_asset_rag.evaluation_v2``.

The v2 harness adds 50+ Chinese-primary multi-dimensional cases and a
prefix-tolerant matcher that has to survive ``_NNN_hash`` variants
of the same content. These tests pin down the matcher's contract
without running the full Qdrant-backed eval loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from mm_asset_rag.evaluation_v2 import _expand, _match, _title_of, load_cases, run_eval_v2


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


def test_match_slug_normalises_hyphen_and_spaces() -> None:
    """A filename-style hyphenated id must match the spaced paper title —
    the real-world case that broke eval self-consistency: ``clip.pdf``
    uploads as ``attention-is-all-you-need_<hash>`` while the eval expected
    set uses the spaced title ``Attention Is All You Need``."""
    actuals = ["attention-is-all-you-need_0c713762"]
    expected = ["Attention Is All You Need"]
    assert _match(actuals, expected) == 1


def test_match_stacked_hash_collapses_to_bare_title() -> None:
    """A double-stacked hash suffix (source filename carried its own hash,
    upload added another) must collapse to the bare title so a bare
    expected id still matches."""
    actuals = ["Resnext_69df8de4_903a9c76"]
    expected = ["Resnext"]
    assert _match(actuals, expected) == 1


def test_match_hits_on_title_when_asset_id_doesnt() -> None:
    """The CLIP case: the returned asset_id is a filename stem (clip) that
    never matches the paper-title expected id, but the LLM-derived title
    does. With AUTO_META on, hit.title carries that canonical title, so
    passing an (asset_id, title) pair hits where a bare asset_id misses."""
    actuals = [
        ("clip_b14b418e", "Learning Transferable Visual Models From Natural Language Supervision")
    ]
    expected = ["Learning Transferable Visual Models From Natural Language Supervision"]
    assert _match(actuals, expected) == 1


def test_match_bare_asset_id_still_works_without_title() -> None:
    """Backwards compat: a bare asset_id str (no title) still matches the
    way it did before the title pairing was added — image route + old tests."""
    assert _match(["Caltech Airplanes 01_9fe67b3f"], ["Caltech Airplanes"]) == 1


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


def test_load_cases_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    """An explicit ``--cases`` path that doesn't exist raises a clear
    ``FileNotFoundError`` naming the path, not a bare traceback. Guards
    the documented contract so a future refactor can't silently fall back
    to the bundled default on a typo'd ``--cases`` value."""
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_cases(str(missing), version="v2")


def test_load_cases_malformed_json_raises(tmp_path: Path) -> None:
    """A non-JSON file surfaces a parse error rather than scoring 0 cases
    with exit 0."""
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_cases(str(bad), version="v2")


def test_load_cases_missing_groups_raises_value_error(tmp_path: Path) -> None:
    """A file with a valid ``version`` but no ``groups`` object raises
    ``ValueError`` — this is the guard against silently scoring 0 cases
    when the file is structurally a case file but empty."""
    p = tmp_path / "nogroups.json"
    p.write_text('{"version": "v2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="groups"):
        load_cases(str(p), version="v2")


def test_load_cases_non_dict_top_level_raises_value_error(tmp_path: Path) -> None:
    """A top-level JSON array isn't a valid case file."""
    p = tmp_path / "array.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON object"):
        load_cases(str(p), version="v2")


def test_load_cases_version_mismatch_raises(tmp_path: Path) -> None:
    """Loading a v1 file under v2 (or vice versa) raises instead of
    silently scoring 0 cases — the cross-version mix-up guard."""
    p = tmp_path / "mismatch.json"
    p.write_text('{"version": "v1", "groups": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        load_cases(str(p), version="v2")


def test_load_cases_bundled_default_loads(tmp_path: Path) -> None:
    """No ``path`` + no ``EVAL_CASES_PATH`` → the bundled default loads
    and is a non-empty dict of group→case-list."""
    groups = load_cases(None, version="v2")
    assert isinstance(groups, dict)
    assert groups, "bundled v2 default case set is empty"


def test_match_empty_expected_returns_none() -> None:
    """Negative samples (expected=[]) should never be marked hit."""
    assert _match(["Picsum 240", "Caltech Panda 01_3443a5d5"], []) is None


def test_run_eval_v2_wraps_text_to_text() -> None:
    """``run_eval_v2`` is the CLI / API entry point. It must delegate to
    ``run_text_to_text_eval_v2`` and forward ``top_k`` + ``cases_path``
    — that is what makes the v2 production path (``mmrag eval --v2`` /
    ``POST /eval {v2:true}``) work without each caller knowing about
    the per-group runners. Regression for M11: before this alias the v2
    functions were only reachable from tests.
    """
    with patch(
        "mm_asset_rag.evaluation_v2.run_text_to_text_eval_v2",
        return_value=[],
    ) as stub:
        out = run_eval_v2(top_k=7, cases_path="cases.json")
    stub.assert_called_once_with(top_k=7, cases_path="cases.json")
    assert out == []


@dataclass
class _FakeV2Result:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool
    rank: int | None
    group: str


def test_eval_endpoint_v2_path_routes_to_run_eval_v2() -> None:
    """``POST /eval`` with ``v2: true`` must invoke ``run_eval_v2`` and
    return the same ``{"results": [...]}`` shape as v1, with a
    ``version: "v2"`` tag so clients can tell which set ran.
    """
    from fastapi.testclient import TestClient

    from mm_asset_rag.api import app

    fake = [
        _FakeV2Result(
            query="CLIP 模型",
            expected_asset_ids=["Learning Transferable Visual Models"],
            actual_asset_ids=["Learning Transferable Visual Models_79e328a2"],
            hit=True,
            rank=1,
            group="zh_on_en",
        )
    ]
    with patch("mm_asset_rag.evaluation_v2.run_eval_v2", return_value=fake):
        # The endpoint imports ``run_eval_v2`` lazily inside the handler
        # (``from .evaluation_v2 import run_eval_v2``), so patching the
        # attribute on the module is enough.
        client = TestClient(app, base_url="http://127.0.0.1")
        response = client.post("/eval", json={"v2": True, "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "v2"
    assert isinstance(body["results"], list)
    assert len(body["results"]) == 1
    row = body["results"][0]
    assert row["query"] == "CLIP 模型"
    assert row["hit"] is True
    assert row["rank"] == 1
    assert row["group"] == "zh_on_en"
    # Shape parity with v1: same keys.
    assert set(row.keys()) == {
        "query",
        "expected_asset_ids",
        "actual_asset_ids",
        "hit",
        "rank",
        "group",
    }


def test_eval_endpoint_v1_default_unchanged() -> None:
    """``POST /eval`` without ``v2`` keeps the v1 path and does not
    include a ``version`` field (so existing clients are unaffected).
    """
    from fastapi.testclient import TestClient

    from mm_asset_rag.api import app

    with patch("mm_asset_rag.api.run_eval", return_value=[]):
        client = TestClient(app, base_url="http://127.0.0.1")
        response = client.post("/eval", json={})

    assert response.status_code == 200
    body = response.json()
    assert body == {"results": []}
    assert "version" not in body
