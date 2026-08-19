"""Tests for mm_asset_rag.evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mm_asset_rag.evaluation import (
    EvalResult,
    load_cases,
    run_eval,
    strip_trailing_hash,
    write_eval_report,
)
from mm_asset_rag.schema import SearchHit

# asset_id returned by the fake retriever for each query substring.
# Keyed on a unique phrase that appears in the bundled default case set.
# Keeps the test independent of how the asset_id is built (bare / full /
# hashed) — the prefix-tolerant _match handles the suffix transparently.
_QUERY_TO_ASSET = {
    "retrieval augmented generation": "Retrieval Augmented Generation_caaa534b",
    "attention is all you need": "Attention Is All You Need_caaa534b",
    "deep residual learning": "Resnet_caaa534b",
    "generative adversarial": "Gan_caaa534b",
    "transferable visual models": "Learning Transferable Visual Models From Natural Language Supervision_caaa534b",
    "残差网络": "Resnet_caaa534b",
    "自注意力": "Attention Is All You Need_caaa534b",
}


def _fake_hybrid(query: str, **kwargs):
    """Return a single ``SearchHit`` keyed on the test's query → asset map."""
    q = query.lower()
    asset_id = next(
        (aid for needle, aid in _QUERY_TO_ASSET.items() if needle.lower() in q),
        "other",
    )
    return [
        SearchHit(
            route="text",
            score=0.9,
            asset_id=asset_id,
            title=asset_id,
            source_type="pdf",
            source_path=f"{asset_id}.pdf",
        )
    ]


def _default_case_count() -> int:
    """Total text→text cases in the bundled default v1 case set."""
    groups = load_cases()
    return sum(len(groups.get(g, ())) for g in ("en", "zh", "zh_doc", "legacy"))


def test_eval_cases_count() -> None:
    # The bundled default ships a small generic en + zh text→text set.
    groups = load_cases()
    assert len(groups["en"]) == 5
    assert len(groups["zh"]) == 3
    assert _default_case_count() == 8


def test_run_eval_hits_expected_assets() -> None:
    # Skip bare→full expansion so the expected ids stay as the bare
    # titles from the case file; the prefix-tolerant _match handles the
    # test's mock asset_id suffix transparently.
    with patch("mm_asset_rag.evaluation.hybrid_search", side_effect=_fake_hybrid):
        results = run_eval()
    assert len(results) == _default_case_count()
    misses = [r for r in results if not r.hit]
    assert not misses, misses


def test_run_eval_misses_when_assets_wrong() -> None:
    with patch(
        "mm_asset_rag.evaluation.hybrid_search",
        return_value=[
            SearchHit(
                route="text",
                score=0.9,
                asset_id="unrelated",
                title="",
                source_type="pdf",
                source_path="x.pdf",
            )
        ],
    ):
        results = run_eval()
    assert not any(result.hit for result in results)
    # Every miss should record rank=None and the actual asset_id of "unrelated".
    for r in results:
        assert r.rank is None
        assert r.actual_asset_ids == ["unrelated"]


def test_run_eval_records_rank_and_group() -> None:
    with patch("mm_asset_rag.evaluation.hybrid_search", side_effect=_fake_hybrid):
        results = run_eval()
    en = [r for r in results if r.group == "en"]
    zh = [r for r in results if r.group == "zh"]
    assert en, "expected en queries"
    assert zh, "expected zh queries"
    for r in en:
        assert r.rank == 1, r
    for r in zh:
        assert r.rank == 1, r


def test_run_eval_respects_cases_path(tmp_path: Path) -> None:
    """``cases_path`` overrides the default file for one run — a custom
    two-case file produces exactly two results, regardless of the
    bundled default's size."""
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "version": "v1",
                "groups": {
                    "en": [
                        {"query": "q1", "expected_asset_ids": ["A"]},
                        {"query": "q2", "expected_asset_ids": ["B"]},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    with patch("mm_asset_rag.evaluation.hybrid_search", side_effect=_fake_hybrid):
        results = run_eval(cases_path=custom)
    assert len(results) == 2
    assert all(r.group == "en" for r in results)


def test_load_cases_reads_eval_cases_path_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The middle resolution tier: when no explicit path is passed,
    ``load_cases`` falls back to ``Settings.EVAL_CASES_PATH``."""
    from mm_asset_rag.settings import get_settings

    custom = tmp_path / "env_cases.json"
    custom.write_text(
        json.dumps(
            {
                "version": "v1",
                "groups": {"en": [{"query": "env", "expected_asset_ids": ["X"]}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_CASES_PATH", str(custom))
    get_settings.cache_clear()
    try:
        groups = load_cases()
    finally:
        get_settings.cache_clear()
    assert groups == {"en": [{"query": "env", "expected_asset_ids": ["X"]}]}


def test_load_cases_explicit_path_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``path`` wins over ``EVAL_CASES_PATH``."""
    from mm_asset_rag.settings import get_settings

    env_file = tmp_path / "env.json"
    env_file.write_text(
        json.dumps(
            {"version": "v1", "groups": {"en": [{"query": "env", "expected_asset_ids": ["E"]}]}}
        ),
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit.json"
    explicit.write_text(
        json.dumps(
            {
                "version": "v1",
                "groups": {"en": [{"query": "explicit", "expected_asset_ids": ["X"]}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_CASES_PATH", str(env_file))
    get_settings.cache_clear()
    try:
        groups = load_cases(explicit)
    finally:
        get_settings.cache_clear()
    assert groups["en"][0]["query"] == "explicit"


def test_load_cases_version_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v2 file fed to v1 ``load_cases`` must raise — otherwise the v1
    runner's groups are all absent and it silently scores 0 cases."""
    from mm_asset_rag.evaluation_v2 import load_cases as load_v2

    v2_file = tmp_path / "v2.json"
    v2_file.write_text(
        json.dumps(
            {"version": "v2", "groups": {"zh_on_en": [{"query": "q", "expected_asset_ids": ["A"]}]}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version mismatch"):
        load_cases(v2_file)  # v1 loader, v2 file
    # Omitted version is tolerated (user-authored file may skip the field).
    no_version = tmp_path / "no_version.json"
    no_version.write_text(
        json.dumps({"groups": {"en": [{"query": "q", "expected_asset_ids": ["A"]}]}}),
        encoding="utf-8",
    )
    assert load_cases(no_version)  # v1 loader, no version → ok
    # And v2 loader accepts a v2 file.
    assert load_v2(v2_file, version="v2")


def test_write_eval_report(tmp_path: Path) -> None:
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["a"],
            actual_asset_ids=["a"],
            hit=True,
            rank=1,
            group="en",
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    # New schema: a top-level object (not a bare list) with per_query +
    # aggregate metrics.
    assert payload["total"] == 1
    assert payload["hit_count"] == 1
    assert payload["per_query"][0]["query"] == "q"
    assert payload["per_query"][0]["rank"] == 1
    assert payload["per_query"][0]["group"] == "en"
    assert "metrics" in payload
    # Per-group metrics are built dynamically from the groups present in
    # ``results`` (all + every group that appears). A single en case
    # produces all + en; zh is absent because no zh case was supplied.
    assert "all" in payload["metrics"]
    assert "en" in payload["metrics"]
    assert "zh" not in payload["metrics"]


def test_write_eval_report_metrics_cover_every_present_group(tmp_path: Path) -> None:
    """The metrics block reflects every group present in ``results`` —
    no group's aggregate is silently dropped. Regression for the
    ``zh_doc`` loss: ``write_eval_report`` used to hard-code only
    en/zh, so a run carrying zh_doc counted it in ``all`` but had no
    breakdown of its own."""
    results = [
        EvalResult(
            query="a",
            expected_asset_ids=["x"],
            actual_asset_ids=["x"],
            hit=True,
            rank=1,
            group="en",
        ),
        EvalResult(
            query="b",
            expected_asset_ids=["y"],
            actual_asset_ids=["y"],
            hit=True,
            rank=1,
            group="zh_doc",
        ),
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload["metrics"].keys()) == {"all", "en", "zh_doc"}
    assert payload["metrics"]["zh_doc"]["hit_rate"]["5"] == 1.0


# ── strip_trailing_hash + casefold normalisation ─────────────────────────


def test_strip_trailing_hash_drops_8_hex_suffix() -> None:
    assert strip_trailing_hash("Alexnet_0c1c2b23") == "alexnet"
    # casefold applies to the title portion.
    assert strip_trailing_hash("AlexNet_caaa534b") == "alexnet"
    # Non-hex tail (length 8 but not hex) → keep whole, casefold only.
    assert strip_trailing_hash("foo_longtail") == "foo_longtail"
    # No underscore → casefold the whole id.
    assert strip_trailing_hash("CLIP") == "clip"
    assert strip_trailing_hash("") == ""


def test_match_case_insensitive_partial_title() -> None:
    """The R-CNN failure: retriever returns the correct paper with a
    different casing + content hash, expected is a partial bare title.
    After normalisation the matcher must count this as a hit.
    """
    from mm_asset_rag.evaluation import _match

    actual = [
        "Rich Feature Hierarchies for Accurate Object Detection And Semantic Segmentation_b857cf69"
    ]
    expected = ["Rich feature hierarchies"]
    assert _match(actual, expected) == 1


def test_match_casefold_hash_variant() -> None:
    """Same content re-parsed under a new hash + different casing still
    counts as the same document."""
    from mm_asset_rag.evaluation import _match

    assert _match(["ALEXNET_0c1c2b23"], ["Alexnet_caaa534b"]) == 1


def test_match_still_misses_unrelated() -> None:
    from mm_asset_rag.evaluation import _match

    assert _match(["Caltech Panda 01_3443a5d5"], ["Caltech Dolphin"]) is None
    # Empty expected never hits.
    assert _match(["anything"], []) is None


def test_write_eval_report_normalises_ids_for_metrics(tmp_path: Path) -> None:
    """aggregate_metrics gets hash-stripped + casefolded ids so a
    re-parse with a different sha counts in the strict-set metric."""
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["Alexnet_caaa534b"],
            actual_asset_ids=["Alexnet_0c1c2b23"],
            hit=True,
            rank=1,
            group="en",
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    # hit_rate@1 in the aggregate block should be 1.0 once the two
    # hash variants normalise to the same bare title.
    assert payload["metrics"]["all"]["hit_rate"]["1"] == 1.0


def test_write_eval_report_handles_title_list_shorter_than_ids(tmp_path: Path) -> None:
    """If actual_titles is shorter than actual_asset_ids (shouldn't happen in
    production but a future code path might), zip_longest fills with "" so the
    trailing asset_ids are still scored against expected — no silent drop."""
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["Alexnet"],
            actual_asset_ids=["Alexnet_0c1c2b23", "Resnext_69df8de4"],
            hit=True,
            rank=1,
            group="en",
            actual_titles=["Learning Transferable"],  # only 1 title for 2 ids
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    # Both actual_ids still scored (the 2nd falls back to bare asset_id).
    assert payload["metrics"]["all"]["hit_rate"]["5"] == 1.0


def test_write_eval_report_empty_titles_falls_back_to_asset_id(tmp_path: Path) -> None:
    """When actual_titles is empty (image route / old reports), _agg falls back
    to scoring on bare asset_ids — no crash, no empty actual_ids list."""
    results = [
        EvalResult(
            query="q",
            expected_asset_ids=["Alexnet_caaa534b"],
            actual_asset_ids=["Alexnet_0c1c2b23"],
            hit=True,
            rank=1,
            group="en",
            actual_titles=[],
        )
    ]
    target = tmp_path / "report.json"
    write_eval_report(results, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["metrics"]["all"]["hit_rate"]["1"] == 1.0
