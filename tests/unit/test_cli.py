"""Tests for mm_asset_rag.cli argparse plumbing."""

from __future__ import annotations

import pytest

from mm_asset_rag.cli import build_parser


def test_cli_help_lists_all_subcommands(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_parse_subcommand_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["parse", "paper.pdf", "image.png"])
    assert args.command == "parse"
    assert args.files == ["paper.pdf", "image.png"]
    assert args.pdf_parser == "auto"
    assert args.document_parser == "markitdown"
    assert args.ocr is False
    assert args.vlm is False


def test_cli_parse_subcommand_accepts_document_parser_choice() -> None:
    parser = build_parser()
    args = parser.parse_args(["parse", "doc.docx", "--document-parser", "docling"])
    assert args.document_parser == "docling"
    with pytest.raises(SystemExit):
        parser.parse_args(["parse", "doc.docx", "--document-parser", "bogus"])


def test_cli_index_subcommand_removed() -> None:
    """``mmrag index`` was removed: the same effect comes from
    ``mmrag parse`` (which always indexes after parsing) and
    ``mmrag reindex`` (full rebuild).
    """
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["index"])


def test_cli_search_subcommand_modes() -> None:
    parser = build_parser()
    for mode in ("text", "text-to-image", "image-to-image", "hybrid"):
        args = parser.parse_args(["search", "q", "--mode", mode])
        assert args.mode == mode


def test_cli_search_image_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["search", "q", "--image", "/tmp/img.png"])
    assert args.image == "/tmp/img.png"


def test_cli_answer_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["answer", "why?", "--top-k", "3"])
    assert args.question == "why?"
    assert args.top_k == 3


def test_cli_eval_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["eval", "--top-k", "10"])
    assert args.top_k == 10
    # v2 is opt-in; default is v1 so existing scripts keep their numbers.
    assert args.v2 is False


def test_cli_eval_subcommand_v2_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["eval", "--v2", "--top-k", "7"])
    assert args.v2 is True
    assert args.top_k == 7


def test_cli_eval_v2_invokes_run_eval_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mmrag eval --v2`` must route to ``run_eval_v2`` (not v1's
    ``run_eval``) and write the v2 report. Regression for M11:
    v2 was unreachable from production before this flag existed.
    """
    from dataclasses import dataclass, field

    import mm_asset_rag.cli as cli_mod

    @dataclass
    class _FakeV2Result:
        query: str = "q"
        expected_asset_ids: list[str] = field(default_factory=list)
        actual_asset_ids: list[str] = field(default_factory=list)
        hit: bool = False
        rank: int | None = None
        group: str = "zh_on_en"

    calls: dict[str, object] = {}

    def fake_run_eval_v2(top_k: int):
        calls["top_k"] = top_k
        calls["v2_called"] = True
        return [_FakeV2Result()]

    def fake_write_v2(by_group, path=None):
        calls["write_v2"] = by_group

    def fake_write_v1(results, path=None):
        calls["write_v1"] = results

    # Block ``load_env`` from touching the real env in case the test
    # runner has no .env; it is a no-op when no .env exists, but patching
    # keeps the test hermetic.
    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    import mm_asset_rag.evaluation_v2 as ev2

    monkeypatch.setattr(ev2, "run_eval_v2", fake_run_eval_v2)
    monkeypatch.setattr(ev2, "write_eval_report_v2", fake_write_v2)
    # Guard: v1 must NOT be called when --v2 is set.
    monkeypatch.setattr(
        cli_mod, "run_eval", lambda top_k: (_ for _ in ()).throw(AssertionError("v1 ran"))
    )
    monkeypatch.setattr(cli_mod, "write_eval_report", fake_write_v1)

    args = build_parser().parse_args(["eval", "--v2", "--top-k", "4"])
    cli_mod.command_eval(args)

    assert calls.get("v2_called") is True
    assert calls.get("top_k") == 4
    assert "write_v2" in calls
    assert "write_v1" not in calls


def test_cli_retry_subcommand_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["retry", "abc123def456"])
    assert args.command == "retry"
    assert args.task_id == "abc123def456"
    assert args.force is False
    assert args.failed_only is False


def test_cli_retry_subcommand_force_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["retry", "abc123def456", "--force"])
    assert args.force is True
    assert args.failed_only is False


def test_cli_retry_subcommand_failed_only_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["retry", "abc123def456", "--failed-only"])
    assert args.failed_only is True


def test_cli_retry_subcommand_force_and_failed_only_compose() -> None:
    parser = build_parser()
    args = parser.parse_args(["retry", "abc123def456", "--force", "--failed-only"])
    assert args.force is True
    assert args.failed_only is True


def test_cli_delete_subcommand_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["delete", "abc123def456"])
    assert args.command == "delete"
    assert args.asset_id == "abc123def456"
    assert args.yes is False
    assert args.dry_run is False
    args = parser.parse_args(["delete", "abc123def456", "--yes", "--dry-run"])
    assert args.yes is True
    assert args.dry_run is True


def test_cli_reindex_subcommand_yes_flag() -> None:
    """``--yes`` skips the interactive confirmation. Needed for CI
    and for the "switch CLIP / embedding model" recipe in
    ``docs/eval-report-v3.md``.
    """
    parser = build_parser()
    args = parser.parse_args(["reindex", "--image-only"])
    assert args.yes is False
    args = parser.parse_args(["reindex", "--text-only", "--yes"])
    assert args.yes is True
    assert args.text_only is True
    assert args.image_only is False
