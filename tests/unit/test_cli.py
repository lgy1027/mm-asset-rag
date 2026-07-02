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
    assert args.ocr is False
    assert args.vlm is False


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
