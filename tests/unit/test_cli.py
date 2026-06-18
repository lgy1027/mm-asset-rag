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
    args = parser.parse_args(["parse"])
    assert args.command == "parse"
    assert args.pdf_parser == "auto"
    assert args.ocr is False
    assert args.vlm is False
    assert args.limit == 0


def test_cli_index_subcommand_no_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["index"])
    assert args.command == "index"


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
