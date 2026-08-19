"""Tests for mm_asset_rag.cli argparse plumbing."""

from __future__ import annotations

from pathlib import Path

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
    # --cases is optional; default is None (use the bundled sample).
    assert args.cases is None


def test_cli_eval_subcommand_cases_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["eval", "--cases", "my_cases.json"])
    assert args.cases == "my_cases.json"


def test_cli_eval_v1_passes_cases_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``mmrag eval --cases <file>`` (v1 default) must parse ``--cases``
    and forward a resolved ``cases_path`` to ``run_eval`` — the main path,
    not just v2. The path is resolved under ``eval_cases/`` so the forwarded
    value is absolute, but its basename must be the user's filename."""
    import mm_asset_rag.cli as cli_mod

    # Drop a case file in the allowed eval_cases/ dir so resolution succeeds.
    from mm_asset_rag.paths import get_eval_cases_dir

    case_file = get_eval_cases_dir() / "my_cases.json"
    case_file.parent.mkdir(parents=True, exist_ok=True)
    case_file.write_text('{"version":"v1","groups":{}}', encoding="utf-8")

    calls: dict[str, object] = {}

    def fake_run_eval(top_k, *, cases_path=None):
        calls["top_k"] = top_k
        calls["cases_path"] = cases_path
        return []

    def fake_write_v1(results, path=None):
        calls["write_v1"] = results

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", fake_run_eval)
    monkeypatch.setattr(cli_mod, "write_eval_report", fake_write_v1)

    args = build_parser().parse_args(["eval", "--cases", "my_cases.json", "--top-k", "3"])
    cli_mod.command_eval(args)

    assert calls.get("top_k") == 3
    forwarded = calls.get("cases_path")
    assert forwarded is not None
    assert Path(forwarded).name == "my_cases.json"
    # Resolved path actually exists on disk (the resolver checks existence).
    assert Path(forwarded).is_file()
    assert "write_v1" in calls


def test_cli_eval_cases_path_falls_back_to_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``--cases <name>`` whose file lives only in the repo ``examples/``
    dir resolves to the examples/ path (the chapter11 baselines ship there).
    Pins the examples/ fallback so the resolver doesn't silently 422 a file
    that exists in the second allowed dir."""
    import mm_asset_rag.cli as cli_mod

    # A case file that genuinely ships in examples/ (created by the eval
    # parameterization). Use a unique bare name resolvable only there.
    repo_examples = Path(__file__).resolve().parents[2] / "examples"
    assert repo_examples.exists(), "repo examples/ must exist for this test"
    # Pick the real chapter11 v1 file that ships there.
    target = repo_examples / "eval_cases_chapter11_v1.json"
    assert target.exists(), "chapter11 v1 baseline must ship in examples/"

    calls: dict[str, object] = {}

    def fake_run_eval(top_k, *, cases_path=None):
        calls["cases_path"] = cases_path
        return []

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", fake_run_eval)
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    # Bare name → eval_cases/ miss → examples/ fallback.
    args = build_parser().parse_args(["eval", "--cases", "eval_cases_chapter11_v1.json"])
    cli_mod.command_eval(args)

    forwarded = calls.get("cases_path")
    assert forwarded is not None
    assert Path(forwarded).name == "eval_cases_chapter11_v1.json"
    assert Path(forwarded).is_file()


def test_cli_eval_cases_path_accepts_examples_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The README/docs use the ``--cases examples/<name>`` form. The resolver
    strips the leading ``examples/`` segment for the examples/ root (which
    *is* examples/) so it doesn't double up to ``examples/examples/<name>``.
    Pins the documented usage."""
    import mm_asset_rag.cli as cli_mod

    calls: dict[str, object] = {}

    def fake_run_eval(top_k, *, cases_path=None):
        calls["cases_path"] = cases_path
        return []

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", fake_run_eval)
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    args = build_parser().parse_args(["eval", "--cases", "examples/eval_cases_chapter11_v1.json"])
    cli_mod.command_eval(args)

    forwarded = calls.get("cases_path")
    assert forwarded is not None
    assert Path(forwarded).name == "eval_cases_chapter11_v1.json"
    assert Path(forwarded).is_file()


def test_cli_eval_cases_path_rejects_missing_in_both_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``--cases <name>`` not in eval_cases/ or examples/ raises
    SystemExit (clear CLI error) rather than silently loading nothing."""
    import mm_asset_rag.cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", lambda *a, **kw: [])
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    args = build_parser().parse_args(["eval", "--cases", "definitely_missing.json"])
    with pytest.raises(SystemExit, match="not found"):
        cli_mod.command_eval(args)


def test_cli_eval_cases_path_rejects_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--cases`` must reject path traversal so a CLI user can't point the
    eval loader at an arbitrary file (``--cases ../../etc/passwd.json``).
    Pins the request-side guard's CLI mirror — both surfaces restrict the
    path to ``eval_cases/`` or ``examples/``."""
    import mm_asset_rag.cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", lambda *a, **kw: [])
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    args = build_parser().parse_args(["eval", "--cases", "../../etc/passwd.json"])
    with pytest.raises(SystemExit, match="relative path"):
        cli_mod.command_eval(args)


def test_cli_eval_cases_path_rejects_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-.json ``--cases`` value is bounced before the loader reads it."""
    import mm_asset_rag.cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", lambda *a, **kw: [])
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    args = build_parser().parse_args(["eval", "--cases", "secret.txt"])
    with pytest.raises(SystemExit, match=r"\.json"):
        cli_mod.command_eval(args)


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

    def fake_run_eval_v2(top_k: int, *, cases_path=None):
        calls["top_k"] = top_k
        calls["cases_path"] = cases_path
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
        cli_mod,
        "run_eval",
        lambda top_k, cases_path=None: (_ for _ in ()).throw(AssertionError("v1 ran")),
    )
    monkeypatch.setattr(cli_mod, "write_eval_report", fake_write_v1)

    args = build_parser().parse_args(["eval", "--v2", "--top-k", "4"])
    cli_mod.command_eval(args)

    assert calls.get("v2_called") is True
    assert calls.get("top_k") == 4
    assert calls.get("cases_path") is None
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


def test_wait_for_task_exits_on_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ``_wait_for_task``'s terminal-status set must include
    ``"cancelled"``. Before the fix a task cancelled via the API left the
    CLI polling forever (``cancelled`` wasn't in the exit set, so the loop
    slept and re-polled with no escape). Pins ``TaskStatus.CANCELLED`` as a
    terminal status the CLI treats as done.
    """
    from types import SimpleNamespace

    from mm_asset_rag import cli
    from mm_asset_rag.service import TaskStatus

    assert "cancelled" in TaskStatus.terminal()

    # A cancelled task record the poller returns.
    cancelled_rec = SimpleNamespace(
        task_id="t1", status="cancelled", current="cancelled by request"
    )
    calls = {"n": 0}

    def fake_get_task(task_id):
        calls["n"] += 1
        return cancelled_rec

    monkeypatch.setattr(cli.get_service(), "get_task", fake_get_task)
    # ``_wait_for_task`` must return on the first poll (not loop).
    cli._wait_for_task("t1", poll_interval=0.01)
    assert calls["n"] == 1, "cancelled status did not terminate the poll loop"
