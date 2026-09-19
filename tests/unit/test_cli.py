"""Tests for mm_asset_rag.cli argparse plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.cli import _collect_upload_files, build_parser
from mm_asset_rag.core.schema import SearchHit


def test_cli_help_lists_all_subcommands(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_parse_subcommand_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["parse", "paper.pdf", "image.png", "--collection", "team", "--principal", "alice"]
    )
    assert args.command == "parse"
    assert args.files == ["paper.pdf", "image.png"]
    assert args.pdf_parser == "auto"
    assert args.document_parser == "markitdown"
    assert args.ocr is False
    assert args.vlm is False
    assert args.collection == "team"
    assert args.principals == ["alice"]


def test_cli_parse_help_lists_document_and_table_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["parse", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "DOCX" in help_text
    assert "CSV" in help_text
    assert "TSV" in help_text


def test_cli_documents_help_describes_current_document_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["documents", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "current documents" in help_text
    assert "versions" not in help_text


def test_collect_upload_files_keeps_relative_directory_labels(tmp_path: Path) -> None:
    root = tmp_path / "images"
    poster = root / "2026年KO活动" / "poster.png"
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b"png")

    assert _collect_upload_files([root]) == [("2026年KO活动/poster.png", poster)]


def test_cli_parse_subcommand_accepts_document_parser_choice() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "parse",
            "doc.docx",
            "--document-parser",
            "docling",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
    assert args.document_parser == "docling"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "parse",
                "doc.docx",
                "--document-parser",
                "bogus",
                "--collection",
                "team",
                "--principal",
                "alice",
            ]
        )


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
        args = parser.parse_args(
            ["search", "q", "--mode", mode, "--collection", "team", "--principal", "alice"]
        )
        assert args.mode == mode


def test_cli_search_image_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["search", "q", "--image", "/tmp/img.png", "--collection", "team", "--principal", "alice"]
    )
    assert args.image == "/tmp/img.png"


def test_cli_search_translates_invalid_image_path_to_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI renders search validation failures without a Python traceback."""
    import mm_asset_rag.query.search_service as search_service_mod
    from mm_asset_rag.query.search_service import SearchService

    class _Backend:
        pass

    monkeypatch.setattr(search_service_mod, "get_search_service", lambda: SearchService(_Backend()))
    args = build_parser().parse_args(
        [
            "search",
            "q",
            "--mode",
            "hybrid",
            "--image",
            "../outside.png",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )

    with pytest.raises(SystemExit, match="error: image_path resolves outside assets/"):
        args.func(args)


def test_cli_search_preserves_runtime_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend/runtime failures retain the existing RuntimeError behavior."""
    import mm_asset_rag.cli as cli_mod

    def fail_search(**kwargs):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(cli_mod, "dispatch_search", fail_search)
    args = build_parser().parse_args(
        ["search", "q", "--collection", "team", "--principal", "alice"]
    )

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        args.func(args)


def test_cli_search_serializes_only_public_hit_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import mm_asset_rag.cli as cli_mod

    hit = SearchHit(
        route="text",
        score=0.9,
        asset_id="private-cache-key",
        title="Handbook",
        source_type="pdf",
        source_path="pdfs/handbook.pdf",
        evidence="body",
        metadata={
            "document_id": "handbook",
            "chunk_id": "handbook:0",
            "access_policy": {"allowed_principals": ["alice"]},
        },
    )
    monkeypatch.setattr(cli_mod, "dispatch_search", lambda **_kwargs: [hit])

    cli_mod.command_search(
        build_parser().parse_args(["search", "q", "--collection", "team", "--principal", "alice"])
    )

    row = json.loads(capsys.readouterr().out)[0]
    assert row["document_id"] == "handbook"
    assert "version_id" not in row
    assert row["chunk_id"] == "handbook:0"
    assert "asset_id" not in json.dumps(row)
    assert "access_policy" not in json.dumps(row)


def test_cli_answer_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "answer",
            "why?",
            "--top-k",
            "3",
            "--collection",
            "team",
            "--principal",
            "alice",
            "--min-confidence",
            "0.5",
        ]
    )
    assert args.question == "why?"
    assert args.top_k == 3
    assert args.collection == "team"
    assert args.principal == "alice"
    assert args.min_confidence == 0.5


def test_cli_answer_supplies_the_search_service(monkeypatch: pytest.MonkeyPatch) -> None:
    import mm_asset_rag.cli as cli_mod

    service = object()
    calls: dict[str, object] = {}
    monkeypatch.setattr(cli_mod, "get_search_service", lambda: service)
    monkeypatch.setattr(
        cli_mod,
        "answer_json",
        lambda question, top_k, *, search_service, collection, metadata_filter, principal, min_confidence: (
            calls.update(
                question=question,
                top_k=top_k,
                search_service=search_service,
                collection=collection,
                metadata_filter=metadata_filter,
                principal=principal,
                min_confidence=min_confidence,
            )
            or "{}"
        ),
    )

    cli_mod.command_answer(
        build_parser().parse_args(
            [
                "answer",
                "why?",
                "--top-k",
                "3",
                "--collection",
                "team",
                "--principal",
                "alice",
                "--min-confidence",
                "0.5",
            ]
        )
    )

    assert calls == {
        "question": "why?",
        "top_k": 3,
        "search_service": service,
        "collection": "team",
        "metadata_filter": None,
        "principal": "alice",
        "min_confidence": 0.5,
    }


def test_cli_eval_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["eval", "--top-k", "10", "--collection", "team", "--principal", "alice"]
    )
    assert args.top_k == 10
    # v2 is opt-in; default is v1 so existing scripts keep their numbers.
    assert args.v2 is False
    # --cases is optional; default is None (use the bundled sample).
    assert args.cases is None


def test_cli_eval_subcommand_cases_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["eval", "--cases", "my_cases.json", "--collection", "team", "--principal", "alice"]
    )
    assert args.cases == "my_cases.json"


def test_cli_eval_v1_passes_cases_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``mmrag eval --cases <file>`` (v1 default) must parse ``--cases``
    and forward a resolved ``cases_path`` to ``run_eval`` — the main path,
    not just v2. The path is resolved under ``eval_cases/`` so the forwarded
    value is absolute, but its basename must be the user's filename."""
    import mm_asset_rag.cli as cli_mod

    # Drop a case file in the allowed eval_cases/ dir so resolution succeeds.
    from mm_asset_rag.core.paths import get_eval_cases_dir

    case_file = get_eval_cases_dir() / "my_cases.json"
    case_file.parent.mkdir(parents=True, exist_ok=True)
    case_file.write_text('{"version":"v1","groups":{}}', encoding="utf-8")

    calls: dict[str, object] = {}

    def fake_run_eval(top_k, *, cases_path=None, collection, principal, metadata_filter=None):
        calls["top_k"] = top_k
        calls["cases_path"] = cases_path
        return []

    def fake_write_v1(results, path=None, *, collection=None):
        calls["write_v1"] = results

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", fake_run_eval)
    monkeypatch.setattr(cli_mod, "write_eval_report", fake_write_v1)

    args = build_parser().parse_args(
        [
            "eval",
            "--cases",
            "my_cases.json",
            "--top-k",
            "3",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
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

    def fake_run_eval(top_k, *, cases_path=None, collection, principal, metadata_filter=None):
        calls["cases_path"] = cases_path
        return []

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", fake_run_eval)
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    # Bare name → eval_cases/ miss → examples/ fallback.
    args = build_parser().parse_args(
        [
            "eval",
            "--cases",
            "eval_cases_chapter11_v1.json",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
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

    def fake_run_eval(top_k, *, cases_path=None, collection, principal, metadata_filter=None):
        calls["cases_path"] = cases_path
        return []

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", fake_run_eval)
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    args = build_parser().parse_args(
        [
            "eval",
            "--cases",
            "examples/eval_cases_chapter11_v1.json",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
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

    args = build_parser().parse_args(
        [
            "eval",
            "--cases",
            "definitely_missing.json",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
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

    args = build_parser().parse_args(
        [
            "eval",
            "--cases",
            "../../etc/passwd.json",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
    with pytest.raises(SystemExit, match="relative path"):
        cli_mod.command_eval(args)


def test_cli_eval_cases_path_rejects_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-.json ``--cases`` value is bounced before the loader reads it."""
    import mm_asset_rag.cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(cli_mod, "run_eval", lambda *a, **kw: [])
    monkeypatch.setattr(cli_mod, "write_eval_report", lambda *a, **kw: None)

    args = build_parser().parse_args(
        [
            "eval",
            "--cases",
            "secret.txt",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
    with pytest.raises(SystemExit, match=r"\.json"):
        cli_mod.command_eval(args)


def test_cli_eval_subcommand_v2_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["eval", "--v2", "--top-k", "7", "--collection", "team", "--principal", "alice"]
    )
    assert args.v2 is True
    assert args.top_k == 7


def test_cli_eval_default_uses_bundled_qrels(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real default CLI path loads and serializes the bundled qrels cases."""
    from types import SimpleNamespace

    import mm_asset_rag.cli as cli_mod
    import mm_asset_rag.eval.evaluation as evaluation_mod

    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    monkeypatch.setattr(
        evaluation_mod,
        "get_search_service",
        lambda: SimpleNamespace(execute=lambda _command: []),
    )

    cli_mod.command_eval(
        build_parser().parse_args(["eval", "--collection", "team", "--principal", "alice"])
    )

    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 8
    assert all(set(row) >= {"query_id", "qrels", "actual_document_ids"} for row in rows)
    assert not any("expected_asset_ids" in row for row in rows)
    # Reports are scoped per collection so one knowledge base's eval
    # cannot clobber another's.
    assert (tmp_home / "eval_report_team.json").is_file()


def test_cli_eval_v2_invokes_run_eval_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mmrag eval --v2`` must route to ``run_eval_v2`` (not v1's
    ``run_eval``) and write the v2 report. Regression for M11:
    v2 was unreachable from production before this flag existed.
    """
    from dataclasses import dataclass, field

    import mm_asset_rag.cli as cli_mod

    @dataclass
    class _FakeV2Result:
        query_id: str = "q1"
        query: str = "q"
        qrels: dict[str, int] = field(default_factory=dict)
        actual_document_ids: list[str] = field(default_factory=list)
        hit: bool = False
        rank: int | None = None
        group: str = "zh_on_en"

    calls: dict[str, object] = {}

    def fake_run_eval_v2(
        top_k: int, *, cases_path=None, collection, principal, metadata_filter=None
    ):
        calls["top_k"] = top_k
        calls["cases_path"] = cases_path
        calls["v2_called"] = True
        return [_FakeV2Result()]

    def fake_write_v2(by_group, path=None, *, collection=None):
        calls["write_v2"] = by_group

    def fake_write_v1(results, path=None, *, collection=None):
        calls["write_v1"] = results

    # Block ``load_env`` from touching the real env in case the test
    # runner has no .env; it is a no-op when no .env exists, but patching
    # keeps the test hermetic.
    monkeypatch.setattr(cli_mod, "load_env", lambda: None)
    import mm_asset_rag.eval.evaluation_v2 as ev2

    monkeypatch.setattr(ev2, "run_eval_v2", fake_run_eval_v2)
    monkeypatch.setattr(ev2, "write_eval_report_v2", fake_write_v2)
    # Guard: v1 must NOT be called when --v2 is set.
    monkeypatch.setattr(
        cli_mod,
        "run_eval",
        lambda top_k, cases_path=None: (_ for _ in ()).throw(AssertionError("v1 ran")),
    )
    monkeypatch.setattr(cli_mod, "write_eval_report", fake_write_v1)

    args = build_parser().parse_args(
        [
            "eval",
            "--v2",
            "--top-k",
            "4",
            "--collection",
            "team",
            "--principal",
            "alice",
        ]
    )
    cli_mod.command_eval(args)

    assert calls.get("v2_called") is True
    assert calls.get("top_k") == 4
    assert calls.get("cases_path") is None
    assert "write_v2" in calls
    assert "write_v1" not in calls


def test_cli_eval_image_runs_strict_image_runner_and_scoped_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass, field

    import mm_asset_rag.cli as cli_mod
    import mm_asset_rag.eval.evaluation_v2 as ev2

    @dataclass
    class _Result:
        query_id: str = "tti-airplanes-zh-01"
        query: str = "飞机"
        qrels: dict[str, int] = field(default_factory=lambda: {"Caltech Airplanes 01_9fe67b3f": 1})
        actual_document_ids: list[str] = field(
            default_factory=lambda: ["Caltech Airplanes 01_9fe67b3f"]
        )
        hit: bool = True
        rank: int | None = 1
        group: str = "text_to_image_zh"

    calls: dict[str, object] = {}
    monkeypatch.setattr(cli_mod, "_resolve_cli_cases_path", lambda _value: "image-qrels.json")
    monkeypatch.setattr(
        ev2,
        "run_image_eval_v2",
        lambda **kwargs: calls.setdefault("run", kwargs) and [_Result()],
    )
    monkeypatch.setattr(
        ev2,
        "write_eval_report_v2",
        lambda groups, **kwargs: calls.setdefault("write", (groups, kwargs)),
    )

    args = cli_mod.build_parser().parse_args(
        [
            "eval",
            "--image",
            "--cases",
            "eval_cases_images_v2.json",
            "--collection",
            "image-test",
            "--principal",
            "alice",
        ]
    )
    cli_mod.command_eval(args)

    assert calls["run"] == {
        "top_k": 5,
        "cases_path": "image-qrels.json",
        "collection": "image-test",
        "metadata_filter": None,
        "principal": "alice",
    }
    groups, write_kwargs = calls["write"]
    assert groups == {"text_to_image_zh": [_Result()]}
    assert write_kwargs["collection"] == "image-test"
    assert write_kwargs["run_context"]["retrieval_gate"] == "primitive_image_routes"


def test_cli_ingest_image_eval_builds_manifest_file_list_and_uses_default_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import mm_asset_rag.cli as cli_mod

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        cli_mod,
        "_run_cli_ingest",
        lambda files, **kwargs: calls.setdefault("call", (list(files), kwargs)),
    )
    monkeypatch.setattr(
        "mm_asset_rag.eval.image_cases.image_eval_corpus_files",
        lambda _path: [tmp_path / "Caltech Cat 01_a.jpg"],
    )

    args = cli_mod.build_parser().parse_args(
        [
            "ingest-image-eval",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--collection",
            "image-test",
            "--principal",
            "alice",
        ]
    )
    cli_mod.command_ingest_image_eval(args)

    files, kwargs = calls["call"]
    assert files == [tmp_path / "Caltech Cat 01_a.jpg"]
    assert kwargs["collection"] == "image-test"
    assert kwargs["principals"] == ["alice"]
    assert kwargs["document_id"] is None


def test_cli_ingest_image_eval_rejects_document_id_override() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ingest-image-eval",
                "--manifest",
                "manifest.json",
                "--collection",
                "team",
                "--principal",
                "alice",
                "--document-id",
                "custom",
            ]
        )


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


def test_cli_document_lifecycle_subcommand_replaces_asset_delete() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["documents"])
    args = parser.parse_args(["documents", "--collection", "team", "--principal", "alice"])
    assert args.command == "documents"
    with pytest.raises(SystemExit):
        parser.parse_args(["delete", "physical-asset-id"])


def test_cli_documents_enforces_acl_and_hides_policy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import mm_asset_rag.cli as cli_mod
    from mm_asset_rag.core.knowledge_models import AccessPolicy, Asset, Document, Source
    from mm_asset_rag.ingest.asset_index import DocumentRecord

    def record(document_id: str, principal: str) -> DocumentRecord:
        document = Document(
            document_id,
            document_id.title(),
            Source(source_id=f"upload:{document_id}"),
            AccessPolicy(
                collection="team",
                allowed_principals=(principal,),
                metadata={"department": "research", "secret": principal},
            ),
        )
        return DocumentRecord(
            document=document,
            asset=Asset(principal * 64, "pdf", f"pdfs/{document_id}.pdf"),
        )

    monkeypatch.setattr(
        "mm_asset_rag.ingest.asset_index.load_records",
        lambda: [record("visible", "a"), record("hidden", "b")],
    )
    args = build_parser().parse_args(
        [
            "documents",
            "--collection",
            "team",
            "--principal",
            "a",
            "--metadata-filter",
            '{"department":"research"}',
        ]
    )

    cli_mod.command_documents(args)

    payload = json.loads(capsys.readouterr().out)
    assert [row["document_id"] for row in payload] == ["visible"]
    assert "access_policy" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload)


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
