"""Regression checks for executable user-facing documentation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _section(text: str, title: str, next_title: str) -> str:
    return text.split(title, 1)[1].split(next_title, 1)[0]


def test_ingest_examples_include_required_access_policy_arguments() -> None:
    """Every documented CLI ingest command supplies its required ACL inputs."""
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "quickstart.md",
        ROOT / "docs" / "upload-flow.md",
    )
    for document in documents:
        parse_lines = [
            line.strip()
            for line in document.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("mmrag parse ")
        ]
        assert parse_lines, f"{document} must contain a CLI ingest example"
        assert all("--collection" in line and "--principal" in line for line in parse_lines), (
            f"{document} has an ingest example without its required access policy"
        )


def test_cli_examples_include_their_required_arguments() -> None:
    """Every documented retrieval command is executable without hidden ACL flags."""
    required = {
        "mmrag parse ": ("--collection", "--principal"),
        "mmrag search ": ("--collection", "--principal"),
        "mmrag eval ": ("--collection", "--principal"),
        "mmrag answer ": ("--collection", "--principal", "--min-confidence"),
    }
    for document in (ROOT / "README.md", ROOT / "README.zh-CN.md", *(ROOT / "docs").glob("*.md")):
        for line in document.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            for command, flags in required.items():
                if stripped.startswith(command):
                    assert all(flag in stripped for flag in flags), f"{document}: {stripped}"


def test_api_examples_include_required_access_context() -> None:
    """Answer, chat, and evaluation docs show their required request fields."""
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    answer = _section(api, "## `POST /answer`", "## `POST /chat`")
    chat = _section(api, "## `POST /chat`", "## `POST /chat/stream`")
    evaluation = _section(api, "## `POST /eval`", "## Evaluation metrics")
    for text in (answer, chat):
        for field in ('"collection"', '"principal"', '"min_confidence"'):
            assert field in text
    assert "| `collection` | required" in evaluation
    assert "| `principal` | required" in evaluation


def test_user_docs_describe_tables_and_current_document_model() -> None:
    """Published docs include table formats and avoid removed version terminology."""
    docs = {
        path: path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            ROOT / "README.zh-CN.md",
            ROOT / "docs" / "api.md",
            ROOT / "docs" / "upload-flow.md",
            ROOT / "docs" / "configuration.md",
            ROOT / "docs" / "architecture.md",
        )
    }
    for path, text in docs.items():
        assert "csv" in text.lower() and "tsv" in text.lower(), path
        assert "ParsedDocument" not in text, path
        assert "latest versions" not in text, path
        assert "DELETE /assets" not in text, path


def test_image_eval_commands_are_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    zh_readme = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    api_docs = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    assert "mmrag ingest-image-eval" in readme
    assert "mmrag eval" in readme and "--image" in readme
    assert "mmrag ingest-image-eval" in zh_readme
    assert "mmrag eval" in zh_readme and "--image" in zh_readme
    assert '"image": true' in api_docs
    assert "primitive_image_routes" in api_docs
