# Contributing to mm-asset-rag

Thanks for your interest in contributing. This document covers the basics.

## Development setup

The project uses a `src/` layout. You must install the package in editable mode before running tests — otherwise `pytest` will pick up un-installed source files and produce confusing import errors.

```bash
git clone https://github.com/lgy1027/mm-asset-rag.git
cd mm-asset-rag

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,clip]"
```

## Running tests

```bash
# Default: only the offline unit suite, suitable for CI
pytest tests/unit -q

# Include integration tests (require a local Qdrant binary and outbound network)
pytest tests/ -q
```

Integration tests are marked with `@pytest.mark.integration` and excluded from the default test path.

## Code style

- Formatter and linter: [`ruff`](https://docs.astral.sh/ruff/). Run `ruff check .` and `ruff format .` before opening a PR.
- Line length: 100 characters.
- Imports: sorted with `I` ruleset; no manual sorting needed.
- Type hints: encouraged but not enforced at CI time (yet).

## Project layout

```
src/mm_asset_rag/
├── parsers/        Parser Protocol + concrete PDF / image parsers
├── providers/      embedding, image_embedding, llm, ocr (HTTP-based integrations)
├── backends/       VectorBackend Protocol + qdrant / llamaindex implementations
└── retrieval/      hybrid merge + normalization helpers
```

When adding a new vector backend, implement the `VectorBackend` Protocol in `backends/base.py` and register it in `backends/__init__.py`.

When adding a new parser, implement the `Parser` Protocol in `parsers/base.py`.

## Commit messages

This project follows a lightweight convention:

```
<type>(<scope>): <subject>

<body>
```

Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`. Scope is usually a module name (e.g. `parsers`, `backends.qdrant`).