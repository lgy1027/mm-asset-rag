# Contributing to mm-asset-rag

Thanks for your interest in contributing. This document covers the basics.

## Development setup

The project is a flat-layout Python package — clone, install in editable mode, run.

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

Integration tests are marked with `@pytest.mark.integration` and excluded from the default test path (`testpaths = ["tests"]` in `pyproject.toml` plus the unit-only command above).

## Code style

- Formatter and linter: [`ruff`](https://docs.astral.sh/ruff/). Run `ruff check .` and `ruff format .` before opening a PR.
- Line length: 100 characters.
- Imports: sorted with `I` ruleset; no manual sorting needed.
- Type hints: encouraged but not enforced at CI time (yet).

## Project layout

```
mm-asset-rag/
├── mm_asset_rag/         # single Python package (flat layout)
│   ├── api.py            # FastAPI app: uploads, tasks, chat/stream, static UI
│   ├── cli.py            # `mmrag` / `mmrag-api` console scripts
│   ├── paths.py          # on-disk layout under $MM_ASSET_RAG_HOME
│   ├── assets.py         # asset_manifest loader + Asset dataclass
│   ├── pdf_parser.py     # PyMuPDF + PaddleOCR-VL backends
│   ├── image_parser.py   # OCR + VLM captioning for image assets
│   ├── qdrant_store.py   # Qdrant client, collection mgmt, incremental upsert
│   ├── embedding_config.py
│   ├── providers.py      # OpenAI-compatible embedder + image embedder
│   ├── retrieval.py      # hybrid merge + normalize
│   ├── answer.py         # grounded answer generation (streaming + sync)
│   ├── document_store.py # unified ParsedDocument JSONL store
│   ├── evaluation.py     # mini regression suite
│   ├── schema.py         # SearchHit, ParsedDocument
│   ├── config.py         # load_env() + env_bool()
│   └── web/              # bundled single-page web UI
│       └── index.html
├── examples/data/        # 30 PDFs + 184 photos + asset_manifest.json
├── tests/unit/           # offline unit tests (fast)
├── tests/integration/    # marked @pytest.mark.integration
├── docs/                 # architecture, configuration, api
└── scripts/              # eval_rag.py, build_manifest.py
```

If you want to plug in a different parser or vector backend, the practical swap points today are:

- **Different PDF backend** → extend `parse_pdf()` in `pdf_parser.py` with a new branch in the `parser in (...)` dispatch.
- **Different image embedding** → implement `ImageEmbeddingProvider` in `providers.py`.
- **Different vector backend** → today only Qdrant is wired. The retrieval layer (`retrieval.py`) calls `qdrant_*_search` helpers, so swapping the backend means adding a parallel set of functions and a dispatch table.

## Commit messages

This project follows a lightweight convention:

```
<type>(<scope>): <subject>

<body>
```

Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`. Scope is usually a module name (e.g. `qdrant_store`, `api`, `web`).

## Pull request checklist

- [ ] Tests pass (`pytest tests/unit -q`).
- [ ] `ruff check . && ruff format .` is clean.
- [ ] If user-facing behavior changed, `docs/api.md` / `README.md` updated.
- [ ] If a new environment variable was added, it shows up in `.env.example`.
- [ ] Commit messages follow the convention above (and do not include AI co-author trailers).

By submitting a patch you agree to license your contribution under Apache-2.0 (see [LICENSE](LICENSE)).