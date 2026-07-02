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
├── mm_asset_rag/         # single Python package (flat layout + sub-packages)
│   ├── api.py            # FastAPI app: thin route layer, delegates to service.py
│   ├── cli.py            # `mmrag` / `mmrag-api` console scripts, also delegate
│   ├── service.py        # IngestService: parse / index / task-history (shared)
│   ├── upload_pipeline.py# preview → confirm upload flow
│   ├── sniff.py          # file magic + local metadata detection
│   ├── auto_meta.py      # VLM JSON-mode metadata extraction
│   ├── settings.py       # pydantic-settings: every env var in one place
│   ├── protocols.py      # Parser / Embedder / VectorBackend Protocol definitions
│   ├── registry.py       # Module-level parsers / embedders / backends registries
│   ├── paths.py          # on-disk layout under $MM_ASSET_RAG_HOME
│   ├── config.py         # load_env() + env_bool() (legacy helpers)
│   ├── assets.py         # Asset dataclass
│   ├── schema.py         # SearchHit, ParsedDocument
│   ├── document_store.py # unified ParsedDocument JSONL store
│   ├── answer.py         # grounded answer generation (streaming + sync)
│   ├── evaluation.py     # mini regression suite
│   ├── retrieval.py      # hybrid merge + normalize (pure functions)
│   ├── parsers/          # Parser implementations, registered at import time
│   │   ├── pdf_parser.py # PyMuPDF + PaddleOCR-VL backends
│   │   └── image_parser.py
│   ├── embedders/        # Embedder implementations (Protocol conformers)
│   │   ├── text_embedder.py
│   │   └── image_embedder.py
│   └── backends/         # VectorBackend implementations
│       └── qdrant_backend.py
├── examples/             # API client examples
├── tests/unit/           # offline unit tests (fast)
├── tests/integration/    # marked @pytest.mark.integration
├── docs/                 # architecture, configuration, api
└── scripts/              # eval_rag.py, benchmark.py
```

## Adding a new modality (audio, video)

Three-line change, no central dispatch to edit:

1. Drop `parsers/audio_parser.py` whose class satisfies the
   `Parser` Protocol in `mm_asset_rag/protocols.py`.
2. In `parsers/__init__.py`, `register_parser(AudioParser())`.
3. Drop `embedders/audio_embedder.py` whose class satisfies the
   `Embedder` Protocol, and `register_embedder(...)` it.

The FastAPI app, the CLI, and the Qdrant backend all read from the
registries at runtime — no `if asset.source_type == "pdf"` branch ever
needs touching.

## Adding a different parser implementation

For a new PDF parser (e.g. a hypothetical `pdfplumber`):

```python
# parsers/pdf_parser.py — add a class:
class PdfPlumberParser:
    name = "pdfplumber"
    source_type = "pdf"
    def parse(self, asset, **options):
        ...
```

Register it in `parsers/__init__.py`:

```python
register_parser(PdfPlumberParser())
```

It is now selectable via `--pdf-parser pdfplumber` on the CLI. The web
upload flow still auto-sniffs file type first, then uses parser settings
inside the service layer.

## Adding a new vector backend

Implement the `VectorBackend` Protocol in `backends/<name>_backend.py` and
`register_backend(...)` it in `backends/__init__.py`.

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