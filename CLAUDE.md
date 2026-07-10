# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick reference

- Package: `mm-asset-rag`, flat layout under `mm_asset_rag/` plus `parsers/`, `embedders/`, `backends/`.
- Console scripts: `mmrag` (CLI) and `mmrag-api` (FastAPI on `127.0.0.1:8011`, web UI at `/`, OpenAPI at `/docs`).
- Python: 3.10 / 3.11 / 3.12.
- Upload-first architecture: no `asset_manifest.json`. Users upload files, the system sniffs type + metadata, optionally VLM-tags them, then parses and indexes confirmed previews.

## Common commands

```bash
# Setup
pip install -e ".[dev,clip]"          # [clip] optional; SentenceTransformers image embedder
pip install -e ".[docling]"           # [docling] optional; multi-format parser (docx/pptx/html/…)

# Tests
pytest tests/unit -q                  # offline suite; what CI runs
pytest tests/unit/test_upload_pipeline.py -q
pytest tests/unit/test_api.py -q
pytest tests/unit -q --cov=mm_asset_rag --cov-report=term

# Lint / format
ruff check .
ruff format .

# Run
mmrag-api                             # FastAPI server + web UI
mmrag parse ./paper.pdf ./photo.jpg   # sniff + parse + index files
mmrag reindex                         # drop + rebuild Qdrant collections
mmrag search "your query"             # mode=hybrid by default
mmrag answer "your question"
mmrag eval
```

## Architecture (one-screen summary)

The flow is `thin entry point → upload pipeline → service → registries → implementations`, with `Settings` as the only env-var reader.

- **Entry points** — `api.py` and `cli.py` stay thin. `/upload/preview` and `/upload/confirm` are the web flow; `mmrag parse <files...>` is the CLI equivalent.
- **`upload_pipeline.UploadPipeline`** — owns preview → confirm. Preview copies files to `.preview-cache`, calls `sniff.py` and optional VLM metadata extraction (`auto_meta.py`); confirm applies edits, moves files to `assets/pdfs` or `assets/images`, and constructs `Asset` objects.
- **`service.IngestService`** — owns background `threading.Thread` workers, parse + index orchestration, and `$MM_ASSET_RAG_HOME/tasks.jsonl` history. `dispatch_search` is the single source of truth for `text` / `text-to-image` / `image-to-image` / `hybrid` routing.
- **Protocols + registry** — `protocols.py` defines `Parser`, `Embedder`, `VectorBackend`; `registry.py` exposes keyed registries. PDF and image parsers both register in `parsers/__init__.py`.
- **`parsers/`** — PyMuPDF and PaddleOCR-VL for PDFs; OCR + VLM caption for images.
- **`embedders/`** — `TextEmbedder` (OpenAI-compatible) and `ImageEmbedder` (CLIP via `sentence-transformers` when `[clip]` is installed; built-in lite provider otherwise).
- **`backends/qdrant_backend.py`** — Qdrant local-file or remote-server. Active collection name auto-suffixes by vector dim. Text index fuses dense + BM25 + BM25-zh via RRF; image index uses CLIP vectors.
- **`retrieval.hybrid_search`** — pure (no I/O), normalizes per-route scores and merges with configurable weights (defaults text 0.80 / text-to-image 0.20 / image-to-image 0.0).
- **`answer.llm_answer` / `stream_answer_chunks`** — OpenAI-compatible chat completion with evidence context; falls back to an evidence summary when no LLM is configured.

## Configuration

Every env var the code reads is a typed field on `Settings` in `mm_asset_rag/settings.py`. `.env` in the cwd is loaded automatically by `config.load_env()`. New code should call `get_settings().foo` rather than `os.environ.get("FOO")`. The canonical reference is `docs/configuration.md`; the template is `.env.example`.

If you add a new env var: declare it on `Settings`, document it in `.env.example`, and (if user-facing) in `docs/configuration.md`.

### Upload pipeline

- `sniff.py` is pure local file inspection: magic bytes, PDF metadata/page count, image dimensions/EXIF.
- `auto_meta.py` uses VLM JSON mode for title / description / tags. If VLM is unconfigured or fails, upload falls back to sniff-only metadata.
- `/upload/preview` never indexes. `/upload/confirm` starts the ingest task.
- Do not reintroduce manifest-based ingestion unless the user explicitly asks for a separate import feature.

### Retrieval tuning

- `hybrid_weight_text` / `hybrid_weight_text_to_image` / `hybrid_weight_image_to_image` — weights passed to `merge_hits`.
- `max_chunks_per_pdf` — per-asset chunk cap applied during indexing; requires `mmrag reindex` to affect existing collections.
- `bm25_zh_enabled` / `bm25_zh_k1` / `bm25_zh_b` / `bm25_zh_vector_name` — Chinese-aware BM25 sparse vector.
- `image_relevance_threshold` / `image_prefilter_fields` / `image_prefilter_min_token_len` — image-route precision controls.

## Testing notes

- `tests/conftest.py` autouses `_clear_settings_cache` and provides `tmp_home`, `fake_qdrant_client`, and `fixed_vector` fixtures.
- Tests build temporary PDFs/images themselves; do not add dependencies on bundled sample corpora.
- `tests/integration/` is reserved for `@pytest.mark.integration` tests (real Qdrant binary or outbound network). CI only runs `tests/unit`.

## Common pitfalls

- **Qdrant local-file lock is single-process.** Stop the API server before running `mmrag reindex` against local mode, or use `QDRANT_URL` for concurrent access.
- **`get_settings()` is `lru_cache`-wrapped.** Env changes between tests require `get_settings.cache_clear()` (the conftest does this automatically).
- **`/chat/stream` with reasoning models.** `<think>...</think>` blocks are stripped across chunk boundaries; reasoning tokens never reach the client.
- **`register_*` is idempotent with `replace=True`** — used by tests to swap implementations.
- **Collection name auto-suffixes on dim change.** Changing embedding dimensionality creates a new collection; use `mmrag reindex` for a clean rebuild.

## Where to look

| If you're changing... | Read first |
| --- | --- |
| Upload preview / confirm | `mm_asset_rag/upload_pipeline.py`, `sniff.py`, `auto_meta.py`, `api.py`, `docs/upload-flow.md` |
| Env vars / typed settings | `mm_asset_rag/settings.py` + `docs/configuration.md` |
| HTTP endpoints / streaming | `mm_asset_rag/api.py` + `docs/api.md` |
| CLI subcommands | `mm_asset_rag/cli.py` |
| Background tasks, history, parse / index orchestration | `mm_asset_rag/service.py` |
| Retrieval math | `mm_asset_rag/retrieval.py` |
| Parser / embedder / backend extension | `mm_asset_rag/protocols.py`, `registry.py`, and matching sub-package `__init__.py` |
| Qdrant collection layout | `mm_asset_rag/backends/qdrant_backend.py` |
| Web UI | `mm_asset_rag/web/index.html` |

## Project conventions

- **Commit messages**: `<type>(<scope>): <subject>` (e.g. `fix(qdrant): ...`). No `Co-Authored-By` trailer and no AI-generation marker.
- **Lint/format**: `ruff` with line length 100 (`ruff.toml`). CI runs `ruff check .` and `ruff format --check .`.
- **Type hints**: encouraged but not enforced at CI time.
- **Layout is intentional**: flat `mm_asset_rag/` + three sub-packages. Don't migrate to `src/`.

## Knowledge graph (`graphify-out/`)

This project has a graphify knowledge graph at `graphify-out/`.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists.
- Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
