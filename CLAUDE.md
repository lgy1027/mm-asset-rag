# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick reference

- Package: `mm-asset-rag`, flat layout under `mm_asset_rag/` plus three implementation sub-packages: `parsers/`, `embedders/`, `backends/`.
- Console scripts: `mmrag` (CLI) and `mmrag-api` (FastAPI on `127.0.0.1:8011`, web UI at `/`, OpenAPI at `/docs`).
- Python: 3.10 / 3.11 / 3.12 (CI matrix in `.github/workflows/test.yml`).
- Bundled sample data: `examples/data/chapter11_assets/` (30 PDFs + 184 photos + `asset_manifest.json`). Tests reuse it via the `examples_home` fixture — there are no mock embedding backends by default.

## Common commands

```bash
# Setup
pip install -e ".[dev,clip]"          # [clip] optional; only for the SentenceTransformers image embedder

# Tests
pytest tests/unit -q                  # offline suite; what CI runs
pytest tests/unit/test_retrieval.py -q
pytest tests/unit/test_api.py::test_health -q
pytest tests/unit -q --cov=mm_asset_rag --cov-report=term

# Lint / format (CI runs both `check` and `format --check`)
ruff check .
ruff format .

# Run
mmrag-api                             # FastAPI server
mmrag parse --pdf-parser pymupdf --vlm
mmrag index                           # incremental; skips already-indexed docs
mmrag reindex                         # drop + rebuild (see pitfall below)
mmrag search "your query"             # mode=hybrid by default
mmrag answer "your question"
mmrag eval
```

## Architecture (one-screen summary)

The flow is `thin entry point → service → registries → implementations`, with `Settings` as the only env-var reader.

- **Entry points** — `api.py` (FastAPI routes + `lifespan` startup) and `cli.py` (`mmrag` argparse). Both are thin wrappers around the same `IngestService`, so the parse / index pipeline only lives in one place.
- **`service.IngestService`** — owns background `threading.Thread` workers, parse + index orchestration, and `$MM_ASSET_RAG_HOME/tasks.jsonl` history. `dispatch_search` in `service.py` is the single source of truth for the four `mode` routes (`text` / `text-to-image` / `image-to-image` / `hybrid`); use it instead of calling backend search helpers directly.
- **Protocols + registry** — `protocols.py` defines `Parser`, `Embedder`, `VectorBackend` (`@runtime_checkable`). `registry.py` exposes keyed registries and `register_*` / `get_*` helpers. **Adding a modality is a three-line change**: drop the implementation in `parsers/` or `embedders/`, call `register_*` in that sub-package's `__init__.py`. No central dispatch table needs editing — `api.py`, `cli.py`, and `service.py` all read from the registries at runtime.
- **`parsers/`** — `pymupdf` and `paddleocr_vl` for PDFs, OCR + VLM caption for images. Selectable per `/upload` and via `--pdf-parser` on the CLI.
- **`embedders/`** — `TextEmbedder` (OpenAI-compatible) and `ImageEmbedder` (CLIP via `sentence-transformers` when `[clip]` is installed; built-in lite provider otherwise).
- **`backends/qdrant_backend.py`** — Qdrant local-file or remote-server. Active collection name auto-suffixes by vector dim (`multimodal_text_2560d`), so changing the embedding model picks up a fresh collection. Text index fuses dense + BM25 via RRF; image index uses CLIP vectors. `QdrantLockHeldError` surfaces when another process holds the local-file lock.
- **`retrieval.hybrid_search`** — pure (no I/O), normalizes per-route scores and merges with weights text 0.55 / text-to-image 0.30 / image-to-image 0.15.
- **`answer.llm_answer` / `stream_answer_chunks`** — OpenAI-compatible chat completion with the retrieved evidence as context. When `settings.has_llm` is False (any of `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` missing), the API returns an evidence-summary fallback instead of failing — useful for offline eval.

## Configuration

Every env var the code reads is a typed field on `Settings` in `mm_asset_rag/settings.py`. `.env` in the cwd is loaded automatically by `config.load_env()`. New code should call `get_settings().foo` rather than `os.environ.get("FOO")`. The canonical reference is `docs/configuration.md`; the template is `.env.example`. **If you add a new env var: declare it on `Settings`, document it in `.env.example`, and (if user-facing) in `docs/configuration.md`.**

### Retrieval tuning

Four knobs in `Settings` control how `hybrid_search` and `build_qdrant_text_index` behave:

- `hybrid_weight_text` / `hybrid_weight_text_to_image` / `hybrid_weight_image_to_image` — weights passed to `merge_hits`. Defaults `0.80 / 0.20 / 0.0` (tightened from the historical `0.55 / 0.30 / 0.15` because text-to-image was polluting pure text queries).
- `max_chunks_per_pdf` — per-asset chunk cap applied during indexing. None keeps every chunk; a positive int keeps only the top-N chunks per asset (selected by a local BM25 Okapi score against the asset title). **Requires `mmrag reindex` to take effect** because the existing Qdrant collection already has the un-capped chunks.
- `bm25_zh_enabled` / `bm25_zh_k1` / `bm25_zh_b` / `bm25_zh_vector_name` — toggle and tune the Chinese-aware BM25 sparse vector (`jieba` + Okapi). Stored alongside the English `Qdrant/bm25` and the dense embedding; all three are prefetched and RRF-fused at query time. Disable for English-only corpora to skip the indexing cost.

### OCR pipelines

Two paths exist for getting searchable text out of image content:

- **PDFs** — `PDF_PARSER=paddleocr_vl` (or `auto` with `PADDLEOCR_VL_API_TOKEN` set). Goes through the PaddleOCR-VL online API; suitable for scanned PDFs without an embedded text layer. Each page lands as `parsed/<asset_id>/page_N.md`.
- **Images** — `ENABLE_OCR=true` (per `/upload`) plus `OCR_HTTP_URL=http://127.0.0.1:8000/ocr`. POSTs base64 to a local OCR service; normalised blocks land at `parsed/<asset_id>/ocr.json`.

Either parser path is a prerequisite for retrieving against scanned or image-only assets — without it the Qdrant collection will have no text rows for those documents.

## Testing notes

- `tests/conftest.py` autouses `_clear_settings_cache` (clears the `lru_cache`-wrapped `get_settings()` around each test) and provides `examples_home` (fresh tmp copy of the bundled samples), `tmp_home` (empty tmp `MM_ASSET_RAG_HOME`), and `fake_qdrant_client` fixtures.
- Tests do **not** mock embedding APIs by default — they exercise the real sample data. To pin vectors deterministically, use the `fixed_vector` fixture (it monkeypatches `TextEmbedder.embed*` and `ImageEmbedder.embed_*`).
- `tests/integration/` is reserved for `@pytest.mark.integration` tests (real Qdrant binary or outbound network). CI only runs `tests/unit`.

## Common pitfalls

- **Qdrant local-file lock is single-process.** Stop the API server (or any other `mm-asset-rag` process) before running `mmrag reindex` against local mode, or use `QDRANT_URL` (server mode) for concurrent access. A stale `.lock` after SIGKILL is cleaned on the next startup by `_clean_stale_lock` in `qdrant_backend.py`.
- **`get_settings()` is `lru_cache`-wrapped.** Env changes between tests require `get_settings.cache_clear()` (the conftest does this automatically). Production code never clears it. To bypass `.env` in tests, instantiate `Settings(_env_file=None)`.
- **`/chat/stream` with reasoning models.** `<think>...</think>` blocks emitted by DeepSeek-R1 / Qwen3-Thinking / etc. are stripped across chunk boundaries in `stream_answer_chunks`; reasoning tokens never reach the client.
- **`register_*` is idempotent with `replace=True`** — used by tests to swap implementations. Production `parsers/__init__.py`, `embedders/__init__.py`, and `backends/__init__.py` register exactly once per implementation.
- **Collection name auto-suffixes on dim change.** When the active embedding model changes dimension, a new collection is created (e.g. `multimodal_text_2560d`) — the old one is not auto-dropped. Use `mmrag reindex` if you want to rebuild from scratch.

## Where to look

| If you're changing... | Read first |
| --- | --- |
| Env vars / typed settings | `mm_asset_rag/settings.py` + `docs/configuration.md` |
| HTTP endpoints / streaming | `mm_asset_rag/api.py` + `docs/api.md` |
| CLI subcommands | `mm_asset_rag/cli.py` |
| Background tasks, history, parse / index orchestration | `mm_asset_rag/service.py` |
| Retrieval math (RRF, weights, normalization) | `mm_asset_rag/retrieval.py` |
| Adding a parser / embedder / backend | `mm_asset_rag/protocols.py`, `registry.py`, and the matching sub-package's `__init__.py` |
| Qdrant collection layout / dense+sparse hybrid | `mm_asset_rag/backends/qdrant_backend.py` |
| The bundled web UI | `mm_asset_rag/web/index.html` (single-file, served at `/`) |

## Project conventions

- **Commit messages**: `<type>(<scope>): <subject>` (e.g. `fix(qdrant): ...`, `refactor(api+cli): ...`). No `Co-Authored-By: Claude` trailer and no `Generated with Claude Code` line — see `CONTRIBUTING.md`.
- **Lint/format**: `ruff` with line length 100 (`ruff.toml`). CI runs `ruff check .` and `ruff format --check .` on every PR.
- **Type hints**: encouraged but not enforced at CI time.
- **Layout is intentional**: flat `mm_asset_rag/` + the three sub-packages. Don't migrate to a `src/` layout.

## Knowledge graph (`graphify-out/`)

This project has a graphify knowledge graph built at `graphify-out/` (1375 nodes, 2128 edges, 139 communities). It indexes the Python code (AST), docs, PDFs under `examples/data/chapter11_assets/pdfs/`, and the bundled photos.

**When to use the graph** — call `graphify query "<question>"` (or `path` / `explain`) before answering:

- "How does X work" / "What calls Y" / "Trace the data flow through Z"
- "Why did we pick X over Y"
- Cross-module impact analysis before refactoring
- Anything where you're about to reason about module topology, call chains, or architectural decisions

The graph is the source of truth for this project's internals — its specifics are not in pre-training data, so guessing is unreliable. Prefer citing `source_location` from the graph output over re-deriving from memory.

The graph auto-stays-fresh via `.git/hooks/post-commit` (`graphify --update` runs after every commit). To rebuild from scratch use `graphify` (full pipeline). For ad-hoc deep inspection, open `graphify-out/graph.html` in a browser.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
