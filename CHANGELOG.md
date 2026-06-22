# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Bundled single-page web UI (`mm_asset_rag/web/index.html`) for upload, task status, and streaming chat. Served by FastAPI at `/`.
- NDJSON streaming `/chat/stream` endpoint (sources → token* → done). Strips `<think>...</think>` blocks across chunk boundaries for reasoning models.
- `/upload` endpoint: multipart batch upload + same-hash deduplication + background parse / index thread; per-request form fields override `.env` defaults (`PDF_PARSER`, `ENABLE_OCR`, `ENABLE_VLM`, `IMAGE_PROVIDER`, `AUTO_INDEX`).
- `/tasks/{id}` and `/tasks` endpoints with persistent history (`$MM_ASSET_RAG_HOME/tasks.jsonl`). Tasks still in `running` state when the previous process exited are marked `interrupted` on startup.
- Incremental Qdrant indexing (`build_qdrant_text_index` / `build_qdrant_image_index`): only newly added documents are embedded and upserted; existing points are skipped via `client.retrieve(ids=...)`.
- Explicit `mmrag reindex` subcommand (with `--text-only` / `--image-only`) for force-rebuilding collections.
- FastAPI `lifespan` graceful shutdown closes the qdrant client (removes its `.lock`). Startup also tolerates a stale `.lock` from a crashed previous session.
- `docs/api.md` now documents `/upload`, `/chat/stream`, `/tasks`, and `interrupted` state.

### Changed
- `_run_parse_task` / `_run_ingest_task` now catch `BaseException` (not just `Exception`) so `SystemExit` / `KeyboardInterrupt` surface as `task.status="failed"` with the message in `task.error`, instead of silently leaving the task at `done` with stale state.
- `_run_ingest_task` now invokes `build_qdrant_text_index` / `build_qdrant_image_index` directly with a `progress_cb`, so the task's `current` text tracks parse / embed / upsert phases across the whole pipeline.
- README / `CONTRIBUTING.md` / `docs/api.md` rewritten to match the actual flat package layout and current endpoints (previously referenced a non-existent `src/mm_asset_rag/{parsers,backends}/` layout and a `POST /ingest` endpoint that no longer exists).
- `.env.example` documents every env var the code reads, including `EMBEDDING_MAX_INPUT_CHARS`, `QDRANT_BM25_MODEL`, `QDRANT_HYBRID_PREFETCH_LIMIT`, `PADDLEOCR_VL_POLL_INTERVAL`, `PADDLEOCR_VL_POLL_RETRY`, and the optional `OCR_HTTP_*` / `VLM_*` blocks.
- `pyproject.toml` no longer lists unused `llama-index-core` / `llama-index-embeddings-openai` dependencies; `python-multipart` is now a hard requirement for `/upload`.

### Removed
- `POST /ingest` endpoint (replaced by `POST /upload`).
- `src/mm_asset_rag/` layout documentation in `README.md` / `CONTRIBUTING.md` — package is a flat layout, not src-layout.
- Unused llama-index dependencies from `pyproject.toml`.

### Fixed
- `iter_lines(chunk_size=1, decode_unicode=True)` in `stream_answer_chunks` was shredding multi-byte UTF-8 into mojibake. Now reads raw bytes with `iter_content` and decodes UTF-8 manually.
- Qdrant `.lock` left over from a SIGKILL'd previous session blocking startup (handled by `_clean_stale_lock` + lifespan).
- `rejected({...})` was being called as a function in `/upload`'s empty-filename branch, crashing every upload that touched a blank filename.
- `_run_parse_task` was hard-coding `enable_ocr=False, enable_vlm=False` regardless of form / env; now reads from `ParseOptions`.

## [0.1.0] - 2026-06-21

### Added
- Initial extraction from `llamaindex-tutorial` project.
- Multimodal asset RAG pipeline: PDF parsing (PyMuPDF + PaddleOCR-VL), image OCR/VLM caption, text + image embedding, hybrid retrieval, LLM-grounded answering.
- Qdrant backend (default, local or server) with dense + BM25 hybrid retrieval via RRF.
- Retrieval modes: text, text-to-image, image-to-image, hybrid.
- CLI (`mmrag`) and FastAPI server (`mmrag-api`).
- Bundled 30-PDF + 184-photo sample set under `examples/data/chapter11_assets/`.

### Changed
- Image embedding provider supports `lite` (built-in) and `sentence-transformers` (optional `[clip]` extra).