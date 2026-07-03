# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Eval expansion**: `mm_asset_rag/evaluation.py` now ships 32 English arxiv-paper queries + 6 Chinese cross-language queries (the original 3-case regression kept as a subset). `run_eval` resolves bare expected ids (`"Alexnet"`) to the actual hashed asset_ids via `asset_index.jsonl` so `aggregate_metrics` can score them with hit_rate / precision / recall / f1 / ndcg at k=1,3,5,10 + MRR + MAP. Per-query results + per-group aggregate metrics are persisted to `$MM_ASSET_RAG_HOME/eval_report.json`. Measured on the 32-paper chapter11 corpus: EN hit_rate@5=0.750, ZH=0.667, MRR=0.734 / 0.583.
- **v2 eval harness** (`mm_asset_rag/evaluation_v2.py` + `docs/eval-report-v2.md`): 83 Chinese-primary multi-dimensional cases across 3 modes. Findings (drives the 0.2.0 roadmap): text→text MRR=0.133, text→image hit_rate=0.087, image→image hit_rate=1.000. Surfaces three P0 issues — (a) image_parser writes a placeholder text chunk per image (filename + `图片标题/Picsum XXX`) into the text collection, polluting text→text recall; (b) CLIP ViT-B-32 is English-only so Chinese text→image returns 0; (c) `hybrid_search` has no `min_score` threshold so negative queries (`强化学习 / 联邦学习 / 元学习`) still return top-5 irrelevant hits instead of an empty / "I'm not sure" answer.
- **Image-route evals**: `evaluation.py` now also ships `TEXT_TO_IMAGE_QUERIES` (10 English + 3 Chinese on the Caltech-101 image set) and `IMAGE_TO_IMAGE_QUERIES` (6 categories). `run_text_to_image_eval` / `run_image_to_image_eval` hit the Qdrant image collection directly. Bare `Caltech <Category>` expected ids expand to all 3 sample asset_ids per category via `_expand_bare_expected_to_full`, so the strict set match in `aggregate_metrics` works correctly. Measured: text-to-image hit_rate@5=0.692 (10/13, with the 3 Chinese misses explained by CLIP ViT-B-32 being English-only), image-to-image hit_rate@5=1.000 (6/6 at rank 1).
- **`mmrag parse --no-auto-meta`** flag: skips the VLM-based title / tags / description extraction in the preview phase. Critical for ingesting hundreds of images where the per-file ollama round-trip dominates wall-clock (318 images × 30 s timeout ≈ 1 h hang). Backed by a module-level `_auto_meta_disabled` switch in `mm_asset_rag/upload_pipeline.py` plus `disable_auto_meta()` / `enable_auto_meta()` helpers (the latter for tests that need to toggle the switch).
- Bundled single-page web UI (`mm_asset_rag/web/index.html`) for upload, task status, and streaming chat. Served by FastAPI at `/`.
- NDJSON streaming `/chat/stream` endpoint (sources → token* → done). Strips `<think>...</think>` blocks across chunk boundaries for reasoning models.
- `/upload` endpoint: multipart batch upload + same-hash deduplication + background parse / index thread; per-request form fields override `.env` defaults (`PDF_PARSER`, `ENABLE_OCR`, `ENABLE_VLM`, `IMAGE_PROVIDER`, `AUTO_INDEX`).
- `/tasks/{id}` and `/tasks` endpoints with persistent history (`$MM_ASSET_RAG_HOME/tasks.jsonl`). Tasks still in `running` state when the previous process exited are marked `interrupted` on startup.
- Incremental Qdrant indexing (`build_qdrant_text_index` / `build_qdrant_image_index`): only newly added documents are embedded and upserted; existing points are skipped via `client.retrieve(ids=...)`.
- Explicit `mmrag reindex` subcommand (with `--text-only` / `--image-only`) for force-rebuilding collections.
- FastAPI `lifespan` graceful shutdown closes the qdrant client (removes its `.lock`). Startup also tolerates a stale `.lock` from a crashed previous session.
- `docs/api.md` now documents `/upload`, `/chat/stream`, `/tasks`, and `interrupted` state.
- **Retrieval tuning knobs** on `Settings` (`hybrid_weight_text`, `hybrid_weight_text_to_image`, `hybrid_weight_image_to_image`, `max_chunks_per_pdf`). The previously hard-coded `[0.55, 0.30, 0.15]` weights are now configured defaults `[0.80, 0.20, 0.0]` — tighter image-route weight prevents text-to-image pollution on pure text queries.
- **`_select_top_chunks_per_pdf`** — per-asset BM25 Okapi chunk cap. Selects the top-N chunks per asset (by score against the asset title) when `MAX_CHUNKS_PER_PDF` is set; otherwise preserves prior behaviour. Solves dense-embedding bias toward long PDFs (`clip` 48 chunks / `flamingo` 54 / `gpt3` 75 on the bundled set).
- **Chinese BM25** (`mm_asset_rag/bm25_zh.py`) — `jieba.cut` + Latin-mask + Okapi BM25 with deterministic sha1 term-to-index mapping. Stored as a third sparse vector (`bm25_zh`) alongside the existing `bm25` (English fastembed) and `dense` channels. `_hybrid_text_query` now prefetches all three and RRF-fuses them. IDF table is persisted to `$MM_ASSET_RAG_HOME/indexes/bm25_zh_idf.json`. Tunable via `BM25_ZH_ENABLED` / `BM25_ZH_K1` / `BM25_ZH_B`. Disable for English-only corpora to skip the indexing cost.
- **Cross-scenario evaluation harness** (`scripts/expand_corpus.py`, `scripts/eval_extended.py`, `scripts/eval_rag.py`). Adds 22 diverse public-domain PDFs (Wikipedia EN/ZH, arXiv, IRS forms, scan-style variants) and 39 ground-truth queries across 6 categories, used to verify retrieval accuracy on non-ML content. Without `MAX_CHUNKS_PER_PDF` (no L2), cross-scenario `hit_rate=0.974`; with `=10`, `hit_rate=1.000` (MRR `0.974`); English bundled set remains at `hit_rate=0.923` (no regression).
- **`scripts/prune_corpus.py`** — mirrors `expand_corpus.py` to clean up parsed markdown, `documents.jsonl` rows, and stale `tasks.jsonl` entries for assets that are no longer in `asset_manifest.json`. Default is dry-run; `--yes` to apply. Re-running `mmrag reindex` afterwards is the recommended way to rebuild the Qdrant index.
- **Schema-mismatch detection** in `_create_collection`: when the existing Qdrant text collection's sparse-vector config doesn't match the current `Settings` (e.g. toggling `BM25_ZH_ENABLED` without reindexing), the indexer now fails fast with a clear "Run `mmrag reindex`" message instead of silently writing partial points.

### Changed
- `_run_parse_task` / `_run_ingest_task` now catch `BaseException` (not just `Exception`) so `SystemExit` / `KeyboardInterrupt` surface as `task.status="failed"` with the message in `task.error`, instead of silently leaving the task at `done` with stale state.
- `_run_ingest_task` now invokes `build_qdrant_text_index` / `build_qdrant_image_index` directly with a `progress_cb`, so the task's `current` text tracks parse / embed / upsert phases across the whole pipeline.
- README / `CONTRIBUTING.md` / `docs/api.md` rewritten to match the actual flat package layout and current endpoints (previously referenced a non-existent `src/mm_asset_rag/{parsers,backends}/` layout and a `POST /ingest` endpoint that no longer exists).
- `.env.example` documents every env var the code reads, including `EMBEDDING_MAX_INPUT_CHARS`, `QDRANT_BM25_MODEL`, `QDRANT_HYBRID_PREFETCH_LIMIT`, `PADDLEOCR_VL_POLL_INTERVAL`, `PADDLEOCR_VL_POLL_RETRY`, the optional `OCR_HTTP_*` / `VLM_*` blocks, and the new `BM25_ZH_*` / `HYBRID_WEIGHT_*` / `MAX_CHUNKS_PER_PDF` retrieval-tuning fields.
- `pyproject.toml` no longer lists unused `llama-index-core` / `llama-index-embeddings-openai` dependencies; `python-multipart` is now a hard requirement for `/upload`. Adds `jieba>=0.42` for `bm25_zh`.
- `CLAUDE.md` now has a "Retrieval tuning" and "OCR pipelines" section documenting the four retrieval knobs and the two OCR paths (PaddleOCR-VL for PDFs, `OCR_HTTP_URL` for images).

### Removed
- `POST /ingest` endpoint (replaced by `POST /upload`).
- `src/mm_asset_rag/` layout documentation in `README.md` / `CONTRIBUTING.md` — package is a flat layout, not src-layout.
- Unused llama-index dependencies from `pyproject.toml`.
- `examples/data/chapter11_assets/` — bundled sample assets (≈30 PDFs + 184 photos) removed from the repo; upload-first design means users bring their own files via `/upload/preview`. Developers who need the corpus locally can `git checkout HEAD -- examples/data/` (kept locally via `.gitignore`).
- `scripts/build_manifest.py`, `expand_corpus.py`, `prune_corpus.py`, `import_caltech101.py`, `import_vlm_captions.py`, `eval_extended.py`, `eval_multimodal.py` — manifest-era utilities superseded by the upload-first pipeline.
- `docs/asset-store-design.md`, `docs/multimodal-eval-report.md` — superseded by `docs/upload-flow.md` + `docs/eval-report.md`.
- `tests/unit/test_assets.py` — covered by the service-layer test suite after the upload-first refactor.

### Fixed
- **v4 0.2.0: 5 阶段全量优化 (代码 ready,reindex 需手动)**:
  - **P1 multilingual embedding** — new `Settings.embedding_backend: sentence_transformers` + `SentenceTransformerTextEmbedder` 支持 `BAAI/bge-m3` / `intfloat/multilingual-e5-large` 等本地 HF model。`build_default_text_embedder()` 工厂按 Settings 选 backend,默认仍是 OpenAI 兼容。
  - **P2 query preprocessing** — 新 `mm_asset_rag.query_preprocess` 模块:lowercase / fuzzy typo corrector(difflib, vocab 从 `documents.jsonl` 构建)/ 同义词 expansion (JSON file 驱动)。`qdrant_text_search` 走预处理器,BM25 channel 用预处理版,dense 保留原版(多语言 embedding 大小写敏感)。
  - **P3 per-channel RRF 权重** — `Settings.rrf_weight_dense / rrf_weight_bm25 / rrf_weight_bm25_zh` 默认 1.0/1.0/1.0。`qdrant-client 1.18` 不支持 per-prefetch weight,workaround 是 `_channel_limit(weight)` 按权重 scale prefetch `limit`(1.0→20, 1.5→30, 4.0 封顶)。
  - **P4 Chinese-CLIP backbone** — `mmrag reindex --yes` flag 跳过交互式确认(CI / 脚本用);`.env.example` 加 `CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16` (768d, ZH+EN) 切换说明。
  - **P5 PDF chunk-by-section + 关键词** — 新 `parsers/chunk_splitter.split_by_heading`(ATX # / font-size ≥ 1.4× / 双边界短行三规则),PyMuPDF 路径每页拆 N chunks,带 `metadata.section` + `chunk_index`。`text_keywords.extract_keywords` (zh=en=auto) 给 chunk append `关键词: ...` footer,BM25 channel 显式得 token(`联宝` / `ESG`)。空 body section 自动跳过,避免 placeholder 污染。
- **v3 eval: 3 P0 weaknesses closed** — text→text hit_rate@5 lifted from 0.220 → 0.300, and 5/8 negative queries now return an empty list. (a) `parse_image` no longer writes a placeholder text chunk to the text collection when title / tags / VLM caption / OCR text are all empty (Picsum photos no longer pollute text→text recall); (b) `qdrant_text_search` now applies a `source_type=pdf` filter to every RRF prefetch so legacy image-source placeholders are kept out of text→text even without a reindex; (c) `merge_hits` and `hybrid_search` take a `min_score` confidence floor (env: `MIN_SCORE`, default `0.30`) and the v2 eval script now strips the trailing `_<8-hex-hash>` from asset_ids so a bare id in `expected` matches every hash variant of the same content. The Chinese-CLIP recommendation is in `.env.example` (`CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16`) for users who want to push text→image from 8.7% to ~70%; the English-only default is unchanged.
- `iter_lines(chunk_size=1, decode_unicode=True)` in `stream_answer_chunks` was shredding multi-byte UTF-8 into mojibake. Now reads raw bytes with `iter_content` and decodes UTF-8 manually.
- Qdrant `.lock` left over from a SIGKILL'd previous session blocking startup (handled by `_clean_stale_lock` + lifespan).
- `rejected({...})` was being called as a function in `/upload`'s empty-filename branch, crashing every upload that touched a blank filename.
- `_run_parse_task` was hard-coding `enable_ocr=False, enable_vlm=False` regardless of form / env; now reads from `ParseOptions`.
- **`pdf/auto` parser registry**: `parsers/__init__.py` now registers an `_AutoPdfParser` (dispatches to `parse_pdf(..., parser="auto")`) so `ParseOptions.pdf_parser="auto"` and `mmrag parse` with the default no longer fail with `KeyError: parser ('pdf', 'auto') not registered`.
- **Daemon thread killed by early `done` status**: `_run_parse_task` flipped the task record to `status="done"` + `finished_at=...` as soon as parsing completed, so the CLI's `_wait_for_task` saw a terminal state and exited the process. The Qdrant index step runs in a daemon thread inside that process, so `upsert_text` never got to run. `_run_ingest_task` now resets `status="running"` + `finished_at=None` after parse completes and lets the index step restore the terminal state, so the daemon survives until the index is built. Previously: no Qdrant collections, zero points. After fix: 32 PDFs → 1378 chunks → `multimodal_text_2560d` populated.
- **text-to-image search crashes when no image collection exists**: `qdrant_text_to_image_search` now catches the `ValueError: Collection ... not found` from `qdrant_client.query_points` and returns `[]` instead of propagating. Lets the user run text search on a PDF-only corpus without a separate `text_only` flag.

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