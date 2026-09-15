# mm-asset-rag

> Multimodal knowledge base — index documents and images, then automatically choose document, image, or image-to-image retrieval. Grounded answers include their evidence and associated in-document figures.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](.github/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25-yellow)](tests/)

[English](README.md) | [中文](README.zh-CN.md)

## At a glance

```
                  ┌─────────────────────────────────────────────────┐
                  │          $ mmrag-api  (FastAPI + Web UI)        │
                  └───────────────┬─────────────────────────────────┘
                                  drag / POST /upload/preview
                                          ▼
   ┌──────────────────────┐   POST /upload/confirm    ┌──────────────────┐
   │  .preview-cache/<id> │ ─────────────────────────▶│  assets/pdfs     │
   │   (sniff + VLM meta) │   background task        │  assets/images   │
   └──────────────────────┘                          │  assets/documents │
                                                      └─────────┬────────┘
                                                                │ parse
                                                                ▼
                                                  ┌──────────────────────┐
                                                  │   documents.jsonl    │
                                                  └─────────┬────────────┘
                                                                │ embed
                                                                ▼
                  ┌─────────────────────────────────────────────────┐
                  │                  Qdrant (local/server)           │
                  │  multimodal_text_<dim>d    multimodal_image_<dim>d│
                  │   dense · bm25 · bm25_zh      CLIP / CN-CLIP     │
                  └───────────────┬─────────────────────────────────┘
                                  │ query (Auto / Documents / Images)
                                  ▼
                  ┌─────────────────────────────────────────────────┐
                  │  RRF 融合 → optional rerank → /answer or /chat  │
                  └─────────────────────────────────────────────────┘
```

The application exposes three user-facing choices, while the dispatcher selects the appropriate internal retrieval route:

```
  query ─┬─ Documents            ──▶ document evidence retrieval
         ├─ Images               ──▶ image-aware retrieval
         ├─ uploaded query image ──▶ image-to-image retrieval
         └─ Auto (default)       ──▶ chooses and fuses the relevant evidence
```

> Looking for a hands-on walkthrough with screenshots of the web UI? See [docs/quickstart.md](docs/quickstart.md).

## What is this?

A small, self-contained Python package for **multimodal retrieval** over user-uploaded assets — PDFs, Office documents (docx/pptx/xlsx), and images. The retrieval engine is the core; generation is an optional layer on top. It supports:

- **Intent-aware retrieval**: users choose Auto, Documents, or Images; Auto combines text evidence and image metadata when appropriate, while image upload enables image-to-image search.
- **Cross-modal retrieval**: embedded figures in PDFs and Office docs are extracted and (optionally) given VLM captions so a text query can hit a figure-only slide; a `find images similar to this one` query hits the CLIP image collection. The same asset store feeds both.
- **Upload-first ingestion**: no `asset_manifest.json`. `/upload/preview` sniffs file magic bytes, extracts dimensions / PDF metadata, optionally asks a VLM for title / description / tags, then `/upload/confirm` parses and indexes.
- **Parsing**: PyMuPDF (local, default) or PaddleOCR-VL (API, better for scanned PDFs) or docling (local, layout-aware) for PDFs; MarkItDown (default) or docling for Office docs (docx/pptx/xlsx/html); OCR + VLM captioning for images.
- **Indexing**: Qdrant (local file or server). Text points carry dense + BM25 + Chinese-aware BM25-zh sparse vectors; image points carry CLIP vectors.
- **Optional generation**: OpenAI-compatible chat completion with strict evidence grounding and NDJSON streaming. When no LLM is configured, `/answer` and `/chat` return an evidence summary instead of failing — retrieval still works.
- **Web UI**: a bundled single-page HTML (`mm_asset_rag/web/index.html`) served by FastAPI for upload preview, task status, and chat.

VLM-based auto-tagging is also optional; upload still works with sniff-only metadata.

## Why this project?

If you have a folder of mixed assets — papers, slide decks, photos, diagrams — and want to ask *"find images similar to this one"*, *"which document covers retrieval-augmented generation?"*, or *"show me the slide whose only content is a roadmap diagram"*, this project provides an intent-aware retrieval workflow with independently testable layers.

It is a **modular multimodal knowledge base** whose retrieval and answer layers are independently testable.

Compared to larger frameworks:

- **vs LlamaIndex Studio / Verba**: this ships with a web UI, is multimodal-retrieval-first rather than text-RAG-first, and keeps every module under 2k lines.
- **vs Haystack / txtai**: smaller surface area, four-route retrieval baked in from day one, easier to read end-to-end.

## Installation

Install the latest release from PyPI:

```bash
pip install mm-asset-rag   # core: text retrieval + FastAPI web UI (image indexing needs [clip] extra)
```

Optional CLIP-based image embeddings (recommended if you want text→image / image→image routes on real image corpora):

```bash
pip install "mm-asset-rag[clip]"     # sentence-transformers CLIP
```

Optional multi-format Office document parsing (docx/pptx/xlsx/html) beyond the default MarkItDown:

```bash
pip install "mm-asset-rag[docling]"  # layout-aware docling parser (heavier, pulls torch/transformers)
```

For local development from source:

```bash
git clone https://github.com/lgy1027/mm-asset-rag
cd mm-asset-rag
pip install -e ".[dev,clip]"
```

Or with [uv](https://docs.astral.sh/uv/) (reproducible installs from the committed `uv.lock`):

```bash
uv sync --extra dev
```

## Quick start

> **第一次用?** 先看 [docs/quickstart.md](docs/quickstart.md) —— 从零搭环境(ollama + bge-m3 + Qdrant 本地)到第一次 `mmrag search` 出结果的 30 分钟路径,含新手常见坑。下面的 Quick start 假定环境已配好。

```bash
# 1. Start the API + web UI
mmrag-api
# → http://127.0.0.1:8011/
# → http://127.0.0.1:8011/docs

# 2. Open the web UI, drag PDFs/images, review the preview cards,
#    edit title/tags if needed, then click Confirm & Ingest.

# 3. Search / answer from CLI after ingest completes
mmrag search "which document covers retrieval-augmented generation?" --collection default --principal local-user
mmrag answer "which document covers retrieval-augmented generation?" --collection default --principal local-user --min-confidence 0.5
```

CLI ingestion is also upload-first (PDFs, images, Office docs, and tables —
docx/pptx/xlsx/html/md/txt/csv/tsv):

```bash
mmrag parse ./paper.pdf ./photo.jpg ./deck.pptx --collection default --principal local-user
mmrag reindex
mmrag search "find the beach photo" --collection default --principal local-user
```

> **Qdrant local-file lock is single-process.** While `mmrag-api` is running, run `mmrag reindex` from another terminal and it will fail with a "storage already accessed" lock error. Either stop the API first, or point `QDRANT_URL` at a Qdrant server for concurrent access.

**Task control:** a long parse/index task can be cancelled cooperatively — `POST /tasks/{id}/cancel` sets a stop flag the worker checks between assets (it finishes the current asset, then stops and marks the task `cancelled`). `mmrag retry` re-runs the remaining assets.

**Health check:** `GET /health` returns liveness + index state; `GET /health?deep=true` adds `llm_configured` / `embedder_configured` (config-completeness, no LLM call / no quota) so an orchestrator can tell whether `/answer` and `/search` will work.

## Upload flow

```
POST /upload/preview (multipart files)
  ├─ stream files into .preview-cache/
  ├─ sniff magic bytes: pdf / image / unsupported
  ├─ extract local metadata: PDF /Info, page count, image size, EXIF
  ├─ optional VLM JSON mode: title / description / tags
  └─ return editable preview cards

POST /upload/confirm (cache_id + edited previews)
  ├─ move confirmed files into assets/pdfs, assets/images, or assets/documents
  ├─ parse PDF/image/document into documents.jsonl
  ├─ upsert text chunks into Qdrant text collection
  └─ upsert image vectors into Qdrant image collection
```

## Configuration

All settings come from environment variables (a `.env` file in the current directory is loaded automatically). The most important ones:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | Where to put uploaded assets, parsed data, indexes, task log. | `~/.mm_asset_rag` |
| `MODEL_API_KEY` / `MODEL_BASE_URL` / `LLM_MODEL` | Optional LLM for `/answer` and `/chat`. | — |
| `EMBEDDING_*` | Text embedding provider (defaults to OpenAI-compatible). | — |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server mode (omit to use local file mode). | — |
| `CLIP_MODEL` | Sentence-transformers CLIP model name (with `[clip]` extra). | `clip-ViT-B-32` |
| `VLM_BASE_URL` / `VLM_API_KEY` / `VLM_MODEL` | VLM for upload auto-tagging and image captions. Falls back to `MODEL_*`. | — |
| `AUTO_META_ENABLED` | Enable VLM title/description/tag extraction during upload preview. | `true` |
| `PADDLEOCR_VL_API_TOKEN` | PaddleOCR-VL API token for scanned PDFs. | — |
| `OCR_BACKEND` | Image OCR backend: `local` (PP-OCRv6 via `[ocr]` extra) or `http`. | `local` |
| `OCR_HTTP_URL` | External OCR endpoint (only used when `OCR_BACKEND=http`). | — |

See [`.env.example`](.env.example) and [`docs/configuration.md`](docs/configuration.md) for the full list.

## Evaluation

`mmrag eval` scores grouped query cases against exact logical document IDs and reports document-level Recall, MRR, MAP, and graded NDCG. Each case has a `query_id` and `query`; one top-level `qrels` object maps every query ID to `{document_id: relevance}`. The **default** is a small qrels sample shipped in `mm_asset_rag/eval_data/`. Matching documents must already be ingested under those exact, case-sensitive document IDs; otherwise the cases are reported as misses.

```json
{
  "version": "v1",
  "groups": {"en": [{"query_id": "q1", "query": "..."}]},
  "qrels": {"q1": {"document-id": 3}}
}
```

To score your own corpus, author a case file and pass `--cases` (or set `EVAL_CASES_PATH`):

```bash
# 1. Ingest your eval corpus with document IDs matching the qrels.
mmrag parse ./my_eval_corpus/*.pdf --collection default --principal local-user
# 2. Run the evaluation
mmrag eval --collection default --principal local-user                              # bundled default sample
mmrag eval --cases my_cases.json --collection default --principal local-user        # your own case set
mmrag eval --v2 --collection default --principal local-user                         # v2: multi-dimensional, Chinese-primary
```

When no LLM is configured, the eval still runs (it measures retrieval only); `/answer`-dependent cases degrade gracefully.

### Quick perf check

Once you have a corpus of any size, get a real p50 / p95 / QPS for your machine before tuning weights:

```bash
# stop mmrag-api first (Qdrant local is single-process)
uv run python scripts/benchmark.py --top-k 5 --n-runs 50
# → writes $MM_ASSET_RAG_HOME/benchmark_report.json + a stdout table
```

The benchmark hits the public `hybrid_search` path — no private helpers — so numbers track `Settings` changes (reranker on/off, `MAX_CHUNKS_PER_PDF`, etc.). Full step-by-step on getting from zero to first search: [`docs/quickstart.md`](docs/quickstart.md).

## Project layout

```
mm-asset-rag/
├── mm_asset_rag/         # single Python package (flat layout + sub-packages)
│   ├── api.py            # FastAPI app: thin route layer, delegates to service.py
│   ├── cli.py            # `mmrag` / `mmrag-api` console scripts
│   ├── service.py        # IngestService: parse / index / task-history
│   ├── upload_pipeline.py# preview → confirm upload flow
│   ├── sniff.py          # file magic + local metadata detection
│   ├── auto_meta.py      # VLM JSON-mode metadata extraction
│   ├── settings.py       # pydantic-settings: every env var in one place
│   ├── protocols.py      # Parser / Embedder / VectorBackend Protocol definitions
│   ├── registry.py       # Module-level parsers / embedders / backends registries
│   ├── paths.py          # on-disk layout under $MM_ASSET_RAG_HOME
│   ├── assets.py         # Asset dataclass
│   ├── schema.py         # public retrieval schemas
│   ├── document_store.py # parsed chunk JSONL store
│   ├── answer.py         # grounded answer generation (streaming + sync)
│   ├── evaluation.py     # mini regression suite
│   ├── retrieval.py      # hybrid merge + normalize
│   ├── parsers/          # PDF/image parser implementations
│   ├── embedders/        # text/image embedder implementations
│   └── backends/         # Qdrant backend implementation
├── tests/unit/           # offline unit tests
├── tests/integration/    # marked @pytest.mark.integration
├── docs/                 # architecture, configuration, api
└── scripts/              # benchmark.py (perf)
```

### Adding a new modality (audio, video)

Three-line change, no central dispatch to edit:

1. Drop `parsers/audio_parser.py` whose class satisfies `protocols.Parser`.
2. `register_parser(AudioParser())` in `parsers/__init__.py`.
3. Drop `embedders/audio_embedder.py` whose class satisfies `protocols.Embedder`, and `register_embedder(...)` it.

The FastAPI app, CLI, and Qdrant backend all read from the registries at runtime.

## Documentation

- [Quickstart(从零到第一次搜索)](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Data flow(文本 vs 图片两条线)](docs/data-flow.md)
- [Configuration](docs/configuration.md)
- [HTTP API](docs/api.md)
- [Upload flow](docs/upload-flow.md)
- [FAQ & 故障排查](docs/faq.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md).

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
