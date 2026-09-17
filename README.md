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
                  │          Vector backend (Qdrant built-in)        │
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
- **Indexing**: Qdrant is the built-in backend (local file or server). Select another registered backend with `VECTOR_BACKEND`; Qdrant text points carry dense + BM25 + BM25-zh vectors, and image points carry CLIP vectors.
- **Optional generation**: OpenAI-compatible chat completion with strict evidence grounding and NDJSON streaming. When no LLM is configured, `/answer` and `/chat` return an evidence summary instead of failing — retrieval still works.
- **Web UI**: a bundled single-page HTML (`mm_asset_rag/api/web/index.html`) served by FastAPI for upload preview, task status, and chat.

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

Optional Chinese CLIP and local OCR support:

```bash
pip install "mm-asset-rag[cn_clip]"  # Chinese CLIP
pip install "mm-asset-rag[ocr]"      # PP-OCRv6 via ONNX Runtime
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

> **First time here?** See [docs/quickstart.md](docs/quickstart.md) for the full setup path, from local Ollama and Qdrant to a first `mmrag search`. This quick start assumes the environment is ready.

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
  └─ index text chunks and image vectors through the active backend (Qdrant by default)
```

## Configuration

All settings come from environment variables (a `.env` file in the current directory is loaded automatically). Start with the capability choices and RAG profiles below; detailed tuning stays in the advanced reference.

| Variable | Purpose | Default |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | Where to put uploaded assets, parsed data, indexes, task log. | `~/.mm_asset_rag` |
| `MODEL_API_KEY` / `MODEL_BASE_URL` | Shared OpenAI-compatible connection for LLM, VLM, and embedding. | — |
| `EMBEDDING_MODEL` / `EMBEDDING_*` | Required text embedding model and optional provider override. | — |
| `LLM_MODEL` | Optional LLM for `/answer`, `/chat`, and query rewrite. | — |
| `VLM_MODEL` / `VLM_*` | Optional VLM for upload metadata and image captions. | — |
| `RERANKER_*` | Optional second-stage reranker provider and model. | disabled |
| `INGESTION_PROFILE` | `fast`, `balanced`, or `precision` ingestion cost/quality defaults. | `balanced` |
| `RETRIEVAL_PROFILE` | `fast`, `balanced`, or `precision` retrieval defaults. | `balanced` |
| `VECTOR_BACKEND` | Registered search/index backend. | `qdrant` |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server mode (omit to use local file mode). | — |
| `CLIP_MODEL` | Sentence-transformers CLIP model name (with `[clip]` extra). | `clip-ViT-B-32` |
| `IMAGE_PROVIDER` | `clip` or `cn_clip`. | `clip` |
| `OCR_BACKEND` | Image OCR backend: `local` (PP-OCRv6 via `[ocr]` extra) or `http`. | `local` |

Profiles fill advanced defaults only when that variable is absent, so existing explicit `.env` values keep their behavior. See [`.env.example`](.env.example) for the compact template and [`docs/configuration.md`](docs/configuration.md) for advanced tuning.

## Evaluation

`mmrag eval` scores grouped query cases against exact logical document IDs and reports document-level Recall, MRR, MAP, and graded NDCG. Each case has a `query_id` and `query`; one top-level `qrels` object maps every query ID to `{document_id: relevance}`. The **default** is a small qrels sample shipped in `mm_asset_rag/eval/eval_data/`. Matching documents must already be ingested under those exact, case-sensitive document IDs; otherwise the cases are reported as misses.

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
├── mm_asset_rag/         # single Python package, organised by responsibility
│   ├── cli.py            # `mmrag` / `mmrag-api` console scripts
│   ├── service.py        # IngestService facade: parse / index / task-history
│   ├── core/             # contracts + infra: settings, schema, protocols,
│   │                     #   registry, paths, llm_transport, observability
│   ├── ingest/           # upload → parse: upload_pipeline, sniff, auto_meta,
│   │                     #   document_store, ingest_workflow, task_store
│   ├── query/            # retrieval: search_service, retrieval, query_rewrite,
│   │                     #   query_intent, query_preprocess, evidence_policy
│   ├── answer/           # grounded answer generation + answer evaluation
│   ├── eval/             # eval harnesses + bundled eval_data/
│   ├── api/              # FastAPI thin route layer + bundled web UI
│   ├── parsers/          # PDF/image parser implementations
│   ├── embedders/        # text/image embedder implementations
│   └── backends/         # backend adapters (Qdrant built in)
├── tests/unit/           # offline unit tests
├── tests/integration/    # marked @pytest.mark.integration
├── docs/                 # architecture, configuration, api
└── scripts/              # benchmark.py (perf), run_full_eval.py,
                          #   docker-compose.langfuse.yml (self-hosted tracing)
```

### Adding a new modality (audio, video)

1. Implement and register a parser that satisfies `protocols.Parser`.
2. Implement and register an embedder that satisfies `protocols.Embedder`.
3. Add API/CLI routing for the new source type.
4. Extend the selected backend to index and query that modality.

The registry removes central implementation lookup; routing and backend capabilities remain explicit.

## Documentation

- [Quickstart(从零到第一次搜索)](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Data flow(文本 vs 图片两条线)](docs/data-flow.md)
- [Configuration](docs/configuration.md)
- [Tracing / observability](docs/configuration.md#tracing--observability)
- [音视频检索设计(Phase 1)](docs/design-audio-video.md)
- [HTTP API](docs/api.md)
- [Upload flow](docs/upload-flow.md)
- [FAQ & 故障排查](docs/faq.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md).

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
