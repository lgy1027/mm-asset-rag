# mm-asset-rag

> Multimodal asset RAG — parse PDFs and images, index with Qdrant, retrieve by text/image/hybrid, stream grounded LLM answers through a bundled single-page web UI.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](.github/workflows/test.yml)

## What is this?

A small, self-contained Python package for multimodal Retrieval-Augmented Generation over a collection of PDFs and images. It supports:

- **Parsing**: PyMuPDF (local, free) or PaddleOCR-VL (API, better for scanned PDFs) for PDFs; OCR + VLM captioning for images.
- **Indexing**: Qdrant (local file or server). Text points carry dense + BM25 sparse vectors and are fused with RRF at query time.
- **Retrieval**: text-to-text, text-to-image, image-to-image, and weighted hybrid.
- **Generation**: OpenAI-compatible chat completion with strict evidence grounding and NDJSON streaming.
- **Web UI**: a bundled single-page HTML (`mm_asset_rag/web/index.html`) served by FastAPI for upload, task status, and chat.

It is designed to **run end-to-end with zero external services** when an OpenAI-compatible endpoint (local ollama, vLLM, etc.) is available. When no LLM is configured the `/answer` and `/chat` endpoints return an evidence summary instead of failing.

## Why this project?

If you have a folder of PDFs and images and want to ask questions like *"which document covers retrieval-augmented generation?"* or *"find images similar to this one"*, this is a working starting point. It is not a research-grade system; it is a **reference implementation** that exposes the moving parts so you can replace any layer (parser, embedder, backend, reranker, LLM) without rewriting the rest.

Compared to larger frameworks:

- **vs LlamaIndex Studio / Verba**: this ships with a web UI for free, focuses on multimodal from day one, and stays under 2k lines per module.
- **vs Haystack / txtai**: smaller surface area, multimodal-first, easier to read end-to-end.

## Installation

```bash
pip install mm-asset-rag
```

Optional CLIP-based image embeddings:

```bash
pip install "mm-asset-rag[clip]"
```

For local development:

```bash
git clone https://github.com/lgy1027/mm-asset-rag
cd mm-asset-rag
pip install -e ".[dev,clip]"
```

## Quick start

The repo ships with a sample dataset of 30 PDFs + 184 photos under `examples/data/chapter11_assets/`. Point the data root at it and run the pipeline:

```bash
# Tell the package where to store parsed data, indexes, etc.
# A symlink is the simplest way to reuse the bundled samples.
ln -s "$(pwd)/examples/data/chapter11_assets" ~/.mm_asset_rag/assets
# (Equivalent: export MM_ASSET_RAG_HOME="$(pwd)/examples/data/chapter11_assets")

# 1. Parse all assets into a unified document store
mmrag parse --pdf-parser paddleocr_vl --vlm

# 2. Build the text + image index (incremental — skips already-indexed docs)
mmrag index

# 3. Search
mmrag search "which document covers retrieval-augmented generation?"

# 4. Answer with grounded citations
mmrag answer "which document covers retrieval-augmented generation?"

# 5. Run a small regression suite
mmrag eval
```

Or run the HTTP service (with the bundled web UI):

```bash
mmrag-api
# → http://127.0.0.1:8011/         (web UI)
# → http://127.0.0.1:8011/docs     (OpenAPI / Swagger)
```

## Configuration

All settings come from environment variables (a `.env` file in the current directory is loaded automatically). The most important ones:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | Where to put parsed data, indexes, task log. | `~/.mm_asset_rag` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | LLM for `/answer` and `/chat`. | — |
| `EMBEDDING_*` | Text embedding provider (defaults to OpenAI-compatible). | — |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server mode (omit to use local file mode). | — |
| `CLIP_MODEL` | Sentence-transformers CLIP model name (with `[clip]` extra). | `clip-ViT-B-32` |
| `PADDLEOCR_VL_API_TOKEN` | PaddleOCR-VL API token. | — |
| `PDF_PARSER` / `ENABLE_OCR` / `ENABLE_VLM` / `IMAGE_PROVIDER` | Parser / image embedding defaults, overridable per `/upload`. | `auto` / `false` / `false` / `lite` |

See [`.env.example`](.env.example) and [`docs/configuration.md`](docs/configuration.md) for the full list.

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
│   ├── qdrant_store.py   # Qdrant client, collection mgmt, hybrid upsert
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
├── tests/integration/    # marked @pytest.mark.integration (network / Qdrant)
├── docs/                 # architecture, configuration, api
└── scripts/              # eval_rag.py, build_manifest.py
```

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [HTTP API](docs/api.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

If you use this in research, please cite the underlying projects it depends on (`qdrant-client`, `PyMuPDF`, `fastembed`, etc.) per their respective licenses.