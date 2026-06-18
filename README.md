# mm-asset-rag

> Multimodal asset RAG — parse PDFs and images, index with Qdrant or LlamaIndex, retrieve by text/image/hybrid, answer with grounded LLM citations.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](.github/workflows/test.yml)

## What is this?

A small, self-contained Python package for multimodal Retrieval-Augmented Generation over a collection of PDFs and images. It supports:

- **Parsing**: PyMuPDF (local, free) or PaddleOCR-VL (API, better for scanned PDFs) for PDFs; OCR + VLM captioning for images.
- **Indexing**: Qdrant (local file or server) or LlamaIndex's built-in vector store.
- **Retrieval**: text-to-text, text-to-image, image-to-image, and weighted hybrid.
- **Generation**: OpenAI-compatible chat completion with strict evidence grounding; graceful fallback when no LLM is configured.

It is designed to **run end-to-end with zero external services** — every external dependency has a graceful fallback so the pipeline can be demonstrated offline. Production deployments swap in real services through environment variables.

## Why this project?

If you have a folder of PDFs and images and want to ask questions like *"which document covers retrieval-augmented generation?"* or *"find images similar to this one"*, this is a working starting point. It is not a research-grade system; it is a **reference implementation** that exposes the moving parts so you can replace any layer (parser, embedder, backend, reranker, LLM) without rewriting the rest.

Compared to larger frameworks:

- **vs LlamaIndex Studio / Verba**: this is a CLI + library, no web UI; you wire your own UI.
- **vs Haystack / txtai**: smaller surface area, multimodal-first, easier to read end-to-end.

## Installation

```bash
pip install mm-asset-rag
```

Optional CLIP-based image embeddings:

```bash
pip install "mm-asset-rag[clip]"
```

## Quick start

Prepare an asset manifest describing what you want to index:

```json
{
  "name": "my-assets",
  "records": [
    {
      "id": "doc1",
      "title": "Attention is All You Need",
      "type": "pdf",
      "path": "pdfs/attention-is-all-you-need.pdf",
      "tags": ["transformer", "nlp"]
    },
    {
      "id": "fig1",
      "title": "Architecture diagram",
      "type": "image",
      "path": "images/architecture.png",
      "tags": ["diagram"]
    }
  ]
}
```

Then run the pipeline:

```bash
# Set where to store parsed artifacts, indices, etc.
export MM_ASSET_RAG_HOME="$HOME/.mm_asset_rag"

# 1. Parse all assets into a unified document store
mmrag parse

# 2. Build text + image indexes (Qdrant Local by default)
mmrag index

# 3. Search
mmrag search "which document covers retrieval-augmented generation?"

# 4. Answer with citations
mmrag answer "which document covers retrieval-augmented generation?"

# 5. Run a small regression suite
mmrag eval
```

Or expose as an HTTP service:

```bash
mmrag-api
# → http://127.0.0.1:8011/docs
```

## Configuration

All settings come from environment variables (a `.env` file in the current directory is loaded automatically). The most important ones:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | Where to put parsed data, indexes, etc. | `~/.mm_asset_rag` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | LLM for answer generation | — |
| `EMBEDDING_*` | Text embedding provider (defaults to OpenAI-compatible) | — |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server mode (omit to use local file mode) | — |
| `CLIP_MODEL` | Sentence-transformers model name | `clip-ViT-B-32` |
| `PADDLEOCR_VL_API_TOKEN` | PaddleOCR-VL API token | — |

See [`docs/configuration.md`](docs/configuration.md) for the full list.

## Backends

Two vector backends are supported:

- **`qdrant`** (default) — production-grade, supports local file persistence or remote server.
- **`llamaindex`** — single-process, uses LlamaIndex's built-in `VectorStoreIndex`. Easier for prototyping but more limited for image retrieval.

Choose via the `--backend` flag on `index`, `search`, and `answer` commands. See [`docs/backends.md`](docs/backends.md).

## Project layout

```
src/mm_asset_rag/
├── parsers/        Parser Protocol + concrete PDF / image parsers
├── providers/      embedding, image_embedding, llm, ocr
├── backends/       VectorBackend Protocol + qdrant / llamaindex implementations
└── retrieval/      hybrid merge + normalization helpers
```

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Backends](docs/backends.md)
- [Parsers](docs/parsers.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

If you use this in research, please cite the underlying projects it depends on (`llama-index`, `qdrant-client`, `PyMuPDF`, etc.) per their respective licenses.