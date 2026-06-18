# Architecture

```
                ┌─────────────────────────────────────────────┐
                │                  API / CLI                  │
                │  FastAPI (mmrag-api)  /  argparse (mmrag)   │
                └─────────────────┬───────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│   Parsers     │         │  Retrieval    │         │   Answer      │
│               │         │               │         │               │
│ pdf_parser    │         │ qdrant_text_  │         │ hybrid_search │
│ image_parser  │         │   search      │         │   ↓           │
│   ↓           │         │ qdrant_text_  │         │ llm_answer    │
│ ParsedDocument│         │   to_image_   │         │   OR          │
│   ↓           │         │   search      │         │ fallback_     │
│ document_     │         │ qdrant_image_ │         │   answer      │
│   store       │         │   to_image_   │         │               │
│   (jsonl)     │         │   search      │         │               │
└──────┬────────┘         │   ↓           │         └───────────────┘
       │                  │ hybrid_search │
       ▼                  │   (weighted)  │
┌───────────────┐         └──────┬────────┘
│  Providers    │                │
│               │                ▼
│ Embedding     │         ┌───────────────┐
│   Provider    │         │   Qdrant      │
│   (text)      │         │   (Local or   │
│ ImageEmbed-   │         │    Server)    │
│   dingProvider│         │               │
│   (CLIP)      │         │  text coll.   │
│ OCR / VLM     │         │  image coll.  │
│   (image_parser)         └───────────────┘
│ LLM (answer)  │
└───────────────┘
```

## What the layers do

- **Parsers** turn raw files (PDF, image) into a stream of `ParsedDocument` records. PDF defaults to PyMuPDF (local) and can be swapped for PaddleOCR-VL (API). Image parsing runs OCR (local HTTP) and/or VLM caption (OpenAI-compatible chat completion).
- **Document store** persists parsed records as JSONL — the single source of truth between `parse` and `index` steps.
- **Providers** wrap all external HTTP integrations. Each provider reads its own env vars and falls back to a mock/dummy implementation when configuration is missing, so the pipeline can be exercised end-to-end with zero external services.
- **Qdrant** is the production vector backend (local file or remote server). We use `qdrant-client` directly rather than `llama-index-vector-stores-qdrant` because hybrid retrieval crosses multiple collections and image vectors are not first-class in LlamaIndex's `VectorStore` abstraction.
- **Retrieval** has three single-mode functions (`qdrant_text_search`, `qdrant_text_to_image_search`, `qdrant_image_to_image_search`) plus a `hybrid_search` that normalizes per-route scores and weights them: text 0.55, text-to-image 0.30, image-to-image 0.15 (if a query image is provided).
- **Answer** either calls an OpenAI-compatible chat completion (with `<think>...</think>` blocks stripped) or returns an evidence-summary fallback when no LLM is configured.

## Why no Protocol / abstract base class

The original `multimodal_asset_rag` was a LlamaIndex tutorial example. v0.1 keeps that structure: one concrete module per concern, one Qdrant backend, no factory pattern. A `VectorBackend` Protocol can be added later if a second backend (e.g. Milvus, Weaviate) becomes a real need.

## Why a flat layout

There are 14 modules in a single package. Splitting into `parsers/`, `backends/`, `providers/` subpackages would add navigation cost without buying much at this size. The package is small enough to read end-to-end; reorganize when it stops being small.