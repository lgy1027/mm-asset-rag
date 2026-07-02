# Architecture

## Layered view

```
                       ┌────────────────────────────────────────────┐
                       │       Thin entry points (route layer)      │
                       │  FastAPI app (mm_asset_rag/api.py)        │
                       │  CLI      (mm_asset_rag/cli.py)           │
                       └─────────────────┬──────────────────────────┘
                                         │  delegates
                                         ▼
                       ┌────────────────────────────────────────────┐
                       │     Service (mm_asset_rag/service.py)      │
                       │  IngestService:                            │
                       │   - parse_assets / ingest_assets            │
                       │   - reindex (force-recreate)                │
                       │   - load_history / list_tasks / get_task    │
                       └─────────────────┬──────────────────────────┘
                                         │
        ┌───────────────────┬───────────┼───────────┬───────────────────┐
        ▼                   ▼           ▼           ▼                   ▼
 ┌─────────────┐    ┌─────────────┐  ┌─────────┐ ┌─────────────┐  ┌─────────────┐
 │  Parsers    │    │  Embedders   │  │ Backends│ │  Retrieval   │  │   Answer     │
 │ (registry)  │    │  (registry)  │  │(reg.)   │ │  (pure)     │  │  (LLM)      │
 │             │    │              │  │         │ │              │  │              │
 │ pdf_parser  │    │ text_embed   │  │ Qdrant  │ │ normalize_   │  │ hybrid_      │
 │ image_      │    │ image_embed  │  │ (local/ │ │   scores     │  │   search     │
 │   parser    │    │              │  │ server) │ │ merge_hits   │  │ ↓            │
 │ audio_      │    │              │  │         │ │ hybrid_search│  │ llm_answer   │
 │   parser    │    │              │  │         │ │              │  │   OR         │
 │   ...       │    │              │  │         │ │              │  │ evidence-    │
 │             │    │              │  │         │ │              │  │   summary    │
 └──────┬──────┘    └──────┬───────┘  └────┬────┘ └──────┬───────┘  └─────────────┘
        │                  │              │           │
        ▼                  ▼              ▼           ▼
  ParsedDocument ────► dense + sparse ─► Qdrant ──► SearchHit ──► answer
  (jsonl store)         vectors        collections     (merged)      (text)
```

## What each layer does

- **`api.py` / `cli.py`** are thin entry points. Both use the upload-first
  pipeline: files are sniffed, previewed, confirmed, then passed to the same
  `IngestService` for parse / index / task-history work.
- **`upload_pipeline.UploadPipeline`** owns the two-stage upload flow:
  `/upload/preview` copies files into `.preview-cache`, calls `sniff.py` and
  optional VLM metadata extraction, then `/upload/confirm` moves confirmed
  files into `assets/pdfs` or `assets/images` and constructs `Asset` objects.
- **`service.IngestService`** owns parse + index + task state. It uses
  `Protocol`s from `protocols.py` to dispatch to the right parser /
  embedder / backend via `registry.py`, and persists every state change to
  `$MM_ASSET_RAG_HOME/tasks.jsonl` so history survives restarts.
- **`parsers/`** turn raw files into `ParsedDocument` records. Today:
  PyMuPDF + PaddleOCR-VL (PDF); OCR + VLM caption (image). Each parser
  satisfies `Parser` Protocol and is registered at import time.
- **`embedders/`** generate dense / sparse vectors. Today: OpenAI-
  compatible text embedder + CLIP image embedder. Each satisfies the
  `Embedder` Protocol.
- **`backends/`** store vectors and run similarity search. Today: Qdrant
  (local file or remote server). The `VectorBackend` Protocol is the
  swap-in point for Milvus / Pinecone.
- **`retrieval.hybrid_search`** is pure (no I/O); it normalizes per-route
  scores and weights them from `Settings` (defaults: text 0.80,
  text-to-image 0.20, image-to-image 0.0 unless an image query is provided).
- **`answer.llm_answer` / `stream_answer_chunks`** issue an OpenAI-
  compatible chat completion with the retrieved evidence as context. When
  no LLM is configured, an evidence-summary fallback is returned instead.

## Protocol + registry

Three Protocols are declared in `protocols.py` and registered in `registry.py`:

| Protocol          | Keyed by            | Where the registry is queried                            |
| ----------------- | ------------------- | ------------------------------------------------------- |
| `Parser`          | `(source_type, name)` | `parsers/__init__.py` registers `pymupdf` / `paddleocr_vl` |
| `Embedder`        | `(modality, name)`  | `embedders/__init__.py` registers the default text embedder |
| `VectorBackend`   | `name`              | `backends/__init__.py` registers Qdrant                  |

Adding a new modality (audio, video) is a three-line change — see
[CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-modality-audio-video).

## Task persistence

Background work runs on daemon `threading.Thread`s spawned by
`IngestService._spawn()`. Every `_patch()` writes a JSON snapshot of
the task to `$MM_ASSET_RAG_HOME/tasks.jsonl`. On startup, the FastAPI
`lifespan` calls `service.load_history()`, which rebuilds the in-memory
task list and reclassifies any task that was still `running` when the
previous process exited as `interrupted`.

## Configuration

Every environment variable the codebase reads is declared in
`settings.Settings` (pydantic-settings). The module-level
`get_settings()` returns an `lru_cache`-wrapped singleton. New code
should call `get_settings().foo` rather than `os.environ.get("FOO")`.

## Why a flat package + sub-packages

`mm_asset_rag/` itself is flat (top-level modules), but three
sub-packages hold families of implementations:

- `parsers/` — implementations of the `Parser` Protocol.
- `embedders/` — implementations of the `Embedder` Protocol.
- `backends/` — implementations of the `VectorBackend` Protocol.

A new parser / embedder / backend drops into the matching sub-package and
registers itself. No central dispatch table needs editing.

## Why not LlamaIndex

Earlier versions used `llama-index-vector-stores-qdrant`. The codebase
dropped that integration because:

- It only handles text nodes (`BaseNode`/`TextNode`); image vectors are
  not first-class.
- Hybrid retrieval here crosses multiple collections, and image vectors
  are not first-class in LlamaIndex's `VectorStore` abstraction.

We talk to `qdrant-client` directly and own the sparse + dense hybrid
logic in `retrieval.hybrid_search`.