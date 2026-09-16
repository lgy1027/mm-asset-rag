# Architecture

## Layered view

```
FastAPI / CLI
  ├─ upload ─► UploadPipeline ─► assets/pdfs | assets/images | assets/documents
  │                              └─► IngestService / IngestWorkflow
  │                                   └─► parsers ─► embedders ─► IndexBackend
  ├─ search ─► SearchService.execute(SearchCommand)
  │              └─► query rewrite / retrieval policy ─► SearchBackend ─► SearchHit
  └─ answer/chat ─► SearchService ─► SearchHit
                     └─► llm_answer / stream_answer_chunks / evidence fallback

Registries select the parser, embedder, SearchBackend, and IndexBackend adapters;
Qdrant is the built-in search/index adapter.
```

> This is the **component** view. For the end-to-end **data flow** view
> (how a file moves from upload → parse → index → retrieval → answer,
> and how the text line vs the image line stay separate), see
> [`data-flow.md`](data-flow.md).

## What each layer does

- **`api.py` / `cli.py`** are thin entry points. Retrieval callers create a
  `SearchCommand` and use `SearchService.execute`; uploads use the
  upload-first pipeline, then pass confirmed assets to `IngestService` for
  parse / index / task-history work.
- **`search_service.SearchService`** is the transport-neutral retrieval
  entry point for the API, CLI, answer generation, and evaluation. It owns
  mode validation and routing, and depends on the `SearchBackend` port rather
  than Qdrant search helpers.
- **`upload_pipeline.UploadPipeline`** owns the two-stage upload flow:
  `/upload/preview` copies files into `.preview-cache`, calls `sniff.py` and
  optional VLM metadata extraction, then `/upload/confirm` moves confirmed
  files into `assets/pdfs`, `assets/images`, or `assets/documents` and constructs
  `Asset` objects.
- **`service.IngestService`** is the public facade for parse + index + task
  lifecycle work. `IngestWorkflow` owns parse/enrich/document/index
  sequencing, while `TaskStore` owns SQLite persistence; the facade keeps
  threads, cancellation, retry, and streaming behind the API boundary.
- **`parsers/`** turn raw files into parsed chunks. Today:
  PyMuPDF, PaddleOCR-VL, docling, and PP-OCR (PDF); MarkItDown/docling and
  a row-aware table parser (documents); OCR + VLM caption (image). Each parser
  satisfies `Parser` Protocol and is registered at import time.
- **`embedders/`** generate dense / sparse vectors. Today: OpenAI-
  compatible text embedder + CLIP image embedder. Each satisfies the
  `Embedder` Protocol.
- **`backends/`** store vectors and run similarity search. Today:
  `QdrantBackend` is the Qdrant adapter (local file or remote server),
  registered behind the `SearchBackend` and `IndexBackend` ports. Those ports
  are the swap-in point for Milvus / Pinecone.
- **`retrieval.hybrid_search`** orchestrates backend route searches, rank-based
  fusion, configured weights, and optional reranking. Its `merge_hits` helper is
  pure, but `hybrid_search` itself invokes backend and reranker paths.
- **`answer.answer_question`** obtains evidence through `SearchService` when hits
  are not supplied. **`llm_answer` / `stream_answer_chunks`** generate from those
  hits; when no LLM is configured, an evidence-summary fallback is returned.

## Protocol + registry

The runtime registry selects the adapters that satisfy the declared
`protocols.py` capabilities:

| Protocol          | Keyed by            | Where the registry is queried                            |
| ----------------- | ------------------- | ------------------------------------------------------- |
| `Parser`          | `(source_type, name)` | `parsers/__init__.py` registers `auto` / `pymupdf` / `paddleocr_vl` / `docling` / `ppocr` (PDF), `docling` / `markitdown` (documents, including CSV/TSV/XLSX table routing), and `image` |
| `Embedder`        | `(modality, name)`  | `embedders/__init__.py` registers the default text embedder |
| `SearchBackend` / `IndexBackend` | `name` | `backends/__init__.py` registers the Qdrant adapter |
| `VectorBackend`   | `name`              | Legacy aggregate port retained for compatible adapters   |

Adding a modality requires parser and embedder registration plus explicit
API/CLI routing and active-backend support. See
[CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-modality-audio-video).

## Task persistence

Background work runs on daemon `threading.Thread`s spawned by
`IngestService._spawn()`. `TaskStore` persists every task snapshot to
`$MM_ASSET_RAG_HOME/tasks.db`, while `IngestService` exposes the task
lifecycle to HTTP and CLI callers. On startup, the FastAPI `lifespan` calls
`service.load_history()`, which rebuilds the in-memory task list and
reclassifies any task that was still `running` when the previous process
exited as `interrupted`.

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

`QdrantBackend` talks to `qdrant-client` directly; application services use
its ports, while retrieval policy coordinates Qdrant's route results in
`retrieval.hybrid_search`.
