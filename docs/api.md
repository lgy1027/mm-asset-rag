# HTTP API

Start the server:

```bash
mmrag-api
# → http://127.0.0.1:8011/         (bundled web UI)
# → http://127.0.0.1:8011/docs     (Swagger UI)
```

JSON endpoints return `application/json`. The streaming endpoints return `application/x-ndjson` (one JSON object per line).

## `GET /health`

Returns service liveness + asset / index counts.

```json
{
  "status": "ok",
  "assets": 214,
  "documents_jsonl_exists": true,
  "text_index_exists": true,
  "image_index_exists": true,
  "vector_backend": "qdrant",
  "model": "gemma4:latest"
}
```

## `POST /upload`

Multipart upload of one or more PDF / image files. The endpoint:

1. Streams each file to `$MM_ASSET_RAG_HOME/assets/{pdfs,images}/`, deduplicating by name (a 6-hex hash suffix is appended on conflict).
2. Spawns a background thread that **only parses the just-uploaded files** (manifest bypass).
3. If `auto_index=true` (default), the same thread runs the **incremental** indexer (`build_qdrant_text_index` / `build_qdrant_image_index` — already-indexed documents are skipped).

Returns a `task_id` immediately; poll `/tasks/{id}` for progress. The endpoint does not block on parsing.

**Form fields**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `files` | one or more `multipart/form-data` files | required | `.pdf` / `.jpg` / `.jpeg` / `.png` / `.webp` / `.bmp` |
| `auto_index` | bool string | `true` (or `AUTO_INDEX` env) | run indexer after parse |
| `pdf_parser` | `auto` / `pymupdf` / `paddleocr_vl` | `auto` (or `PDF_PARSER` env) | per-upload override |
| `enable_ocr` | bool string | `false` (or `ENABLE_OCR` env) | local OCR on images |
| `enable_vlm` | bool string | `false` (or `ENABLE_VLM` env) | VLM caption on images |
| `image_provider` | `lite` / `sentence_transformers` | `lite` (or `IMAGE_PROVIDER` env) | image embedding provider |

**Response**

```json
{
  "task_id": "6bfcc3e53100",
  "kind": "ingest",
  "uploaded": ["pdfs/alexnet.pdf", "images/happyfish.jpg"],
  "options": {
    "auto_index": true,
    "pdf_parser": "pymupdf",
    "enable_ocr": false,
    "enable_vlm": false,
    "image_provider": "lite"
  },
  "rejected": []
}
```

## `GET /tasks/{task_id}`

Returns the latest snapshot of a background task.

```json
{
  "task_id": "6bfcc3e53100",
  "kind": "ingest",
  "status": "done",            // pending | running | done | partial | failed | interrupted
  "started_at": 1782048987.4,
  "finished_at": 1782049015.0,
  "total": 30,
  "processed": 30,
  "skipped": 0,
  "failed": 0,
  "current": "index built · text=10 image=4",
  "error": null,
  "uploaded_files": ["pdfs/alexnet.pdf"],
  "elapsed_sec": 27.6,
  "progress": 1.0
}
```

The `current` field reflects the worker's last position: `parsing N/M`, `indexing: skip cached (X/Y)`, `text indexed · qdrant:...:inserted=N:skipped=M`, `index built · text=N image=M`, or `parse crashed: ...` / `index crashed: ...` on failure.

`status="interrupted"` is set on startup for tasks that were still running when the previous process exited. See [Architecture](architecture.md#task-persistence).

## `GET /tasks`

Lists every task known to the service (in-memory + history loaded from `$MM_ASSET_RAG_HOME/tasks.jsonl`).

## `POST /search`

```json
// request
{
  "query": "retrieval augmented generation",
  "mode": "hybrid",            // text | text-to-image | image-to-image | hybrid
  "image_path": null,          // required for image-to-image
  "top_k": 5
}

// response
{
  "query": "...",
  "mode": "hybrid",
  "hits": [
    {
      "route": "qdrant_text",
      "score": 0.91,
      "asset_id": "retrieval_augmented_generation",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "source_type": "pdf",
      "source_path": "pdfs/retrieval-augmented-generation.pdf",
      "evidence": "We provide a general-purpose fine-tuning recipe for RAG...",
      "metadata": { ... }
    }
  ]
}
```

The four modes map to:

| Mode | Backend call |
| --- | --- |
| `text` | `qdrant_text_search` — dense + BM25 RRF on the text collection |
| `text-to-image` | `qdrant_text_to_image_search` — embeds the query with the CLIP text encoder, queries the image collection |
| `image-to-image` | `qdrant_image_to_image_search` — embeds `image_path` with the CLIP image encoder |
| `hybrid` | weighted merge of text + text-to-image (and image-to-image if `image_path` provided) |

`image-to-image` without `image_path` returns HTTP 400.

## `POST /answer`

Synchronous answer: retrieval + grounded LLM completion in one call.

```json
// request
{ "question": "which document covers retrieval-augmented generation?", "top_k": 5 }

// response
{
  "question": "...",
  "answer": "Based on the retrieved sources, ...",
  "sources": [
    {
      "asset_id": "retrieval_augmented_generation",
      "title": "...",
      "score": 0.91,
      "page": 1,
      "parser": "pymupdf"
    }
  ]
}
```

If no LLM is configured (missing `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`), the response contains an evidence-summary `answer` instead of failing — useful for offline eval.

## `POST /chat/stream`

NDJSON streaming of the same flow as `/answer`. Each line is a JSON object:

| Event | Fields | Fires |
| --- | --- | --- |
| `sources` | `sources: [...]` | once, up front |
| `token` | `text: "..."` | once per LLM token |
| `done` | — | exactly once at the end |
| `error` | `message: "..."` | on any exception (streamed once, then closes) |

**Reasoning-model note**: `<think>...</think>` blocks emitted by reasoning models (DeepSeek-R1, Qwen3-Thinking, etc.) are stripped across chunk boundaries. The token stream never exposes reasoning tokens to the client.

## `POST /eval`

Runs the built-in regression set. Each case reports whether any of the top-`top_k` hits matched an expected `asset_id`.

```json
{ "results": [{ "query": "...", "expected_asset_ids": [...], "actual_asset_ids": [...], "hit": true }] }
```