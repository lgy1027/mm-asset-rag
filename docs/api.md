# HTTP API

Start the server:

```bash
mmrag-api
# → http://127.0.0.1:8011/         (bundled web UI)
# → http://127.0.0.1:8011/docs     (Swagger UI)
```

JSON endpoints return `application/json`. The streaming endpoints return `application/x-ndjson` (one JSON object per line).

## `GET /health`

Returns service liveness plus file and index state.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "files": 12,
  "documents_jsonl_exists": true,
  "text_index_exists": true,
  "image_index_exists": true,
  "vector_backend": "qdrant",
  "model": "gemma4:latest"
}
```

## `POST /upload/preview`

Multipart upload of one or more PDF / image files. This is **preview only**: no parse, embedding, or Qdrant call runs here.

The endpoint:

1. Enforces per-file and batch upload size limits (HTTP 413 when exceeded).
2. Streams each file into `$MM_ASSET_RAG_HOME/.preview-cache/incoming_<id>/`.
3. Copies files into a stable preview cache `$MM_ASSET_RAG_HOME/.preview-cache/<cache_id>/`.
4. Sniffs magic bytes and local metadata (PDF /Info, page count, image dimensions, EXIF).
5. Optionally calls a VLM in JSON mode for `title`, `description`, `tags`, and `dominant_objects`.
6. Returns editable preview cards.

**Form fields**

| Field | Type | Notes |
| --- | --- | --- |
| `files` | one or more `multipart/form-data` files | PDF / PNG / JPEG / GIF / BMP / WEBP |

**Response**

```json
{
  "cache_id": "c3f74c9df25a",
  "previews": [
    {
      "preview_id": "dc4c709c9925",
      "cache_id": "c3f74c9df25a",
      "sniff": {
        "asset_id": "paper",
        "title": "Paper",
        "source_type": "pdf",
        "relative_path": "paper.pdf",
        "file_size": 123456,
        "page_count": 8,
        "pdf_metadata": {"title": "Paper Title"},
        "error": null
      },
      "auto_meta": {
        "title": "Edited by VLM",
        "description": "Short searchable summary.",
        "tags": ["rag", "retrieval"],
        "dominant_objects": []
      },
      "effective_title": "Edited by VLM",
      "effective_tags": ["rag", "retrieval"],
      "effective_description": "Short searchable summary.",
      "rejected_reason": null
    }
  ],
  "rejected": []
}
```

Unsupported or over-limit files return a preview with `sniff.source_type="unknown"` or `rejected_reason`; the UI checks “skip this file” by default. Multipart bodies that exceed `UPLOAD_MAX_FILE_BYTES` or `UPLOAD_MAX_BATCH_BYTES` return HTTP 413 before preview cards are built.

## `POST /upload/confirm`

Applies user edits to preview cards, moves confirmed files into `assets/pdfs/`, `assets/images/`, or `assets/documents/`, and starts a background parse + index task. `IngestService` is the public task facade; it delegates workflow sequencing to `IngestWorkflow` and durable task snapshots to `TaskStore`.

```json
// request
{
  "cache_id": "c3f74c9df25a",
  "edits": [
    {
      "preview_id": "dc4c709c9925",
      "title": "User-corrected title",
      "tags": ["custom", "tag"],
      "description": "Optional corrected description",
      "document_id": "rag-paper",
      "collection": "team-knowledge",
      "allowed_principals": ["alice", "engineering"],
      "rejected": false
    }
  ]
}

// response
{
  "task_id": "6bfcc3e53100",
  "kind": "ingest",
  "uploaded": ["pdfs/User-corrected title_dc4c709c.pdf"]
}
```

Poll `/tasks/{task_id}` for progress. The endpoint returns immediately; parsing and indexing run in a background thread. Confirm validates that cached files are still inside the requested preview cache before moving them. If confirm fails before the task is created, no background work starts; if the later background task fails, confirmed files remain under `assets/` and the task error explains the parse/index failure.

## `GET /tasks/{task_id}`

Returns the latest snapshot of a background task.

```json
{
  "task_id": "6bfcc3e53100",
  "kind": "ingest",
  "status": "done",
  "started_at": 1782048987.4,
  "finished_at": 1782049015.0,
  "total": 2,
  "processed": 2,
  "skipped": 0,
  "failed": 0,
  "current": "index built · text=10 image=4",
  "error": null,
  "uploaded_files": ["pdfs/paper.pdf", "images/photo.jpg"],
  "version_statuses": {
    "rag-paper@1-a1b2c3d4e5f6": "indexed"
  },
  "elapsed_sec": 27.6,
  "progress": 1.0
}
```

The `current` field reflects the worker's last position: `parsing N/M`, `text indexed ...`, `index built ...`, or `parse crashed: ...` / `index crashed: ...` on failure.

`status="interrupted"` is set on startup for tasks that were still running when the previous process exited. See [Architecture](architecture.md#task-persistence).

## `GET /tasks`

Lists every task known to the service (in-memory + history loaded from `$MM_ASSET_RAG_HOME/tasks.db`).

## `GET /tasks/{task_id}/stream`

NDJSON stream of live task snapshots. Each line is a JSON object:

| Event | Fields | Fires |
| --- | --- | --- |
| `snapshot` | task record | on every patch (parse / embed / upsert) |
| `heartbeat` | — | every ~15 s of silence |
| `done` | — | when the task reaches a terminal status |
| `error` | `message: "..."` | when `task_id` is unknown |

Companion to `/tasks/{task_id}` polling — the web UI uses this to drive the live progress bar without re-fetching the whole record every second.

## `POST /tasks/{task_id}/cancel`

Cooperative cancellation: sets a per-task stop flag the worker checks between document versions; the task ends as `status="cancelled"` (terminal). A task already at a terminal state is returned unchanged. Cancellation is cooperative — a task mid-version finishes that version first.

```json
// response
{ "task_id": "6bfcc3e53100", "status": "cancelled", "finished_at": 1782049050.1 }
```

- `200` — cancellation requested (or task already terminal).
- `404` — `task_id` is unknown.

## `POST /tasks/{task_id}/retry`

Re-run a previously failed, partial, or interrupted task. The original task's `kind` and `parse_options` are preserved; the new task is recorded with `source="retry"` and `origin_task_id` pointing back to the original.

Query parameters:

- `force=true` — clear the targeted document-version caches and chunk rows before re-parsing.
- `failed_only=true` — only re-run document versions whose previous status was failed, skipped, or failed during indexing. Only meaningful for tasks with per-version outcome data (`version_statuses`).

```json
// response
{
  "task_id": "f0e1a2b3c4d5",
  "kind": "ingest",
  "origin_task_id": "6bfcc3e53100",
  "source": "retry",
  "force": false,
  "failed_only": false,
  "uploaded": ["pdfs/paper.pdf", "images/photo.jpg"]
}
```

Status codes:

- `200` — retry task created.
- `400` — original task is not in a retryable state, or no document versions are available.
- `404` — `task_id` is unknown.

## `GET /documents`

Returns the latest visible version of each document. Every request must supply
`collection` and `principal` query parameters; an optional JSON
`metadata_filter` further restricts the persisted access policy.

```json
{
  "documents": [
    {
      "document_id": "rag-paper",
      "title": "RAG Paper",
      "source": {"source_id": "upload:rag-paper"},
      "latest_version": {
        "document_id": "rag-paper",
        "version_id": "rag-paper@2-b1c2d3e4f5a6",
        "version_number": 2,
        "content_hash": "b1c2d3e4f5a6..."
      }
    }
  ]
}
```

## `GET /documents/{document_id}`

Returns the visible immutable version history for one document using the same
required access-context query parameters as `/documents`. Returns `404` when
the document is unknown or no version is visible to that context.

## `GET /parsed-image/{document_id}/{version_id}/{filename}`

Serves one image extracted for a visible document version. The server resolves
the internal physical cache key from the persisted version record; that key is
not part of the public URL.

- `200` — image bytes (`image/<ext>`).
- `404` — the document version or filename is unknown, inaccessible, or unsafe.

## `POST /search`

```json
// request
{
  "query": "retrieval augmented generation",
  "mode": "hybrid",
  "image_path": null,
  "top_k": 5,
  "collection": "team-knowledge",
  "principal": "alice"
}

// response
{
  "query": "...",
  "mode": "hybrid",
  "hits": [
    {
      "score": 0.91,
      "document_id": "rag-paper",
      "version_id": "rag-paper@2-b1c2d3e4f5a6",
      "chunk_id": "rag-paper@2-b1c2d3e4f5a6:3",
      "title": "Paper Title",
      "source_type": "pdf",
      "source_path": "pdfs/paper.pdf",
      "evidence": "...",
      "routes": ["qdrant_text"],
      "page": 4,
      "parser": "pymupdf",
      "images": []
    }
  ]
}
```

The four modes are selected by `SearchService` and dispatched through the
active `SearchBackend` adapter:

| Mode | Retrieval route |
| --- | --- |
| `text` | Dense + BM25 RRF on the text collection (with query rewrite when configured) |
| `text-to-image` | Embeds the query with the CLIP text encoder and queries the image collection |
| `image-to-image` | Embeds `image_path` with the CLIP image encoder |
| `hybrid` | weighted merge of text + text-to-image (and image-to-image if `image_path` provided) |

`image-to-image` without `image_path` returns HTTP 400.

The API route is an HTTP adapter only: it converts request fields to a
`SearchCommand`; retrieval itself is executed by `SearchService`, not by
Qdrant-specific helpers.

## `POST /answer`

Synchronous answer: retrieval + grounded LLM completion in one call.

```json
// request
{ "question": "which document covers retrieval-augmented generation?", "top_k": 5 }

// response
{
  "question": "...",
  "answer": "Based on the retrieved sources, ...",
  "sources": []
}
```

If no LLM is configured (missing `LLM_MODEL` or its resolved connection), the response contains an evidence-summary `answer` instead of failing.

## `POST /chat`

Same as `/answer` but takes a `ChatRequest` (`question` + routing fields `mode` / `image_path` / `top_k`) and runs the full retrieve + grounded-LLM flow in one non-streaming call. Useful when you don't want NDJSON streaming. Returns the same shape as `/answer` (`question`, `answer`, `sources`).

```json
// request
{ "question": "which document covers RAG?", "mode": "hybrid", "top_k": 5 }
```

Like `/answer`, `/chat` spends LLM quota, so once `MMRAG_API_TOKEN` is set it is guarded the same way as the other write/quota endpoints (see [Configuration](configuration.md)).

## `POST /chat/stream`

NDJSON streaming of the same flow as `/answer`. Each line is a JSON object:

| Event | Fields | Fires |
| --- | --- | --- |
| `sources` | `sources: [...]` | once, up front |
| `token` | `text: "..."` | once per LLM token |
| `done` | — | exactly once at the end |
| `error` | `message: "..."` | on any exception |

Reasoning-model note: `<think>...</think>` blocks emitted by reasoning models are stripped across chunk boundaries.

## `POST /eval`

Runs the retrieval regression set. Each case reports whether an exact positively judged `document_id` appears in the top-`top_k` results.

| Field | Default | Notes |
| --- | --- | --- |
| `top_k` | `5` | 1–200 |
| `v2` | `false` | Run the v2 (multi-dimensional, Chinese-primary) set instead of v1 |
| `cases_path` | `null` | Optional path to a case JSON overriding the default (`EVAL_CASES_PATH` → the bundled `mm_asset_rag/eval_data/<version>_cases.json`). Same schema as `mmrag eval --cases`. |

```json
{
  "results": [{
    "query_id": "q1",
    "query": "...",
    "qrels": {"document-id": 3},
    "actual_document_ids": ["document-id"],
    "hit": true,
    "rank": 1,
    "group": "en"
  }]
}
```

Cases live in JSON files with grouped `{query_id, query}` objects and top-level `qrels: {query_id: {document_id: relevance}}`; the bundled default is a small text→text sample. Point `cases_path` / `EVAL_CASES_PATH` at your own qrels file to override it. Without documents ingested under the exact judged IDs, every positive case returns `hit: false`.

Cases with positive qrel grades are **positive** retrieval cases: Recall, MRR,
MAP, and graded NDCG measure whether judged documents were retrieved. Cases
with an explicit empty qrels mapping are **negative** rejection cases: they
report empty-result and false-retrieval rates separately and do not lower
positive retrieval metrics. These measurements describe the chosen corpus and
cases; they do not establish a universal retrieval-quality threshold.
