# HTTP API

Start the server:

```bash
mmrag-api
# → http://127.0.0.1:8011/         (bundled web UI)
# → http://127.0.0.1:8011/docs     (Swagger UI)
```

JSON endpoints return `application/json`. The streaming endpoints return `application/x-ndjson` (one JSON object per line).

## `GET /health`

Returns service liveness + asset / index state.

```json
{
  "status": "ok",
  "assets": 12,
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

Applies user edits to preview cards, moves confirmed files into `assets/pdfs/` or `assets/images/`, and starts a background parse + index task.

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
  "elapsed_sec": 27.6,
  "progress": 1.0
}
```

The `current` field reflects the worker's last position: `parsing N/M`, `text indexed ...`, `index built ...`, or `parse crashed: ...` / `index crashed: ...` on failure.

`status="interrupted"` is set on startup for tasks that were still running when the previous process exited. See [Architecture](architecture.md#task-persistence).

## `GET /tasks`

Lists every task known to the service (in-memory + history loaded from `$MM_ASSET_RAG_HOME/tasks.jsonl`).

## `POST /tasks/{task_id}/retry`

Re-run a previously failed, partial, or interrupted task. The original task's `kind` and `parse_options` are preserved; the new task is recorded with `source="retry"` and `origin_task_id` pointing back to the original.

Query parameters:

- `force=true` — clear `parsed/<asset_id>/raw.jsonl` cache before re-running so every asset is re-parsed.
- `failed_only=true` — only re-run assets whose previous status was failed or skipped. Only meaningful for tasks that have per-asset outcome data (`asset_statuses`).

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
- `400` — original task is not in a retryable state, or no assets are available.
- `404` — `task_id` is unknown.
- `400` — `force` and `failed_only` are mutually exclusive.

## `GET /assets`

Return every non-deleted asset recorded in the content-hash index.

```json
{
  "assets": [
    {
      "asset_id": "Beach_d7e16fe3",
      "relative_path": "images/Beach_d7e16fe3.png",
      "source_type": "image",
      "asset_title": "Beach",
      "ingested_at": 1782048987.4
    }
  ]
}
```

## `DELETE /assets/{asset_id}`

Best-effort cleanup of every trace of `asset_id`. Removes the source file, `parsed/<id>/`, `captions/<id>.json`, the matching `documents.jsonl` rows, the Qdrant text + image points, and tombstone the asset index entry.

```json
// response
{
  "asset_id": "Beach_d7e16fe3",
  "file_deleted": true,
  "parsed_deleted": true,
  "captions_deleted": true,
  "documents_removed": 1,
  "text_collections_scanned": 1,
  "image_collections_scanned": 1,
  "errors": [],
  "was_known": true
}
```

Status codes:

- `200` — report returned (even when nothing remained to delete).
- `404` — `asset_id` is unknown to the asset index.

## `POST /search`

```json
// request
{
  "query": "retrieval augmented generation",
  "mode": "hybrid",
  "image_path": null,
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
      "asset_id": "paper",
      "title": "Paper Title",
      "source_type": "pdf",
      "source_path": "pdfs/paper.pdf",
      "evidence": "...",
      "metadata": {}
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
  "sources": []
}
```

If no LLM is configured (missing `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`), the response contains an evidence-summary `answer` instead of failing.

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

Runs the built-in regression set. Each case reports whether any of the top-`top_k` hits matched an expected `asset_id`.

```json
{ "results": [{ "query": "...", "expected_asset_ids": [...], "actual_asset_ids": [...], "hit": true }] }
```
