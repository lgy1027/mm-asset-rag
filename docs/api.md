# HTTP API

Start the server:

```bash
mmrag-api
# → http://127.0.0.1:8011
# Interactive docs at /docs (Swagger UI)
```

All endpoints return JSON. Standard `application/json` POSTs.

## `GET /health`

```json
{
  "status": "ok",
  "assets": 30,
  "documents_jsonl_exists": true,
  "text_index_exists": false,
  "vector_backend": "qdrant"
}
```

## `POST /ingest`

Parse + reindex from the current manifest. Idempotent: re-runs overwrite the indexes.

```json
// request
{
  "limit": 0,                  // 0 = all assets; >0 truncates
  "pdf_parser": "auto",       // auto | pymupdf | paddleocr_vl
  "ocr": false,                // run local OCR on image assets
  "vlm": false                 // run VLM caption on image assets
}

// response
{
  "status": "ok",
  "documents_jsonl": "/Users/.../mm_asset_rag_home/documents.jsonl",
  "backend": "qdrant"
}
```

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
  "hits": [
    {
      "route": "qdrant_text",
      "score": 0.91,
      "asset_id": "pdf_rag",
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
| `text` | `qdrant_text_search` — single-collection vector search on text embeddings |
| `text-to-image` | `qdrant_text_to_image_search` — embeds the query with the CLIP text encoder, queries the image collection |
| `image-to-image` | `qdrant_image_to_image_search` — embeds `image_path` with the CLIP image encoder, queries the image collection |
| `hybrid` | weighted merge of text + text-to-image (and image-to-image if `image_path` provided) |

## `POST /answer`

```json
// request
{ "question": "which document covers retrieval-augmented generation?", "top_k": 5 }

// response
{
  "question": "...",
  "answer": "Based on the retrieved sources, ...",
  "sources": [
    {
      "asset_id": "pdf_rag",
      "title": "...",
      "score": 0.91,
      "routes": ["qdrant_text"],
      "page": 1,
      "parser": "pymupdf"
    }
  ]
}
```

If no LLM is configured (no `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`), this returns an evidence-summary fallback instead of an LLM-generated answer — useful for development and offline evaluation.

## `POST /eval`

Runs the built-in regression set (three fixed queries). Each case reports whether any of the top-`top_k` hits matched an expected `asset_id`.

```json
{ "results": [{ "query": "...", "expected_asset_ids": [...], "actual_asset_ids": [...], "hit": true }] }
```