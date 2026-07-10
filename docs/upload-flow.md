# Upload flow

`mm-asset-rag` is upload-first: users do not write `asset_manifest.json` or pre-register metadata. The system creates asset metadata from the uploaded bytes, shows the user an editable preview, then ingests confirmed files.

## Two-phase flow

```text
POST /upload/preview (multipart files)
  ├─ enforce per-file and batch upload size limits
  ├─ stream files to $MM_ASSET_RAG_HOME/.preview-cache/incoming_<id>/
  ├─ copy each file into $MM_ASSET_RAG_HOME/.preview-cache/<cache_id>/
  ├─ write an internal cache manifest with relative cached_name values
  ├─ sniff magic bytes and local metadata
  │    ├─ PDF: %PDF-, page count, /Info title/author/subject
  │    ├─ image: PNG/JPEG/GIF/BMP/WEBP signatures, dimensions, EXIF
  │    └─ document: office ZIP (docx/pptx/xlsx) or HTML/Markdown/text by extension
  ├─ optional VLM auto-metadata
  │    ├─ title
  │    ├─ description
  │    ├─ tags
  │    └─ dominant_objects / page_summary
  └─ return editable preview cards

POST /upload/confirm (cache_id + edited previews)
  ├─ validate cache_id and cached files stay inside that preview cache
  ├─ apply user edits
  ├─ skip user-rejected / unsupported / over-limit files
  ├─ move confirmed files to assets/pdfs, assets/images, or assets/documents
  ├─ construct Asset objects directly (no manifest)
  ├─ parse into documents.jsonl
  └─ upsert into Qdrant text/image collections
```

## Why preview first?

VLM metadata is useful but not authoritative. The preview card lets users correct wrong titles, tags, or descriptions before the content is embedded and indexed. This avoids re-indexing just to fix a hallucinated tag.

## Local sniffing

`mm_asset_rag/sniff.py` is pure local inspection:

- never calls a network service;
- trusts file magic bytes over filename extension;
- returns `source_type="unknown"` for unsupported files instead of crashing;
- extracts basic metadata even when VLM is disabled.

Supported types:

| Type | Detection | `source_type` |
| --- | --- | --- |
| PDF | `%PDF-` magic bytes + PyMuPDF metadata | `pdf` |
| PNG | PNG signature | `image` |
| JPEG | `0xff 0xd8 0xff` | `image` |
| GIF | `GIF87a` / `GIF89a` | `image` |
| BMP | `BM` | `image` |
| WEBP | `RIFF....WEBP` | `image` |
| DOCX / PPTX / XLSX | Office Open XML ZIP container (`PK\x03\x04`) by extension + `zipfile.is_zipfile` guard | `document` |
| HTML / Markdown / text | by extension (`.html` / `.htm` / `.md` / `.markdown` / `.txt`) | `document` |

`document` types are recognised by sniff, routed to `assets/documents/` at confirm (the original extension is preserved so docling picks the right backend), and parsed by the docling adapter (`pip install -e ".[docling]"`). Without the extra installed, the upload still confirms but parsing raises a friendly install hint. document files skip VLM auto-meta (no first-page render / image path) and fall back to the sniff-derived filename title.

## VLM auto-metadata

`mm_asset_rag/auto_meta.py` calls an OpenAI-compatible VLM endpoint using JSON mode. It reuses the VLM settings:

- `VLM_BASE_URL` / fallback `OPENAI_BASE_URL`
- `VLM_API_KEY` / fallback `OPENAI_API_KEY`
- `VLM_MODEL` / fallback `OPENAI_MODEL`

If any of those are missing, or the request fails, preview falls back to sniff-only metadata. Upload still works.

Control knobs:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `AUTO_META_ENABLED` | `true` | Enable/disable VLM title/tag extraction |
| `AUTO_META_TIMEOUT` | `30.0` | Per VLM request timeout |
| `AUTO_META_MAX_TOKENS` | `800` | JSON response token budget |
| `AUTO_META_IMAGE_PROMPT` | unset | Override image prompt |
| `AUTO_META_PDF_PROMPT` | unset | Override PDF first-page prompt |
| `AUTO_META_PDF_MAX_PAGES` | `100` | Skip PDF VLM preview above this page count |
| `AUTO_META_PDF_RENDER_DPI` | `120` | Render DPI for the first-page screenshot |
| `AUTO_META_PDF_MAX_RENDER_PIXELS` | `8000000` | Skip VLM when the first-page render is too large |

## Safety limits

The preview endpoint rejects oversized multipart uploads with HTTP 413 before parsing. Files that sniff as too large or too complex are returned as preview cards with `rejected_reason` and cannot be confirmed.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `UPLOAD_MAX_FILE_BYTES` | `52428800` | Per-file upload cap |
| `UPLOAD_MAX_BATCH_BYTES` | `209715200` | Total batch cap |
| `UPLOAD_MAX_PDF_PAGES` | `500` | Reject PDFs above this page count |
| `UPLOAD_MAX_IMAGE_PIXELS` | `50000000` | Reject images above this pixel count |
| `UPLOAD_SLUG_MAX_LEN` | `80` | Max readable title slug length in final file names |
| `PREVIEW_CACHE_TTL_SECONDS` | `86400` | TTL for unconfirmed preview caches; `<=0` disables cleanup |

## Content-hash dedup & manifest metadata

Each preview cache directory contains a `manifest.json` with two layers:

- `__meta__` — bookkeeping for the cache itself: `created_at` (epoch seconds)
  is written once when the cache is first created and is preserved across
  the VLM-tagging rewrite so it reflects the true cache age. `cleanup_expired_caches`
  prefers this timestamp; legacy caches without `__meta__` fall back to
  `manifest.json` mtime.
- preview entries — keyed by `preview_id`, each holding
  `display_name`, `cached_name`, `source_type`, and the user-facing
  `effective_*` fields.

On `confirm`, the pipeline streams every cached file through SHA-256 and
looks the digest up in the append-only `asset_index.jsonl`. A hit
reuses the existing `asset_id` and `relative_path`; a miss allocates a
new entry. The asset index is also what `DELETE /assets/{id}` uses to
find the on-disk path for a given asset.

## API examples

Preview:

```bash
curl -s -X POST http://127.0.0.1:8011/upload/preview \
  -F "files=@paper.pdf" \
  -F "files=@photo.jpg" \
  | python3 -m json.tool
```

Confirm:

```bash
curl -s -X POST http://127.0.0.1:8011/upload/confirm \
  -H "Content-Type: application/json" \
  -d '{
    "cache_id": "<from preview>",
    "edits": [
      {
        "preview_id": "<from preview>",
        "title": "Edited title",
        "tags": ["custom", "tag"],
        "description": "Optional description"
      }
    ]
  }'
```

The response contains `task_id`. Poll `/tasks/{task_id}` until status is `done`.

## CLI equivalent

The CLI has no editable preview UI, so it accepts all supported previews as-is:

```bash
mmrag parse ./paper.pdf ./photo.jpg
```

This runs preview, confirm, parse, and index in one command.
