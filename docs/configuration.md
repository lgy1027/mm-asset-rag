# Configuration

`mm-asset-rag` reads configuration from environment variables through `mm_asset_rag.settings.Settings`. A `.env` file in the current working directory is loaded automatically.

## Runtime layout

All mutable data lives under `MM_ASSET_RAG_HOME` (default `~/.mm_asset_rag`):

```text
$MM_ASSET_RAG_HOME/
├── assets/
│   ├── pdfs/                # confirmed uploaded PDFs
│   └── images/              # confirmed uploaded images
├── .preview-cache/<id>/     # short-lived upload preview files
├── parsed/<asset_id>/       # PDF page markdown / image OCR JSON
├── captions/<asset_id>.json # VLM captions
├── indexes/qdrant/          # local Qdrant persistence
├── documents.jsonl          # ParsedDocument store
└── tasks.jsonl              # background task history
```

There is no `asset_manifest.json`; uploaded files are auto-sniffed and converted into `Asset` objects during `/upload/confirm`.

## Core variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | `~/.mm_asset_rag` | Runtime data directory |
| `OPENAI_API_KEY` | unset | Chat LLM API key |
| `OPENAI_BASE_URL` | unset | OpenAI-compatible chat base URL |
| `OPENAI_MODEL` | unset | Chat model |
| `LLM_TIMEOUT` | `120.0` | Chat timeout seconds |

When the OpenAI triple is incomplete, `/answer` and `/chat` return evidence-summary fallback answers instead of failing.

## Text embedding

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` fallback | Embedding API key |
| `EMBEDDING_BASE_URL` | `OPENAI_BASE_URL` fallback | Embedding base URL |
| `EMBEDDING_MODEL` | unset | Embedding model |
| `EMBEDDING_BATCH_SIZE` | `5` | Batch size |
| `EMBEDDING_REQUEST_INTERVAL` | `0.25` | Delay between requests |
| `EMBEDDING_RETRY_COUNT` | `5` | Retry attempts |
| `EMBEDDING_TIMEOUT` | `120.0` | Timeout seconds |
| `EMBEDDING_MAX_INPUT_CHARS` | `8192` | Per-text truncation limit |

## Image embedding

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLIP_MODEL` | `clip-ViT-B-32` | Sentence-transformers CLIP model name |
| `IMAGE_PROVIDER` | `lite` | Legacy provider selector |

Install `[clip]` to use sentence-transformers CLIP:

```bash
pip install -e ".[clip]"
```

## Qdrant

| Variable | Default | Purpose |
| --- | --- | --- |
| `QDRANT_URL` | unset | Remote Qdrant URL; unset = local file mode |
| `QDRANT_API_KEY` | unset | Remote Qdrant API key |
| `QDRANT_TEXT_COLLECTION` | `multimodal_text` | Base text collection name |
| `QDRANT_IMAGE_COLLECTION` | `multimodal_image` | Base image collection name |
| `QDRANT_UPSERT_BATCH_SIZE` | `16` | Upsert batch size |
| `QDRANT_BM25_MODEL` | `Qdrant/bm25` | fastembed sparse model |
| `QDRANT_HYBRID_PREFETCH_LIMIT` | `20` | Per-channel prefetch limit |

Collection names auto-suffix by vector dimension, e.g. `multimodal_text_2560d`.

## Retrieval tuning

| Variable | Default | Purpose |
| --- | ---: | --- |
| `HYBRID_WEIGHT_TEXT` | `0.80` | Text-route merge weight |
| `HYBRID_WEIGHT_TEXT_TO_IMAGE` | `0.20` | Text→image merge weight |
| `HYBRID_WEIGHT_IMAGE_TO_IMAGE` | `0.0` | Image→image merge weight when an image query is provided |
| `MAX_CHUNKS_PER_PDF` | unset | Per-PDF chunk cap before text indexing |
| `IMAGE_RELEVANCE_THRESHOLD` | `0.24` | CLIP cosine floor for image routes |
| `IMAGE_PREFILTER_FIELDS` | `tags,asset_id,asset_title` | Payload fields used for sparse image pre-filter |
| `IMAGE_PREFILTER_MIN_TOKEN_LEN` | `3` | Drop shorter tokens from image pre-filter |

Changing `MAX_CHUNKS_PER_PDF` requires `mmrag reindex` to rebuild existing collections.

## Chinese BM25

| Variable | Default | Purpose |
| --- | ---: | --- |
| `BM25_ZH_ENABLED` | `true` | Enable jieba + Okapi sparse vector |
| `BM25_ZH_K1` | `1.5` | BM25 k1 |
| `BM25_ZH_B` | `0.75` | BM25 b |
| `BM25_ZH_VECTOR_NAME` | `bm25_zh` | Qdrant sparse vector name |

## Upload safety limits

These limits protect `/upload/preview` from accidental very large uploads. Oversized multipart bodies return HTTP 413; files that sniff as too large/complex are shown as rejected preview cards and cannot be confirmed.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `UPLOAD_MAX_FILE_BYTES` | `52428800` | Per-file upload cap |
| `UPLOAD_MAX_BATCH_BYTES` | `209715200` | Total multipart batch cap |
| `UPLOAD_MAX_PDF_PAGES` | `500` | Reject confirmed PDFs above this page count |
| `UPLOAD_MAX_IMAGE_PIXELS` | `50000000` | Reject images above this pixel count |
| `UPLOAD_SLUG_MAX_LEN` | `80` | Maximum readable title slug length used in asset file names |

## Upload auto-metadata

The upload preview pipeline can call a VLM once per file to extract title / description / tags as JSON. If the VLM is unconfigured or fails, preview falls back to local sniffing. PDF metadata extraction only renders the first page and has its own guardrails.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `AUTO_META_ENABLED` | `true` | Enable VLM metadata extraction in `/upload/preview` |
| `AUTO_META_TIMEOUT` | `30.0` | Per-file VLM timeout |
| `AUTO_META_MAX_TOKENS` | `800` | JSON response budget |
| `AUTO_META_IMAGE_PROMPT` | unset | Override image prompt |
| `AUTO_META_PDF_PROMPT` | unset | Override PDF-first-page prompt |
| `AUTO_META_PDF_MAX_PAGES` | `100` | Skip PDF VLM preview above this page count |
| `AUTO_META_PDF_RENDER_DPI` | `120` | DPI used for the first-page render |
| `AUTO_META_PDF_MAX_RENDER_PIXELS` | `8000000` | Skip VLM when the rendered first page is too large |

## OCR / VLM backends

| Variable | Default | Purpose |
| --- | ---: | --- |
| `OCR_HTTP_URL` | unset | Optional local OCR service for image text extraction |
| `OCR_HTTP_TIMEOUT` | `60.0` | OCR timeout |
| `VLM_BASE_URL` | `OPENAI_BASE_URL` fallback | VLM endpoint for image captions / auto metadata |
| `VLM_API_KEY` | `OPENAI_API_KEY` fallback | VLM API key |
| `VLM_MODEL` | `OPENAI_MODEL` fallback | VLM model |
| `VLM_TEMPERATURE` | `0.1` | Caption temperature |
| `VLM_MAX_TOKENS` | `2000` | Caption token budget |
| `VLM_TIMEOUT` | `120.0` | Caption timeout |

## PaddleOCR-VL

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PADDLEOCR_VL_API_TOKEN` | unset | Enables PaddleOCR-VL PDF parsing when `pdf_parser=auto` |
| `PADDLEOCR_VL_JOB_URL` | Paddle API URL | Job endpoint |
| `PADDLEOCR_VL_MODEL` | `PaddleOCR-VL-1.6` | Model name |
| `PADDLEOCR_VL_TIMEOUT` | `900.0` | Job timeout |
| `PADDLEOCR_VL_POLL_INTERVAL` | `5.0` | Poll interval |
| `PADDLEOCR_VL_POLL_RETRY` | `5` | Poll retry count |
| `PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY` | `false` | Paddle option |
| `PADDLEOCR_VL_USE_DOC_UNWARPING` | `false` | Paddle option |
| `PADDLEOCR_VL_USE_CHART_RECOGNITION` | `false` | Paddle option |

## Example `.env`

```dotenv
MM_ASSET_RAG_HOME=~/.mm_asset_rag

OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=gemma4:latest

EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=qwen3-embedding:4b

AUTO_META_ENABLED=true
VLM_BASE_URL=http://127.0.0.1:11434/v1
VLM_API_KEY=ollama
VLM_MODEL=gemma4:latest

# Use Qdrant server mode if you want concurrent API + CLI access
# QDRANT_URL=http://127.0.0.1:6333
```
