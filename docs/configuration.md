# Configuration

Every environment variable the codebase reads is declared as a typed
field on [`Settings`](https://github.com/lgy1027/mm-asset-rag/blob/main/mm_asset_rag/settings.py)
(pydantic-settings). The same module is the single source of truth — both
the API and the CLI read from it.

This document groups the fields by purpose and lists defaults,
overrides, and the `.env` keys you should set per environment. For the
canonical list (with types), see
[`Settings` in settings.py](../mm_asset_rag/settings.py).

## Loading

`.env` in the current working directory is loaded automatically at
process startup (`config.load_env()` reads it via `python-dotenv`).
Process-level env vars override `.env` values.

```python
from mm_asset_rag.settings import get_settings

settings = get_settings()  # cached lru_cache singleton
print(settings.openai_model)  # str | None
print(settings.data_dir)      # Path: ~/.mm_asset_rag or MM_ASSET_RAG_HOME
print(settings.has_llm)       # bool: triple (key/url/model) is complete
```

For tests:

```python
from mm_asset_rag.settings import Settings
s = Settings(_env_file=None)   # bypass .env, read os.environ only
```

## Paths

| Field | Env | Default | Purpose |
| --- | --- | --- | --- |
| `mm_asset_rag_home` | `MM_ASSET_RAG_HOME` | `~/.mm_asset_rag` | Root for assets / indexes / task log. |

Layout under `$MM_ASSET_RAG_HOME`:

```
assets/                  # PDFs, images, asset_manifest.json
parsed/<asset_id>/       # per-asset parsed output (markdown pages, OCR JSON)
captions/<asset_id>.json # VLM captions (when VLM is enabled)
indexes/
  qdrant/                # Qdrant local persistence
documents.jsonl          # unified ParsedDocument store
tasks.jsonl              # background-task history (read on startup)
```

## LLM (for `/answer`, `/chat`, `mmrag answer`)

| Field | Env | Default |
| --- | --- | --- |
| `openai_api_key` | `OPENAI_API_KEY` | — |
| `openai_base_url` | `OPENAI_BASE_URL` | — |
| `openai_model` | `OPENAI_MODEL` | — |
| `llm_timeout` | `LLM_TIMEOUT` | `120` (seconds) |

Any local or hosted endpoint works: ollama / vLLM / LM Studio / OpenAI /
DeepSeek / Moonshot / etc., as long as it speaks the OpenAI Chat
Completions protocol.

## Embedding (text)

| Field | Env | Default |
| --- | --- | --- |
| `embedding_api_key` | `EMBEDDING_API_KEY` | falls back to `OPENAI_API_KEY` |
| `embedding_base_url` | `EMBEDDING_BASE_URL` | falls back to `OPENAI_BASE_URL` |
| `embedding_model` | `EMBEDDING_MODEL` | falls back to `OPENAI_MODEL` |
| `embedding_batch_size` | `EMBEDDING_BATCH_SIZE` | `5` |
| `embedding_request_interval` | `EMBEDDING_REQUEST_INTERVAL` | `0.25` (seconds between batches) |
| `embedding_retry_count` | `EMBEDDING_RETRY_COUNT` | `5` |
| `embedding_timeout` | `EMBEDDING_TIMEOUT` | `120` (seconds) |
| `embedding_max_input_chars` | `EMBEDDING_MAX_INPUT_CHARS` | `8192` |

## Image embedding (CLIP)

| Field | Env | Default |
| --- | --- | --- |
| `clip_model` | `CLIP_MODEL` | `clip-ViT-B-32` |

Image embedding requires the optional `[clip]` extra:
`pip install "mm-asset-rag[clip]"`.

## Qdrant

| Field | Env | Default |
| --- | --- | --- |
| `qdrant_url` | `QDRANT_URL` | — (omit to use local file mode) |
| `qdrant_api_key` | `QDRANT_API_KEY` | — |
| `qdrant_text_collection` | `QDRANT_TEXT_COLLECTION` | `multimodal_text` |
| `qdrant_image_collection` | `QDRANT_IMAGE_COLLECTION` | `multimodal_image` |
| `qdrant_upsert_batch_size` | `QDRANT_UPSERT_BATCH_SIZE` | `16` |
| `qdrant_bm25_model` | `QDRANT_BM25_MODEL` | `Qdrant/bm25` |
| `qdrant_hybrid_prefetch_limit` | `QDRANT_HYBRID_PREFETCH_LIMIT` | `20` |

When the embedding dimension changes, the active collection is
auto-suffixed (`f"{base}_{dim}d"`, e.g. `multimodal_text_2560d`).

## Parser defaults (override per `/upload`)

| Field | Env | Default |
| --- | --- | --- |
| `pdf_parser` | `PDF_PARSER` | `auto` (`auto` / `pymupdf` / `paddleocr_vl`) |
| `enable_ocr` | `ENABLE_OCR` | `false` |
| `enable_vlm` | `ENABLE_VLM` | `false` |
| `image_provider` | `IMAGE_PROVIDER` | `lite` (`lite` / `sentence_transformers`) |
| `auto_index` | `AUTO_INDEX` | `true` |

The `/upload` form fields (`pdf_parser`, `enable_ocr`, `enable_vlm`,
`image_provider`, `auto_index`) override these defaults per request.

## PaddleOCR-VL

Required only when `PDF_PARSER=paddleocr_vl` (or `auto` with the
token set).

| Field | Env | Default |
| --- | --- | --- |
| `paddleocr_vl_api_token` | `PADDLEOCR_VL_API_TOKEN` | — |
| `paddleocr_vl_job_url` | `PADDLEOCR_VL_JOB_URL` | `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs` |
| `paddleocr_vl_model` | `PADDLEOCR_VL_MODEL` | `PaddleOCR-VL-1.6` |
| `paddleocr_vl_timeout` | `PADDLEOCR_VL_TIMEOUT` | `900` |
| `paddleocr_vl_poll_interval` | `PADDLEOCR_VL_POLL_INTERVAL` | `5` |
| `paddleocr_vl_poll_retry` | `PADDLEOCR_VL_POLL_RETRY` | `5` |
| `paddleocr_vl_use_doc_orientation_classify` | `PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY` | `false` |
| `paddleocr_vl_use_doc_unwarping` | `PADDLEOCR_VL_USE_DOC_UNWARPING` | `false` |
| `paddleocr_vl_use_chart_recognition` | `PADDLEOCR_VL_USE_CHART_RECOGNITION` | `false` |

## OCR HTTP (image assets)

| Field | Env | Default |
| --- | --- | --- |
| `ocr_http_url` | `OCR_HTTP_URL` | — |
| `ocr_http_timeout` | `OCR_HTTP_TIMEOUT` | `60` |

## VLM (image captioning)

| Field | Env | Default |
| --- | --- | --- |
| `vlm_base_url` | `VLM_BASE_URL` | falls back to `OPENAI_BASE_URL` |
| `vlm_api_key` | `VLM_API_KEY` | falls back to `OPENAI_API_KEY` |
| `vlm_model` | `VLM_MODEL` | falls back to `OPENAI_MODEL` |
| `vlm_temperature` | `VLM_TEMPERATURE` | `0.1` |
| `vlm_timeout` | `VLM_TIMEOUT` | `120` |

## See also

- [`.env.example`](../.env.example) — copy to `.env` and fill in.
- [`Settings` source](../mm_asset_rag/settings.py) — canonical field list with types.
- [HTTP API](api.md) — request / response shapes and example payloads.