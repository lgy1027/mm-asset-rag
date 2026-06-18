# Configuration

All configuration is read from environment variables (a `.env` file in the current directory is loaded automatically via `python-dotenv`).

## Paths

| Variable | Default | Purpose |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | `~/.mm_asset_rag` | Root directory for parsed documents, indexes, and reports. Auto-created on first access. |

Layout under `$MM_ASSET_RAG_HOME`:

```
assets/                  # put your PDFs, images, and asset_manifest.json here
parsed/<asset_id>/       # per-asset parsed output (markdown pages, OCR JSON)
captions/<asset_id>.json # VLM captions (when VLM is enabled)
indexes/
  text/                  # LlamaIndex persistence (only used by text_index module)
  qdrant/                # Qdrant local persistence
documents.jsonl          # unified ParsedDocument store
eval_report.json         # latest /eval output
```

## LLM (for `/answer`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_BASE_URL` | — | OpenAI-compatible chat completion endpoint. |
| `OPENAI_API_KEY` | — | Bearer token. |
| `OPENAI_MODEL` | — | Model name. |
| `LLM_TIMEOUT` | `120` | Seconds. |

If any of `BASE_URL` / `API_KEY` / `MODEL` is missing, `/answer` returns an evidence-summary fallback instead of an LLM-generated answer.

## Embedding (text)

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `openai` | `openai` for real API, `mock` for SHA-256-based fake vectors. |
| `EMBEDDING_API_KEY` | falls back to `OPENAI_API_KEY` | Bearer token. |
| `EMBEDDING_BASE_URL` | falls back to `OPENAI_BASE_URL` | Endpoint root. |
| `EMBEDDING_MODEL` | — | Embedding model name. |
| `EMBEDDING_BATCH_SIZE` | `5` | Texts per request. |
| `EMBEDDING_REQUEST_INTERVAL` | `0.25` | Seconds between batches. |
| `EMBEDDING_RETRY_COUNT` | `5` | On 429 / 5xx. |
| `EMBEDDING_TIMEOUT` | `120` | Seconds. |
| `MOCK_EMBEDDING_DIM` | `384` | Dimension of mock vectors. |

## Image embedding (CLIP)

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLIP_MODEL` | `clip-ViT-B-32` | sentence-transformers model name. |
| `QDRANT_URL` | — | If set, use a remote Qdrant server instead of local file mode. |
| `QDRANT_API_KEY` | — | Bearer token for `QDRANT_URL`. |

`ImageEmbeddingProvider` requires `pip install mm-asset-rag[clip]`.

## Qdrant collections

| Variable | Default | Purpose |
| --- | --- | --- |
| `QDRANT_TEXT_COLLECTION` | `multimodal_text` | Text vector collection. |
| `QDRANT_IMAGE_COLLECTION` | `multimodal_image` | Image vector collection. |
| `QDRANT_UPSERT_BATCH_SIZE` | `16` | Points per `upsert` call. |

When the embedding dimension changes, the collection name auto-suffixes the dimension (e.g. `multimodal_text_1024d`) and the active name is stashed in `QDRANT_ACTIVE_TEXT_COLLECTION` / `QDRANT_ACTIVE_IMAGE_COLLECTION` so subsequent processes target the right collection.

## PDF parsing

| Variable | Default | Purpose |
| --- | --- | --- |
| `PADDLEOCR_VL_API_TOKEN` | — | Required for `--pdf-parser paddleocr_vl`. |
| `PADDLEOCR_VL_JOB_URL` | `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs` | Job submission endpoint. |
| `PADDLEOCR_VL_MODEL` | `PaddleOCR-VL-1.6` | Model name. |
| `PADDLEOCR_VL_TIMEOUT` | `300` | Seconds (per HTTP call). |
| `PADDLEOCR_VL_POLL_INTERVAL` | `5` | Seconds between poll requests. |
| `PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY` | `false` | Toggles optional PaddleOCR-VL feature. |
| `PADDLEOCR_VL_USE_DOC_UNWARPING` | `false` | Same. |
| `PADDLEOCR_VL_USE_CHART_RECOGNITION` | `false` | Same. |

When `pdf_parser="auto"` (the default), PaddleOCR-VL is used if `PADDLEOCR_VL_API_TOKEN` is set, otherwise PyMuPDF is used.

## Image parsing (OCR + VLM)

| Variable | Default | Purpose |
| --- | --- | --- |
| `OCR_HTTP_URL` | `http://127.0.0.1:8000/ocr` | Local OCR HTTP service. |
| `OCR_HTTP_TIMEOUT` | `60` | Seconds. |
| `VLM_BASE_URL` | falls back to `OPENAI_BASE_URL` | VLM endpoint. |
| `VLM_API_KEY` | falls back to `OPENAI_API_KEY` | Bearer token. |
| `VLM_MODEL` | falls back to `OPENAI_MODEL` | Model name. |
| `VLM_TEMPERATURE` | `0.1` | Sampling temperature. |
| `VLM_TIMEOUT` | `120` | Seconds. |