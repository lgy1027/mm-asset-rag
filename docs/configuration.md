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
| `bm25_zh_enabled` | `BM25_ZH_ENABLED` | `true` |
| `bm25_zh_k1` | `BM25_ZH_K1` | `1.5` |
| `bm25_zh_b` | `BM25_ZH_B` | `0.75` |
| `bm25_zh_vector_name` | `BM25_ZH_VECTOR_NAME` | `bm25_zh` |

When the embedding dimension changes, the active collection is
auto-suffixed (`f"{base}_{dim}d"`, e.g. `multimodal_text_2560d`).

### Chinese BM25 (`bm25_zh`)

The Qdrant text collection stores three vector kinds when
`bm25_zh_enabled` is true:

- `dense` — OpenAI-compatible text embeddings (default `qwen3-embedding:4b`).
- `bm25` — fastembed's English `Qdrant/bm25` sparse vector (RRF-fused with dense at query time).
- `bm25_zh` — `jieba.cut` + Okapi BM25 sparse vector from
  `mm_asset_rag.bm25_zh`. Catches token-level Chinese recall that the
  English BM25 misses.

`_hybrid_text_query` prefetches all three and fuses via RRF. The
Chinese IDF table is persisted to
`$MM_ASSET_RAG_HOME/indexes/bm25_zh_idf.json` so query-time encoding
does not re-tokenise the corpus. Setting `bm25_zh_enabled=false` and
running `mmrag reindex` produces a 2-vector collection (English only)
for backwards compatibility.

## Retrieval tuning

| Field | Env | Default | Purpose |
| --- | --- | --- | --- |
| `hybrid_weight_text` | `HYBRID_WEIGHT_TEXT` | `0.80` | Weight of the dense + BM25 (RRF) text route. |
| `hybrid_weight_text_to_image` | `HYBRID_WEIGHT_TEXT_TO_IMAGE` | `0.20` | Weight of the CLIP text-to-image route. |
| `hybrid_weight_image_to_image` | `HYBRID_WEIGHT_IMAGE_TO_IMAGE` | `0.0` | Weight of the CLIP image-to-image route. Only consulted when an `image_path` is supplied. |
| `max_chunks_per_pdf` | `MAX_CHUNKS_PER_PDF` | unset (no cap) | Per-asset chunk cap applied during `mmrag index`. |

`hybrid_search` merges the three routes by max-normalizing each
group's scores and taking a weighted sum by `asset_id`. The weights
list passed to the merge is built from whichever routes actually
participate — the image-to-image route is only included when an
`image_path` is supplied *and* `hybrid_weight_image_to_image > 0`.

### `max_chunks_per_pdf`

On the bundled sample set, three PDFs (`clip` 48 chunks, `flamingo`
54, `gpt3` 75) contribute roughly a third of every dense top-5
ranking. Capping each asset at `MAX_CHUNKS_PER_PDF` chunks
(selected by a local BM25 Okapi score against the asset's title)
gives every asset equal say in the dense ranking. Recommended value
on the bundled set: `10` (reduces the 897-chunk index to ~365
chunks; rebuild with `mmrag reindex` to take effect).

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
token set). Use this parser for **scanned PDFs** (no embedded text
layer): `parse_with_paddleocr_vl` submits the document to the
PaddleOCR-VL API and writes one markdown page per response row to
`$MM_ASSET_RAG_HOME/parsed/<asset_id>/page_N.md`. With
`PADDLEOCR_VL_USE_CHART_RECOGNITION=true` it also extracts charts and
tables; with `PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY=true` it
auto-rotates mis-oriented scans.

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

Image OCR is enabled per-upload via `ENABLE_OCR=true` (or the
`enable_ocr` flag on `/upload`). When the env var is set, every
parsed image is POSTed to the local OCR service with a base64 payload
and the structured `blocks` response is normalised and stored under
`$MM_ASSET_RAG_HOME/parsed/<asset_id>/ocr.json`. See
`scripts/expand_corpus.py` for an example pipeline that exercises this
code path.

For **PDFs** (including scanned PDFs without an embedded text layer)
use `PDF_PARSER=paddleocr_vl` — see the PaddleOCR-VL section below.
The CLI flag is `mmrag parse --pdf-parser paddleocr_vl`.

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