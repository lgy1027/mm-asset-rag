# Configuration

`mm-asset-rag` reads configuration from environment variables through `mm_asset_rag.settings.Settings`. A `.env` file in the current working directory is loaded automatically.

## Runtime layout

All mutable data lives under `MM_ASSET_RAG_HOME` (default `~/.mm_asset_rag`):

```text
$MM_ASSET_RAG_HOME/
├── assets/
│   ├── pdfs/                # confirmed uploaded PDFs
│   ├── images/              # confirmed uploaded images
│   └── documents/           # confirmed office/text (docx/pptx/xlsx/html/md/txt)
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

### LLM ↔ VLM bidirectional fallback

The chat LLM channel (`/answer`, `/chat`) and the image-channel VLM (image caption, `/upload/preview` auto-meta, tier-3 multimodal answer) can each use a different provider. Configure either `OPENAI_*` or `VLM_*` alone and both channels work; configure both to split by purpose (e.g. local ollama for chat, MiniMax-M3 for vision).

- `/answer` LLM channel: `OPENAI_*` preferred, falls back to `VLM_*`.
- `/upload/preview` VLM channel: `VLM_*` preferred, falls back to `OPENAI_*`.

When neither triple is complete, `/answer` and `/chat` return evidence-summary fallback answers instead of failing.

## API auth + host guard

The HTTP API ships with two independent security layers, both with safe loopback defaults so a developer's `mmrag-api` works zero-config:

- **TrustedHostMiddleware** locks the API to loopback (`127.0.0.1`, `localhost`, `[::1]`) by default. A malicious web page cannot reach the API via DNS rebinding — the browser SOP preflight blocks cross-origin JSON POST, but multipart `/upload/preview` is a simple request, and the rebinding trick can read GET responses without it. Set `MMRAG_TRUSTED_HOSTS` to your public hostname(s) when deploying behind a reverse proxy, or `*` to disable the check (unsafe without a token).
- **Bearer token** guards the destructive + write endpoints (`DELETE /assets/*`, `POST /tasks/*/retry`, `POST /upload/preview`, `POST /upload/confirm`, `POST /eval`). Leave `MMRAG_API_TOKEN` unset to keep the zero-config default (no auth); set it when exposing the API beyond localhost. Clients pass it as `Authorization: Bearer <token>` or `X-API-Key: <token>`. Read endpoints (`/search`, `/answer`, `/chat`, `/assets`, `/tasks`, `/health`, `/`) stay open regardless so the bundled web UI's same-origin fetches keep working without a token.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MMRAG_API_TOKEN` | unset | Static bearer token for destructive + write endpoints; unset = no auth |
| `MMRAG_TRUSTED_HOSTS` | `127.0.0.1,localhost,[::1]` | Comma-separated trusted Host headers; `*` disables the check |

When deploying on a public host, set **both** `MMRAG_API_TOKEN` (so destructive endpoints can't be called anonymously) and `MMRAG_TRUSTED_HOSTS` (so the loopback-only host check accepts your public hostname).

## Text embedding

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_BACKEND` | `openai` | `openai` (OpenAI-compatible /v1/embeddings) or `sentence_transformers` (local HF model) |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` fallback | Embedding API key |
| `EMBEDDING_BASE_URL` | `OPENAI_BASE_URL` fallback | Embedding base URL |
| `EMBEDDING_MODEL` | unset | Embedding model |
| `EMBEDDING_BATCH_SIZE` | `5` | Batch size |
| `EMBEDDING_REQUEST_INTERVAL` | `0.25` | Delay between requests |
| `EMBEDDING_RETRY_COUNT` | `5` | Retry attempts |
| `EMBEDDING_TIMEOUT` | `120.0` | Timeout seconds |
| `EMBEDDING_MAX_INPUT_CHARS` | `8192` | Per-text truncation limit |
| `EMBEDDING_SPARSE_ENABLED` | `auto` | `auto` probes the embedder (only bge-m3 via sentence-transformers exposes it); `true`/`false` force on/off |
| `EMBEDDING_COLBERT_ENABLED` | `auto` | Same probe pattern for the ColBERT multi-vector channel |

When `auto` resolves to enabled (bge-m3), the text collection gains extra sparse / multi-vector fields; the indexer raises a schema-mismatch error so you run `mmrag reindex` to rebuild. The OpenAI-compatible embedder never exposes these, so the default config adds no fields and needs no reindex.

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
| `QDRANT_HYBRID_PREFETCH_LIMIT` | `50` | Per-channel prefetch limit |
| `QDRANT_ACTIVE_TEXT_COLLECTION` | unset | Force a specific active text collection (overrides `{base}_{dim}d` auto-resolution) |
| `QDRANT_ACTIVE_IMAGE_COLLECTION` | unset | Force a specific active image collection (overrides `{base}_{dim}d` auto-resolution) |

Collection names auto-suffix by vector dimension, e.g. `multimodal_text_2560d`. Leave `QDRANT_ACTIVE_*_COLLECTION` unset to use the auto suffix; set them only to pin a collection that does not match the current embedder's dim.

## Retrieval tuning

| Variable | Default | Purpose |
| --- | ---: | --- |
| `HYBRID_WEIGHT_TEXT` | `0.80` | Text-route merge weight |
| `HYBRID_WEIGHT_TEXT_TO_IMAGE` | `0.20` | Text→image merge weight |
| `HYBRID_WEIGHT_IMAGE_TO_IMAGE` | `0.15` | Image→image merge weight when an image query is provided |
| `MIN_SCORE` | `0.0` | Soft low-end guard on the final RRF-fused score (0.0 disables; ~0.001 trims tiny-tail noise) |
| `RRF_WEIGHT_DENSE` | `1.0` | Per-channel RRF bias for the dense prefetch |
| `RRF_WEIGHT_BM25` | `1.0` | Per-channel RRF bias for the BM25-en prefetch |
| `RRF_WEIGHT_BM25_ZH` | `1.0` | Per-channel RRF bias for the BM25-zh prefetch (raise to ~1.5 for Chinese-only recall) |
| `MAX_CHUNKS_PER_PDF` | unset | Per-PDF chunk cap before text indexing |
| `IMAGE_RELEVANCE_THRESHOLD` | `0.24` | CLIP cosine floor for image routes |
| `IMAGE_PREFILTER_FIELDS` | `tags,asset_id,asset_title` | Payload fields used for sparse image pre-filter |
| `IMAGE_PREFILTER_MIN_TOKEN_LEN` | `3` | Drop shorter tokens from image pre-filter |

Changing `MAX_CHUNKS_PER_PDF` requires `mmrag reindex` to rebuild existing collections.

## Chunk keyword enrichment

Appends a `关键词: ...` footer (jieba TextRank) to every PDF chunk's text before indexing, so the BM25 channel has explicit tokens to match short queries like `联宝 ESG` against long PDF bodies where the tokens would otherwise be diluted. Disable for non-Chinese corpora or when jieba is unavailable. Requires `mmrag reindex` to affect existing collections.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `ENRICH_CHUNK_WITH_KEYWORDS` | `true` | Append jieba TextRank keyword footer to each chunk |
| `ENRICH_CHUNK_KEYWORD_TOP_K` | `8` | Number of keywords in the footer |
| `ENRICH_CHUNK_LANGUAGE` | `auto` | `zh` / `en` / `auto` (jieba first, stopword-frequency fallback) |

## Recursive chunking

After heading-based splitting, each section body is recursively split to a token budget with overlap, so long sections don't produce oversized chunks that dilute BM25 / get truncated by the embedder / mislead the cross-encoder reranker. Token counts default to a char approximation (token ≈ chars/3.5, mixed zh/en) so no tokenizer is required; set `CHUNK_TOKENIZER` to a HuggingFace id for exact counts (falls back to char approx if unavailable). Changing these requires `mmrag reindex` (chunk text is re-derived at parse time, so a full `mmrag parse` is needed for existing assets).

| Variable | Default | Purpose |
| --- | ---: | --- |
| `CHUNK_TARGET_TOKENS` | `500` | Target tokens per chunk (benchmark sweet spot ~500) |
| `CHUNK_MAX_TOKENS` | `800` | Hard max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap tokens between adjacent chunks |
| `CHUNK_TOKENIZER` | unset | HF tokenizer id for exact counts; unset = char approximation |

## Semantic dedup (asset-level)

On top of the exact content-hash dedup (`find_by_sha256`), the asset index keeps a title / first-chunk embedding index. A new asset whose embedding is cosine-close to an existing active asset (different sha256) reuses the existing `asset_id` so near-duplicates aren't re-indexed. This threshold is the cosine cutoff; default `0.92` matches LlamaIndex's `DeduplicationModule`. Note: this dedup path is implemented in `asset_index` but not yet wired into the ingest/upload pipeline, so it does not trigger on real uploads today.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `DEDUP_SEMANTIC_THRESHOLD` | `0.92` | Cosine cutoff for near-duplicate asset reuse |

## PDF embedded-image extraction

PyMuPDF parses text only by default; embedded figures are dropped. When `PDF_EXTRACT_IMAGES` is on, the parser pulls every image a page references into `parsed/<id>/images/` and attaches the figures a chunk references (or sits next to) to that chunk's `metadata["images"]`. The figures ride in the text hit's payload — surfaced to the LLM (a `关联图片` hint citing the figure caption) and the web UI (a thumbnail served by `GET /parsed-image/{asset_id}/{filename}`). Images are **not** embedded into the vector index (that is tier 2); they are an attachment of the text hit. `PDF_IMAGE_MIN_DIM` filters logos / icons. Requires `mmrag reindex` (or a fresh `mmrag parse`) to populate `images` on existing chunks.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PDF_EXTRACT_IMAGES` | `true` | Extract embedded images + attach to text hits |
| `PDF_IMAGE_MIN_DIM` | `80` | Skip images with either dimension below this (logos/icons) |

## Tier-3 multimodal answer

When `ANSWER_WITH_IMAGES` is on, `/answer` and `/chat/stream` inject each hit's associated images (base64 data URLs) into the chat request as `image_url` content parts alongside the text evidence, so a vision-capable LLM can *see* figure pixels and answer questions whose answer lives in the figure (numbers / tables / flowcharts the body text doesn't repeat). Requires a vision-capable chat model (`OPENAI_MODEL` must be multimodal — e.g. MiniMax-M3, or ollama `gemma3` / `llama3.2-vision`). If the configured model rejects images, the call is retried text-only so the feature is safe to toggle without breaking `/answer`. No effect when `PDF_EXTRACT_IMAGES` is off (no images on the hits to inject).

**Most deployments should leave this off.** Tier-1 already attaches every hit's figures as `metadata.images` so the web UI shows thumbnails below each source — the user sees the figures directly without the LLM having to "read" them, and the LLM context still carries a `关联图片: 图N: <caption>` line so the answer can reference figures by number. Tier-3 is only worth enabling when users frequently ask questions whose answer lives *only* in the image pixels (chart numbers, table values, flowchart steps the body text doesn't repeat) and the deployment has a vision-capable LLM available. For text-only LLMs the tier-3 toggle has no benefit — leave it off.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `ANSWER_WITH_IMAGES` | `false` | opt-in — inject hit images into the LLM chat request |
| `ANSWER_IMAGE_MAX_PER_HIT` | `2` | Max images sent per hit (bounds token cost; hard global cap is 4) |

## Contextual Retrieval

Anthropic-style chunk context: each chunk gets a short LLM-generated preamble situating it within its document, prepended to the embedding/BM25 input so dense + sparse channels can disambiguate generic terms. **Enabled by default** — it costs ~1 LLM call per chunk, generated at parse time and cached under `parsed/<id>/context.jsonl` so `mmrag reindex` reuses it without re-calling the LLM. Disable with `CONTEXTUAL_ENABLED=false` (or `mmrag parse --no-contextual` on the CLI) when no LLM is configured or to skip the per-chunk calls.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `CONTEXTUAL_ENABLED` | `true` | Master switch (default on; set `false` to opt out) |
| `CONTEXTUAL_MODEL` | unset (→ `OPENAI_MODEL`) | LLM model override |
| `CONTEXTUAL_CONCURRENCY` | `4` | Parallel chunk-context calls |
| `CONTEXTUAL_CHUNK_MAX_CHARS` | `8000` | Cap chunk text fed to the LLM |
| `CONTEXTUAL_TIMEOUT` | `60` | Per-call HTTP timeout (seconds) |

## Image caption for embedded figures (opt-in)

Document-embedded figures (docx/pptx pictures via markitdown/docling, PDF figures via PyMuPDF) are saved to `parsed/<id>/images/` and associated with chunks, but their *content* is otherwise invisible to the text index — a slide whose only payload is a diagram is unsearchable. When enabled, each embedded figure with no existing caption gets a VLM-generated Chinese description appended to its chunk's text so the figure's semantics enter the dense + BM25 channels. The caption is also recorded in `metadata["images"][*]["caption"]` so the answer layer can cite it.

This is the **text-route** path only: embedded figures are *not* sent to the CLIP image index — that channel stays reserved for standalone `images/` uploads (`source_type=image`). Works with any OpenAI-compatible VLM via `VLM_*`. Cost: ~1 VLM call per embedded figure at parse time. Generated before Contextual Retrieval so the contextual LLM sees caption-enriched chunks. Cached under `captions/<id>.jsonl` keyed by image path so `mmrag reindex` and force re-parse reuse it without re-calling the VLM (figure bytes are stable across re-parses). When `VLM_*` is unconfigured the step degrades to a no-op — safe to leave on.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `IMAGE_CAPTION_ENABLED` | `false` | opt-in master switch |
| `IMAGE_CAPTION_CONCURRENCY` | `4` | Parallel figure-caption VLM calls |

## Two-stage reranker

bge-m3's model card recommends "hybrid retrieval + re-ranking": pull a candidate pool with dense + BM25, then score each `(query, doc)` pair with a cross-encoder. Catches high-score false positives that `MIN_SCORE` cannot. **Enabled by default** — it adds ~50-200ms latency per query and the model is ~2GB on first download (`BAAI/bge-reranker-v2-m3`). Runs locally via `sentence-transformers.CrossEncoder` (same dep as the bge-m3 embedder); no ollama / API. Disable with `RERANKER_ENABLED=false` when latency / download cost is a concern.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `RERANKER_ENABLED` | `true` | Master switch (default on; set `false` to opt out) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace cross-encoder model id |
| `RERANKER_TOP_N` | `30` | Candidates fetched from each route before rerank (≤ `QDRANT_HYBRID_PREFETCH_LIMIT`) |
| `RERANKER_TOP_K` | unset (→ caller's `top_k`) | Final result count after rerank |
| `RERANKER_HYBRID_BLEND` | `0.6` | Cross-encoder weight in the final blended rank (0 = hybrid only, 1 = pure reranker) |

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
| `PREVIEW_CACHE_TTL_SECONDS` | `86400` | TTL for `/upload/preview` staging files (background sweep deletes expired entries) |

## Upload auto-metadata

The upload preview pipeline can call a VLM once per file to extract title / description / tags as JSON. If the VLM is unconfigured or fails, preview falls back to local sniffing. PDF metadata extraction only renders the first page and has its own guardrails.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `AUTO_META_ENABLED` | `true` | Enable VLM metadata extraction in `/upload/preview` |
| `AUTO_META_TIMEOUT` | `30.0` | Per-file VLM timeout |
| `AUTO_META_MAX_TOKENS` | `800` | JSON response budget |
| `AUTO_META_MAX_CONCURRENCY` | `3` | Parallel VLM calls across a multi-file preview batch |
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

## PDF parser selection

The PDF parser is chosen by `PDF_PARSER` (CLI `--pdf-parser`):

| Value | Backend | Notes |
| --- | --- | --- |
| `auto` | PyMuPDF → fallback | Default. Fast local parse first, falls back to OCR when the result looks scanned (see below) |
| `pymupdf` | PyMuPDF | Local, text-only. Drops embedded figures unless `PDF_EXTRACT_IMAGES` is on |
| `paddleocr_vl` | PaddleOCR-VL | Online API; needs `PADDLEOCR_VL_API_TOKEN`. Best for scanned / image-only PDFs |
| `docling` | docling | Local multi-format parser; needs the `[docling]` extra. Pulls torch / transformers |

`pymupdf` remains a hard dependency; `paddleocr_vl` is online; `docling` is an optional extra (`pip install -e ".[docling]"`). Without the extra, `--pdf-parser docling` raises a friendly install hint at parse time rather than an `ImportError` at startup.

## Document parser selection

Office / text documents (`docx` / `pptx` / `xlsx` / `html` / `md` / `txt` — the `document` source type `sniff` assigns) are parsed by the backend chosen with `DOCUMENT_PARSER` (CLI `--document-parser`):

| Value | Backend | Notes |
| --- | --- | --- |
| `markitdown` | MarkItDown | Default. Core dependency (pure Python, no ML stack). docx/pptx/xlsx converters ship via the `markitdown[docx,pptx,xlsx]` extra bundled in core |
| `docling` | docling | Optional heavy backend (torch / transformers). Needs the `[docling]` extra. Layout-aware; use when MarkItDown's structural extraction isn't enough |

Both backends produce the same `DocumentIR`, so chunking / image association / contextual enrichment are identical downstream. MarkItDown decodes docx/pptx base64-embedded images to `parsed/<id>/images/` and rewrites the refs, so embedded images attach to their chunk and reach the answer layer — same on-disk layout as the docling / PaddleOCR paths. (HTML relative-path images are passed through as-is in v1; they don't associate but don't error.)

### Scanned-PDF fallback (auto parser)

The `auto` parser runs fast local PyMuPDF first, then falls back to an OCR backend when the result looks like a scan (image-only, near-zero text). `PDF_SCAN_TEXT_THRESHOLD` is the total non-empty chars/page budget below which a document is treated as scanned — corpus-agnostic (pure char density, no domain words): `total_chars < threshold * page_count`. `PDF_SCAN_FALLBACK_PARSER` picks the OCR backend: `paddleocr_vl` (default, online API, needs `PADDLEOCR_VL_API_TOKEN`) or `docling` (local, needs the `[docling]` extra). Disable with `PDF_SCAN_FALLBACK_ENABLED=false` to always stay on PyMuPDF (the pre-IR `auto` behaviour).

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PDF_SCAN_FALLBACK_ENABLED` | `true` | Master switch for the scanned-PDF fallback in `auto` mode |
| `PDF_SCAN_TEXT_THRESHOLD` | `10` | Avg non-empty chars/page below which a PDF is treated as scanned |
| `PDF_SCAN_FALLBACK_PARSER` | `paddleocr_vl` | Fallback backend: `paddleocr_vl` or `docling` |

The threshold default of `10` is tuned for genuinely scanned (image-only) PDFs, which yield ~0 extractable chars. A text PDF with a single short page (~45 chars) stays on PyMuPDF since `45 ≥ 10 * 1`. Raise it if your corpus has dense-figure PDFs whose thin text layers should trigger OCR.

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
