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
├── parsed/<cache_key>/      # internal cache resolved from a document version
├── captions/<cache_key>.jsonl # internal VLM-caption cache
├── indexes/qdrant/          # local Qdrant persistence
├── documents.jsonl          # ParsedDocument store
└── tasks.db                  # background task history (SQLite)
```

There is no `asset_manifest.json`; `/upload/confirm` creates a logical `Document`, an immutable `DocumentVersion`, and its physical file record. Public lifecycle and retrieval interfaces use `document_id` and `version_id`; physical assets and cache keys remain internal implementation details.

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
- **Bearer token** guards the destructive + write endpoints (`POST /tasks/*/retry`, `POST /upload/preview`, `POST /upload/confirm`, `POST /eval`). Leave `MMRAG_API_TOKEN` unset to keep the zero-config default (no auth); set it when exposing the API beyond localhost. Clients pass it as `Authorization: Bearer <token>` or `X-API-Key: <token>`. Read endpoints (`/search`, `/answer`, `/chat`, `/documents`, `/tasks`, `/health`, `/`) stay open regardless so the bundled web UI's same-origin fetches keep working without a token. The obsolete public asset list/detail/delete lifecycle has been removed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MMRAG_API_TOKEN` | unset | Static bearer token for destructive + write endpoints; unset = no auth |
| `MMRAG_TRUSTED_HOSTS` | `127.0.0.1,localhost,[::1]` | Comma-separated trusted Host headers; `*` disables the check |
| `MMRAG_API_HOST` | `127.0.0.1` | Uvicorn listener; set `0.0.0.0` to receive LAN traffic |
| `MMRAG_API_PORT` | `8011` | Uvicorn listener port |

For LAN access, set `MMRAG_API_HOST=0.0.0.0` and include the LAN address or
hostname in `MMRAG_TRUSTED_HOSTS`. When deploying on a public host, set
**both** `MMRAG_API_TOKEN` (so destructive endpoints can't be called
anonymously) and `MMRAG_TRUSTED_HOSTS` (so the loopback-only host check accepts
your public hostname).

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
| `IMAGE_PROVIDER` | `lite` | 占位字符串,无独立 lite embedder 实现。实际图片索引依赖 `[clip]` extra 的 sentence-transformers CLIP(`get_default_image_embedder` 只实例化 CLIP);`lite` 在缺 `[clip]` 时效果是"图片跳过索引"而非"用轻量 embedder",`sentence_transformers` 显式要求 CLIP。保留 `lite` 仅为兼容旧 `.env`,行为等价于"未配置图片 embedder" |

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
| `QDRANT_BM25_CACHE_DIR` | unset (fastembed platform default) | Override fastembed's BM25 model cache directory. fastembed does not read any `FASTEMBED_CACHE_PATH`-style env var on its own, so this is the single knob to pin it. Set explicitly when the platform default is ephemeral (e.g. macOS sandboxed `TMPDIR=/var/folders/...`). |
| `QDRANT_HYBRID_PREFETCH_LIMIT` | `50` | Per-channel prefetch limit |
| `QDRANT_ACTIVE_TEXT_COLLECTION` | unset | Force a specific active text collection (overrides `{base}_{dim}d` auto-resolution) |
| `QDRANT_ACTIVE_IMAGE_COLLECTION` | unset | Force a specific active image collection (overrides `{base}_{dim}d` auto-resolution) |

Collection names auto-suffix by vector dimension, e.g. `multimodal_text_2560d`. Leave `QDRANT_ACTIVE_*_COLLECTION` unset to use the auto suffix; set them only to pin a collection that does not match the current embedder's dim.

The built-in `QdrantBackend` is registered as the implementation of the
search and indexing backend ports. Application callers use `SearchService`
and those ports, so Qdrant settings affect the active adapter without making
API, CLI, answer, or evaluation code depend on Qdrant helper functions.

## Retrieval tuning

| Variable | Default | Purpose |
| --- | ---: | --- |
| `HYBRID_WEIGHT_TEXT` | `0.80` | Text-route merge weight (used when `HYBRID_INTENT_ROUTING_ENABLED=false`) |
| `HYBRID_WEIGHT_TEXT_TO_IMAGE` | `0.20` | Text→image merge weight (used when `HYBRID_INTENT_ROUTING_ENABLED=false`) |
| `HYBRID_WEIGHT_IMAGE_TO_IMAGE` | `0.15` | Image→image merge weight when an image query is provided (used when `HYBRID_INTENT_ROUTING_ENABLED=false`) |
| `HYBRID_INTENT_ROUTING_ENABLED` | `false` | Master switch for per-intent RRF weight routing. When ON, `hybrid_search` picks weights via a local query classifier (CJK ratio / length / stopword) — see [Per-intent RRF weights](#per-intent-rrf-weights) below |
| `HYBRID_INTENT_WEIGHTS_PRECISE_KEYWORD` | unset | Override the `PRECISE_KEYWORD` intent triple (JSON or CSV `text,t2i,i2i`) |
| `HYBRID_INTENT_WEIGHTS_DESCRIPTIVE` | unset | Override the `DESCRIPTIVE` intent triple |
| `HYBRID_INTENT_WEIGHTS_ENTITY_LOOKUP` | unset | Override the `ENTITY_LOOKUP` intent triple |
| `HYBRID_INTENT_WEIGHTS_CHINESE` | unset | Override the `CHINESE` intent triple |
| `MIN_SCORE` | `0.0` | Soft low-end guard on the final RRF-fused score (0.0 disables; ~0.001 trims tiny-tail noise) |
| `RRF_WEIGHT_DENSE` | `1.0` | Per-channel RRF bias for the dense prefetch |
| `RRF_WEIGHT_BM25` | `1.0` | Per-channel RRF bias for the BM25-en prefetch |
| `RRF_WEIGHT_BM25_ZH` | `1.0` | Per-channel RRF bias for the BM25-zh prefetch (raise to ~1.5 for Chinese-only recall) |
| `MAX_CHUNKS_PER_PDF` | unset | Per-PDF chunk cap before text indexing |
| `IMAGE_RELEVANCE_THRESHOLD` | `0.24` | CLIP cosine floor for image routes |

Changing `MAX_CHUNKS_PER_PDF` requires `mmrag reindex` to rebuild existing collections.

## Per-intent RRF weights

`hybrid_search` picks its three-route RRF weight triple based on a fast local classifier on the query (`mm_asset_rag.query_intent.classify_intent`). The four intents — `PRECISE_KEYWORD` / `DESCRIPTIVE` / `ENTITY_LOOKUP` / `CHINESE` — map to weight triples tuned for each query shape:

| Intent | Default (text / t2i / i2i) | When |
| --- | --- | --- |
| `PRECISE_KEYWORD` | `0.60 / 0.20 / 0.15` | Short, no-stopword queries: `联宝 ESG`, `BERT`, `双碳目标` |
| `DESCRIPTIVE` | `0.85 / 0.15 / 0.10` | Long (≥30 chars after stripping) sentences — let dense carry the paraphrase signal |
| `ENTITY_LOOKUP` | `0.75 / 0.25 / 0.10` | Short English / mixed fallback (between keyword and descriptive) — slight image boost so a single entity can surface a logo / photo |
| `CHINESE` | `0.70 / 0.20 / 0.15` | CJK ratio ≥ 70% — biases text so the BM25-zh channel (enabled by default) wins |

Classification rule (first match wins): CJK ratio ≥ 70% → `CHINESE`; stripped length ≤ 12 chars and no Chinese stopword (`的` / `是` / `怎么` / …) → `PRECISE_KEYWORD`; stripped length ≥ 30 chars → `DESCRIPTIVE`; otherwise `ENTITY_LOOKUP`.

Classification is purely local (CJK ratio + length + a small Chinese stopword set), <1ms per query, no LLM. When `HYBRID_INTENT_ROUTING_ENABLED=false` (default), `hybrid_search` ignores the intent and uses the historical global `HYBRID_WEIGHT_*` triple — flip the switch on once you have intent-level eval coverage.

Any `HYBRID_INTENT_WEIGHTS_*` field accepts either a JSON object `{"text":0.7,"text_to_image":0.2,"image_to_image":0.15}` or a CSV triple `0.7,0.2,0.15` (CSV is friendlier in a flat `.env`). Invalid JSON / CSV is logged and falls back to the default — a typo in `.env` doesn't break search. Set the keyword-only `weights_override=` argument on `hybrid_search` (or `retrieval.hybrid_search`) to bypass classification entirely in tests / scripted eval.

## Query rewrite (LLM-driven multi-query RAG)

Layers on top of the legacy `QUERY_LOWERCASE` / `QUERY_FUZZY` / `QUERY_EXPANSION` flags (which still run inside `hybrid_search` per-variant). When `QUERY_REWRITE_ENABLED=true`, every text / hybrid search first asks the LLM (shared `OPENAI_*` / `VLM_*` triple, same resolution as `/answer`) for `QUERY_REWRITE_N_VARIANTS` rewordings of the user's query, then runs each variant through `hybrid_search` in parallel and fuses the hits with rank-based RRF — an asset that surfaces in multiple variants accumulates a higher fused score than one that surfaces in just one. Pattern is Anthropic's multi-query RAG cookbook; the per-variant latency caps at `QUERY_REWRITE_TIMEOUT` because a failed rewrite falls back to the original query and a hanging request is pure waste. Image routes (`text-to-image` / `image-to-image`) bypass this layer — the rewrite only helps the text channels; the CLIP cosine match is invariant to the user's exact wording.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `QUERY_REWRITE_ENABLED` | `false` | Master switch. Off = legacy single-query search (legacy `QUERY_*` flags still apply) |
| `QUERY_REWRITE_N_VARIANTS` | `3` | Number of variants the LLM generates (incl. the original). Clamped to `[1, 5]`. 3 is the sweet spot — more dilutes each variant's RRF contribution |
| `QUERY_REWRITE_TIMEOUT` | `30` | Per-call LLM timeout (seconds). Tighter than `LLM_TIMEOUT=120` because failure → original-query fallback |
| `QUERY_REWRITE_CONCURRENCY` | `4` | Max parallel `hybrid_search` invocations during multi-query fusion (Qdrant-blocking; thread pool). Lower on Qdrant 429s, raise with headroom |

Failure modes are silent fallbacks, not errors: missing creds → `[query]` (single-query search); LLM timeout / HTTP 5xx → `[query]`; bad JSON → `[query]`. Every search request goes through, just without the rewrite lift. See `mm_asset_rag/query_rewrite.py` for the rewrite prompt + JSON-tolerance strategy.

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

## PDF embedded-image extraction

PyMuPDF parses text only by default; embedded figures are dropped. When `PDF_EXTRACT_IMAGES` is on, the parser pulls every image a page references into the version's internal `parsed/<cache_key>/images/` directory and attaches the figures a chunk references (or sits next to) to that chunk's `metadata["images"]`. The figures ride in the text hit's payload — surfaced to the LLM (a `关联图片` hint citing the figure caption) and the web UI (a thumbnail served by `GET /parsed-image/{document_id}/{version_id}/{filename}`). Images are **not** embedded into the vector index (that is tier 2); they are an attachment of the text hit. `PDF_IMAGE_MIN_DIM` filters logos / icons. Requires `mmrag reindex` (or a fresh `mmrag parse`) to populate `images` on existing chunks.

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

Anthropic-style chunk context: each chunk gets a short LLM-generated preamble situating it within its document, prepended to the embedding/BM25 input so dense + sparse channels can disambiguate generic terms. **Enabled by default** — it costs ~1 LLM call per chunk, generated at parse time and cached under the version's internal `parsed/<cache_key>/context.jsonl` path so `mmrag reindex` reuses it without re-calling the LLM. Disable with `CONTEXTUAL_ENABLED=false` (or `mmrag parse --no-contextual` on the CLI) when no LLM is configured or to skip the per-chunk calls.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `CONTEXTUAL_ENABLED` | `true` | Master switch (default on; set `false` to opt out) |
| `CONTEXTUAL_MODEL` | unset (→ `OPENAI_MODEL`) | LLM model override |
| `CONTEXTUAL_CONCURRENCY` | `4` | Parallel chunk-context calls |
| `CONTEXTUAL_CHUNK_MAX_CHARS` | `8000` | Cap chunk text fed to the LLM |
| `CONTEXTUAL_TIMEOUT` | `60` | Per-call HTTP timeout (seconds) |

## Image caption for embedded figures (opt-in)

Document-embedded figures (docx/pptx pictures via markitdown/docling, PDF figures via PyMuPDF) are saved to `parsed/<id>/images/` and associated with chunks, but their *content* is otherwise invisible to the text index — a slide whose only payload is a diagram is unsearchable. When enabled, each embedded figure with no existing caption gets a VLM-generated Chinese description appended to its chunk's text so the figure's semantics enter the dense + BM25 channels. The caption is also recorded in `metadata["images"][*]["caption"]` so the answer layer can cite it.

This is the **text-route** path only: embedded figures are *not* sent to the CLIP image index — that channel stays reserved for standalone image uploads (`source_type=image`). Works with any OpenAI-compatible VLM via `VLM_*`. Cost: ~1 VLM call per embedded figure at parse time. Generated before Contextual Retrieval so the contextual LLM sees caption-enriched chunks. Cached under the version's internal `captions/<cache_key>.jsonl` path so `mmrag reindex` and force re-parse reuse it without re-calling the VLM (figure bytes are stable across re-parses). When `VLM_*` is unconfigured the step degrades to a no-op — safe to leave on.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `IMAGE_CAPTION_ENABLED` | `false` | opt-in master switch |
| `IMAGE_CAPTION_CONCURRENCY` | `4` | Parallel figure-caption VLM calls |

## Two-stage reranker

bge-m3's model card recommends "hybrid retrieval + re-ranking": pull a candidate pool with dense + BM25, then score each `(query, doc)` pair with a cross-encoder. Catches high-score false positives that `MIN_SCORE` cannot. **Enabled by default**.

Two provider backends, selected by `RERANKER_PROVIDER`:

- **`local`** (default) — runs `sentence_transformers.CrossEncoder` in-process. Same dep family as the bge-m3 embedder; no network. Needs the model downloaded (~2GB first run). Disable with `RERANKER_ENABLED=false` when latency / download cost is a concern.
- **`siliconflow` / `dashscope`** — call a hosted rerank API. No local model, no `sentence-transformers` dep; latency is a single network round-trip, predictable for interactive search. The two providers speak **different wire shapes** (see below), handled by the same client.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `RERANKER_ENABLED` | `true` | Master switch (default on; set `false` to opt out) |
| `RERANKER_PROVIDER` | `local` | `local` \| `siliconflow` \| `dashscope` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace cross-encoder model id (local provider) |
| `RERANKER_API_BASE` | provider default | Rerank API URL (HTTP providers). SiliconFlow `https://api.siliconflow.cn/v1/rerank` (flat Cohere form); 百炼 `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank` (DashScope-native nested form, universal host — no workspaceId needed) |
| `RERANKER_API_MODEL` | provider default | Rerank API model (HTTP providers). SiliconFlow `BAAI/bge-reranker-v2-m3`; 百炼 `qwen3-rerank` |
| `RERANKER_API_KEY` | → `OPENAI_API_KEY` | Rerank API key, Bearer auth. Falls back to `OPENAI_API_KEY` when unset (shared key config) |
| `RERANKER_API_TIMEOUT` | `30.0` | HTTP timeout (seconds) for the rerank API call |
| `RERANKER_TOP_N` | `30` | Candidates fetched from each route before rerank (≤ `QDRANT_HYBRID_PREFETCH_LIMIT`) |
| `RERANKER_TOP_K` | unset (→ caller's `top_k`) | Final result count after rerank |
| `RERANKER_HYBRID_BLEND` | `0.6` | Reranker weight in the final blended rank (0 = hybrid only, 1 = pure reranker). Scale-free — works for any provider's score |

**Wire shapes:** SiliconFlow speaks the flat Cohere form (`{model, query, documents, top_n}` → top-level `results[]`). 百炼 speaks the DashScope-native nested form (`{model, input:{query, documents}, parameters:{top_n}}` → `output.results[]`). Both rows use `{index, relevance_score}`; the client handles the shape per-provider so you don't.

**HTTP provider example (硅基流动):**

```bash
RERANKER_ENABLED=true
RERANKER_PROVIDER=siliconflow
RERANKER_API_KEY=sk-xxx           # or reuse OPENAI_API_KEY
```

**百炼 (dashscope)** — uses the universal DashScope-native endpoint, only the key is needed (the OpenAI-compatible flat endpoint would need a per-user workspaceId subdomain and is not used):

```bash
RERANKER_ENABLED=true
RERANKER_PROVIDER=dashscope
RERANKER_API_KEY=sk-xxx           # DASHSCOPE_API_KEY value
# RERANKER_API_MODEL=qwen3-rerank  # default; gte-rerank-v2 / qwen3-vl-rerank also
                                   # work at the same native endpoint — just set this.
```

> Verified live (2026-07): 百炼 `qwen3-rerank` at the native endpoint returns 200 with just an API key, no workspace setup. On the chapter11 eval corpus it matches the local bge-reranker-v2-m3 (v1 hit@5 0.927 / v2 hit@5 0.714 vs local 0.927 / 0.735) — the two models are near-equivalent on this corpus, so the cloud provider trades the local model download for a network round-trip with no precision loss.

> A misconfigured HTTP provider (no key) is detected at startup and treated as **unavailable** — `get_default_reranker` returns `None` and search silently skips the two-stage path, rather than 401ing on every query. A *runtime* API failure (transient 5xx / timeout / connection / non-JSON body) is retried once with a short backoff, then on final failure the search falls back to single-stage hybrid — no crash, with a `WARNING` log naming the provider / URL / status. **Stickiness is provider-declared**: an HTTP provider **soft-stickies** for 60s and auto-recovers after the TTL (a transient cloud outage self-heals without a process restart); the `local` provider **hard-stickies** (a corrupted HF cache / missing dep won't self-heal — needs a `mmrag` restart). A *programming* bug (`TypeError` / `ValueError` / …) propagates rather than being silently swallowed into a sticky-disable, so it surfaces in dev.

> A plain-HTTP **non-loopback** `RERANKER_API_BASE` warns once per process: `Authorization: Bearer <key>` would cross the network in cleartext. Use HTTPS, or keep `http://` on loopback (`127.0.0.1` / `localhost` / `::1`) for a local rerank proxy — the same guard the LLM / VLM / embedding base URLs already use.

## Chinese BM25

| Variable | Default | Purpose |
| --- | ---: | --- |
| `BM25_ZH_ENABLED` | `true` | Enable jieba + Okapi sparse vector |
| `BM25_ZH_K1` | `1.5` | BM25 k1 |
| `BM25_ZH_B` | `0.75` | BM25 b |
| `BM25_ZH_VECTOR_NAME` | `bm25_zh` | Qdrant sparse vector name |

## Evaluation cases

`mmrag eval` (and `POST /eval`) score grouped queries against exact logical document IDs. Each case declares a `query_id`; the top-level qrels mapping supplies graded relevance:

```json
{
  "version": "v1",
  "groups": {"en": [{"query_id": "q1", "query": "..."}]},
  "qrels": {"q1": {"document-id": 3}}
}
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `EVAL_CASES_PATH` | unset | Path to a case JSON overriding the bundled default |

The default (unset) loads the small qrels sample shipped with the package (`mm_asset_rag/eval_data/<version>_cases.json`) — a **text→text-only** template over well-known arxiv papers. It runs once those papers are ingested with the exact document IDs named by the qrels. The CLI `--cases` flag overrides `EVAL_CASES_PATH` for one run; `--cases` and `POST /eval {cases_path}` take the same path.

The file's `version` field is checked (`v1` vs `v2`): loading a v2 file under `mmrag eval` (or vice versa) raises an error instead of silently scoring 0 cases.

Use one or more positive integer qrel grades for a **positive** retrieval case.
Recall, MRR, and MAP treat every positive grade as relevant; NDCG preserves the
grade. Use an explicit empty qrels mapping for a **negative** rejection case:
it is reported separately as empty-result rate and false-retrieval rate and is
not counted as a missed positive retrieval. Every case must have a qrels entry,
including negatives. Reported values apply only to the selected corpus and case
set; they are not a retrieval-quality threshold.

## Answer-quality eval (LLM judge)

`mmrag eval --answer-quality` (or `POST /eval {"answer_quality": true}`) scores the **generated** answer, not just retrieval. It runs three scorers per case:

- **coverage** — `|answer ∩ expected_answer_keywords| / |expected_answer_keywords|`. Cheap substring match (NFC + casefold + ZW-char strip), no LLM.
- **citation precision / recall** — regex extracts `[N]` markers from the answer, looks each up in the top-k evidence, and compares the cited `asset_id`s against `expected_answer_assets` (falls back to `expected_asset_ids` when unset). Adds a `citation_present` flag so reports can split "model never cited" from "model cited wrong".
- **faithfulness** — LLM-as-judge: prompts the LLM with the question + evidence block + the model's answer, asks it to score whether all factual claims are supported. Returns a float 0.0-1.0 + a list of unsupported claims. Skipped (with `faithfulness_skipped=True, faithfulness_error=...`) when no creds are configured, when the judge times out, when the case is over `EVAL_JUDGE_MAX_CASES`, or on any judge exception — one bad case never aborts the run.

| Variable | Default | Purpose |
| --- | --- | --- |
| `EVAL_JUDGE_TIMEOUT` | `30.0` | Per-judge-call timeout (s). Tighter than `LLM_TIMEOUT` (120s) since judge is a single-shot JSON response. |
| `EVAL_JUDGE_MODEL` | unset | Judge model name. Unset → reuse `OPENAI_MODEL` (or `VLM_MODEL` fallback). Override with e.g. `gpt-4o-mini` to save tokens (judge prompt is ~3-5K tokens per case). |
| `EVAL_JUDGE_MAX_CASES` | unset | Cap the number of cases the judge runs per eval (CI cost guard). Unset = judge every case. Over the cap → `faithfulness_skipped=True, faithfulness_error="max cases reached"`. |

Coverage + citation always run (no LLM needed). Faithfulness is the only LLM-dependent scorer; without it the report still surfaces the answer-quality signal — it's just missing the hallucination dimension. The output lives at `$MM_ASSET_RAG_HOME/eval_report_answer.json` (payload version `answer_v1`).

v0 supports **text→text cases only**; image-route cases (`image_path`, `text_to_image`, `image_to_image` groups) raise a clear `ValueError` — `llm_answer(question, hits)` doesn't accept image input today.

## Upload safety limits

These limits protect `/upload/preview` from accidental very large uploads. Oversized multipart bodies return HTTP 413; files that sniff as too large/complex are shown as rejected preview cards and cannot be confirmed.

The `UPLOAD_MAX_BATCH_BYTES` cap is enforced by a request-body-size middleware **before** Starlette spools the multipart body to `SpooledTemporaryFile` — so an oversized POST is rejected at the ASGI layer rather than filling `/tmp` first. The per-file and per-batch checks inside the handler remain as a second layer.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `UPLOAD_MAX_FILE_BYTES` | `52428800` | Per-file upload cap (in-handler check) |
| `UPLOAD_MAX_BATCH_BYTES` | `209715200` | Total request-body cap (enforced by the body-size middleware before spool) |
| `UPLOAD_MAX_FILES` | `50` | Max files in one `/upload/preview` batch (bounds VLM auto-meta spend) |
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

## Image OCR (PP-OCRv6)

Standalone image assets (screenshots, photos, scanned pages) have no extractable text layer, so they would be invisible to the text→text route. When OCR is enabled, in-image text is recognised and entered into the text index. Off by default — turn it on only when the corpus has image assets whose text matters for retrieval.

Two backends, selected by `OCR_BACKEND`:

- **`local`** (default) — runs PP-OCRv6 small in-process via the `[ocr]` extra (rapidocr + onnxruntime, pure ONNX, models ship with the wheel). Self-contained, zero-network.
- **`http`** — keeps the legacy external-OCR-server contract. Point `OCR_HTTP_URL` at your own `/ocr` endpoint that accepts `POST {image_base64, file_name}` → `{blocks:[{text}]}`.

Install the extra to use the `local` backend:

```bash
pip install -e ".[ocr]"
```

`ENABLE_OCR` is the legacy master switch (applies to both backends); `OCR_BACKEND` selects which one.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `ENABLE_OCR` | `false` | Master switch for image OCR |
| `OCR_BACKEND` | `local` | `local` (PP-OCRv6, needs `[ocr]`) or `http` (external service) |
| `OCR_HTTP_URL` | unset | External OCR endpoint (only used when `OCR_BACKEND=http`) |
| `OCR_HTTP_TIMEOUT` | `60.0` | External OCR timeout seconds |

For scanned PDFs (a different path — `PDF_PARSER=auto` with `PDF_SCAN_FALLBACK_PARSER=ppocr`), the local PP-OCRv6 page-OCR runs via the same `[ocr]` extra regardless of `ENABLE_OCR`.

## OCR / VLM backends

These cover the legacy `http` OCR backend and the VLM channels; the default `local` PP-OCRv6 backend is documented in the section above.

| Variable | Default | Purpose |
| --- | ---: | --- |
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
| `ppocr` | PP-OCRv6 (local) | Local page-by-page OCR via the `[ocr]` extra (rapidocr + onnxruntime). Zero-network fallback for scanned PDFs |

`pymupdf` remains a hard dependency; `paddleocr_vl` is online; `docling` needs the `[docling]` extra; `ppocr` needs the `[ocr]` extra. Without the extra, `--pdf-parser docling` / `ppocr` raises a friendly install hint at parse time rather than an `ImportError` at startup.

## Document parser selection

Office / text documents (`docx` / `pptx` / `xlsx` / `html` / `md` / `txt` — the `document` source type `sniff` assigns) are parsed by the backend chosen with `DOCUMENT_PARSER` (CLI `--document-parser`):

| Value | Backend | Notes |
| --- | --- | --- |
| `markitdown` | MarkItDown | Default. Core dependency (pure Python, no ML stack). docx/pptx/xlsx converters ship via the `markitdown[docx,pptx,xlsx]` extra bundled in core |
| `docling` | docling | Optional heavy backend (torch / transformers). Needs the `[docling]` extra. Layout-aware; use when MarkItDown's structural extraction isn't enough |

Both backends produce the same `DocumentIR`, so chunking / image association / contextual enrichment are identical downstream. MarkItDown decodes docx/pptx base64-embedded images to `parsed/<id>/images/` and rewrites the refs, so embedded images attach to their chunk and reach the answer layer — same on-disk layout as the docling / PaddleOCR paths. (HTML relative-path images are passed through as-is in v1; they don't associate but don't error.)

### Scanned-PDF fallback (auto parser)

The `auto` parser runs fast local PyMuPDF first, then falls back to an OCR backend when the result looks like a scan (image-only, near-zero text). `PDF_SCAN_TEXT_THRESHOLD` is the total non-empty chars/page budget below which a document is treated as scanned — corpus-agnostic (pure char density, no domain words): `total_chars < threshold * page_count`. `PDF_SCAN_FALLBACK_PARSER` picks the OCR backend; the default `auto` routes by token availability — `paddleocr_vl` (online API, needs `PADDLEOCR_VL_API_TOKEN`) when a token is set, `ppocr` (local PP-OCRv6 via the `[ocr]` extra, zero-network) when it is not. Set it explicitly to `docling` to force the local docling OCR (needs the `[docling]` extra). Disable the whole fallback with `PDF_SCAN_FALLBACK_ENABLED=false` to always stay on PyMuPDF (the pre-IR `auto` behaviour).

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PDF_SCAN_FALLBACK_ENABLED` | `true` | Master switch for the scanned-PDF fallback in `auto` mode |
| `PDF_SCAN_TEXT_THRESHOLD` | `10` | Avg non-empty chars/page below which a PDF is treated as scanned |
| `PDF_SCAN_FALLBACK_PARSER` | `auto` | Fallback backend: `auto` / `paddleocr_vl` / `docling` / `ppocr` |

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
| `PADDLEOCR_VL_IMAGE_HOSTS` | unset | Comma-separated extra hosts allowed for OCR image downloads (SSRF allow-list extension; default: only the `PADDLEOCR_VL_JOB_URL` host). Private/loopback/link-local IPs are always refused. |

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
