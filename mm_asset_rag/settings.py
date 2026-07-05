"""Centralized settings for ``mm-asset-rag``.

Every environment variable the codebase reads is declared here as a typed
``Settings`` field. The module-level :func:`get_settings` returns a
cached singleton so any module can ``from .settings import get_settings; s
= get_settings()`` and get a single source of truth.

Why this exists: previously 30+ ``os.environ.get(...)`` calls were
scattered across ``providers.py``, ``image_parser.py``, ``pdf_parser.py``,
``answer.py``, ``api.py``, ``qdrant_store.py``, ``config.py``, and
``paths.py``. That made it easy to add a new variable in one place and
miss another, hard to validate types, and painful to mock in tests.

This module is the single read site for environment variables. Existing
modules can keep their lazy ``os.environ.get`` calls during a transition
period — but new code should call ``get_settings().foo``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime-tunable knobs in one place."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Paths ───────────────────────────────────────────────────────────
    mm_asset_rag_home: Path | None = None

    # ─── LLM (OpenAI-compatible chat completion) ─────────────────────────
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None
    llm_timeout: float = 120.0

    # ─── Text embedding ───────────────────────────────────────────────────
    # Backend: ``openai`` (OpenAI-compatible /v1/embeddings) or
    # ``sentence_transformers`` (local HF model). For multilingual /
    # cross-language corpora, ``sentence_transformers`` with
    # ``BAAI/bge-m3`` or ``intfloat/multilingual-e5-large`` is much
    # stronger than the OpenAI default on ZH↔EN retrieval.
    embedding_backend: Literal["openai", "sentence_transformers"] = "openai"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_batch_size: int = 5
    embedding_request_interval: float = 0.25
    embedding_retry_count: int = 5
    embedding_timeout: float = 120.0
    embedding_max_input_chars: int = 8192

    # ─── Image embedding (CLIP, optional) ────────────────────────────────
    # Default is ``clip-ViT-B-32`` (English-only). For Chinese corpora,
    # consider ``OFA-Sys/chinese-clip-vit-base-patch16`` (≈ 768d,
    # Chinese + English) or ``sentence-transformers/clip-ViT-B-32-multilingual-v1``.
    # ``OFA-Sys/chinese-clip-vit-huge-patch14`` is the strongest
    # Chinese CLIP we are aware of (~1024d) at the cost of a much larger
    # download. Reindex after changing this — the active collection
    # name is dim-suffixed.
    clip_model: str = "clip-ViT-B-32"

    # ─── Qdrant ──────────────────────────────────────────────────────────
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_text_collection: str = "multimodal_text"
    qdrant_image_collection: str = "multimodal_image"
    # Optional override for the *active* collection name; falls back to
    # the base name from ``qdrant_text_collection`` / ``qdrant_image_collection``
    # suffixed with the embedding dim. Set to e.g. ``multimodal_text_2560d``
    # to force a specific collection (useful when migrating between
    # embedding models without rebuilding from scratch).
    qdrant_active_text_collection: str | None = None
    qdrant_active_image_collection: str | None = None
    qdrant_upsert_batch_size: int = 16
    qdrant_bm25_model: str = "Qdrant/bm25"
    qdrant_hybrid_prefetch_limit: int = 20

    # ─── Retrieval tuning ────────────────────────────────────────────────
    # Weights used by ``retrieval.hybrid_search`` to merge the three
    # routes (text / text-to-image / image-to-image). The list passed to
    # ``merge_hits`` is built dynamically from whichever routes actually
    # participate — ``image-to-image`` is only included when an
    # ``image_path`` is supplied. Defaults tightened from the historical
    # ``0.55 / 0.30 / 0.15`` because, on the bundled sample set, the
    # ``text-to-image`` route was dragging unrelated images into pure
    # text queries.
    hybrid_weight_text: float = 0.80
    hybrid_weight_text_to_image: float = 0.20
    hybrid_weight_image_to_image: float = 0.0
    # Per-asset chunk cap applied during ``build_qdrant_text_index``.
    # Without a cap, dense embeddings skew toward the largest PDFs
    # (clip / flamingo / gpt3 contribute 48 / 54 / 75 chunks each on the
    # bundled set) and crowd smaller, more relevant assets out of the
    # top-k. ``None`` keeps the current behaviour.
    max_chunks_per_pdf: int | None = None
    # Cosine similarity floor for the image search routes. CLIP scores
    # live in roughly 0.15-0.40; off-topic natural-language queries
    # (e.g. "Schrödinger equation" against a photo collection) tend to
    # land below 0.24 even for the closest image, while on-topic
    # queries like "Linux logo" sit at 0.30+. Filtering below this
    # threshold gives the image routes a relevance floor so negative
    # queries return an empty list instead of ten random Picsum photos.
    # Set to ``0.0`` to disable the floor. Note: this is a *partial*
    # fix — when a Picsum photo is genuinely a close CLIP match
    # (e.g. real mountain photos in response to "Mount Everest"), the
    # threshold cannot tell apart "true negative" from "relevant but
    # unlabeled"; a sparse / keyword pre-filter is the next upgrade.
    image_relevance_threshold: float = 0.24
    # Sparse pre-filter for image search: a token-overlap check on
    # payload fields that you control. When the user query shares
    # *zero* tokens with any image's indexed payload fields, the
    # image route returns empty without even calling Qdrant. This
    # catches the cases where dense-only top-k always returns
    # random Picsum-style noise for off-topic queries.
    #
    # Defaults: ``["tags", "asset_id", "asset_title"]`` match the
    # payload fields produced by the upload pipeline; the image payload
    # stores these verbatim. Override if your pipeline uses different field names.
    # Set to ``[]`` to disable the pre-filter entirely.
    image_prefilter_fields: list[str] = ["tags", "asset_id", "asset_title"]
    image_prefilter_min_token_len: int = 3
    # Confidence floor for the merged hybrid result. After ``merge_hits``
    # weights and normalises per-route scores, any result whose weighted
    # score falls below ``min_score`` is dropped. ``0.0`` keeps every
    # result. The default ``0.30`` was tuned on the bundled corpus
    # (v3 eval): positive recall is preserved while 6/8 negative
    # queries return an empty list. ``0.20`` is the recommended lower
    # bound for sparse corpora; raise to ``0.40-0.60`` for very
    # noisy open-domain RAG.
    min_score: float = 0.30

    # ─── Two-stage reranker ───────────────────────────────────────────────
    # bge-m3's model card recommends "hybrid retrieval + re-ranking": pull a
    # candidate pool with dense + BM25, then score each (query, doc) pair
    # with a cross-encoder. Catches high-score false positives that the
    # global ``min_score`` floor cannot (v6b: 1.20 only dropped 1/8
    # negatives while losing 4 positives). opt-in — disabled by default
    # because it adds ~50-200ms latency per query and the model is ~600MB
    # on first download. Runs locally via ``sentence_transformers.CrossEncoder``
    # (same dep as the bge-m3 embedder); no ollama / API. When enabled,
    # ``hybrid_search`` fetches ``reranker_top_n`` candidates, reranks, and
    # returns ``reranker_top_k`` (or the caller's top_k if None).
    # ``reranker_top_n`` should be ≤ ``qdrant_hybrid_prefetch_limit`` (default
    # 20) or the candidate pool is bounded by the prefetch.
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 20
    reranker_top_k: int | None = None

    # ─── Chinese BM25 ─────────────────────────────────────────────────────
    # Companion sparse vector produced by ``mm_asset_rag.bm25_zh``
    # (jieba tokenisation + Okapi BM25). Stored alongside the existing
    # English fastembed BM25 in the same Qdrant collection, then fused
    # via RRF at query time. The hybrid-text query prefetches both
    # sparse vectors so token recall for Chinese is no longer reliant
    # on the dense channel alone.
    bm25_zh_enabled: bool = True
    bm25_zh_k1: float = 1.5
    bm25_zh_b: float = 0.75
    bm25_zh_vector_name: str = "bm25_zh"

    # ─── Chunk enrichment ─────────────────────────────────────────────────
    # When ``ENRICH_CHUNK_WITH_KEYWORDS`` is true (default), the PDF /
    # image parser appends a "关键词: ..." line to each chunk's text
    # before indexing. The keywords come from
    # ``mm_asset_rag.text_keywords.extract_keywords_zh`` (jieba
    # TextRank) which gives the BM25 channel explicit tokens to match
    # short user queries like "联宝 ESG" against a long PDF body
    # where the tokens would otherwise be diluted. Disable for
    # non-Chinese corpora or when jieba is unavailable.
    enrich_chunk_with_keywords: bool = True
    enrich_chunk_keyword_top_k: int = 8
    # Language hint passed to ``extract_keywords``. The parser uses
    # this to pick the right extractor. ``auto`` runs jieba first
    # (Chinese) and falls back to the stopword-frequency extractor
    # (English) when jieba returns nothing — recommended for mixed
    # corpora.
    enrich_chunk_language: Literal["zh", "en", "auto"] = "auto"

    # ─── PDF embedded-image extraction (tier-1 multimodal) ───────────────
    # PyMuPDF parses text only by default; embedded figures are dropped.
    # When ``pdf_extract_images`` is on, ``pdf_images.extract_page_images``
    # pulls every image a page references into ``parsed/<id>/images/`` and
    # ``associate_images`` attaches the figures a chunk references (or sits
    # next to) to ``ParsedDocument.metadata["images"]`` — surfaced to the
    # LLM (as a 关联图片 hint) and the web UI (as a thumbnail). Images are
    # NOT embedded into the vector index (that is tier 2); they ride in the
    # text hit's payload. ``pdf_image_min_dim`` filters logos/icons.
    pdf_extract_images: bool = True
    pdf_image_min_dim: int = 80

    # ─── Tier-3 multimodal answer (opt-in) ───────────────────────────────
    # When on, ``answer.llm_answer`` / ``stream_answer_chunks`` inject the
    # hit's associated images (base64 data URLs) into the chat request as
    # ``image_url`` content parts alongside the text evidence, so a
    # multimodal LLM can *see* figure pixels and answer "图里 2025 年的
    # 数字是多少" questions that the text alone cannot satisfy. Requires a
    # vision-capable chat model (e.g. MiniMax-M3, or ollama gemma3 / llama3.2
    # -vision). If the configured model rejects images, the call is retried
    # text-only so the feature is safe to toggle without breaking /answer.
    # ``answer_image_max_per_hit`` caps images per hit to bound token cost.
    answer_with_images: bool = False
    answer_image_max_per_hit: int = 2

    # ─── Query preprocessing ──────────────────────────────────────────────
    # The hybrid text search runs each query through three normalisations
    # before routing to dense vs BM25 channels. See
    # ``mm_asset_rag.query_preprocess.preprocess`` for the per-stage
    # contract. Defaults are conservative — only the typo corrector is
    # safe to leave on for all corpora.
    query_lowercase: bool = True
    query_fuzzy: bool = True
    query_expansion: bool = False
    query_expansion_pairs: str | None = None  # path to a JSON file

    # ─── Per-channel RRF weights ──────────────────────────────────────────
    # Inside ``_hybrid_text_query`` the three prefetches (dense / BM25-en /
    # BM25-zh) are fused by Qdrant's ``RrfQuery(rrf=Rrf(weights=[...]))``
    # (qdrant-client 1.18+, server 1.17+). The three weights below
    # let the deployer bias the fusion positionally: ``[dense, bm25,
    # bm25_zh]``. Raising ``rrf_weight_bm25_zh`` improves Chinese-only
    # token recall; lowering it makes the dense channel dominant for
    # cross-language queries. The default 1.0/1.0/1.0 matches Qdrant's
    # uniform-fusion default (``FusionQuery(fusion=Fusion.RRF)``).
    rrf_weight_dense: float = 1.0
    rrf_weight_bm25: float = 1.0
    rrf_weight_bm25_zh: float = 1.0

    # ─── PaddleOCR-VL ────────────────────────────────────────────────────
    paddleocr_vl_api_token: str | None = None
    paddleocr_vl_job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    paddleocr_vl_model: str = "PaddleOCR-VL-1.6"
    paddleocr_vl_timeout: float = 900.0
    paddleocr_vl_poll_interval: float = 5.0
    paddleocr_vl_poll_retry: int = 5
    paddleocr_vl_use_doc_orientation_classify: bool = False
    paddleocr_vl_use_doc_unwarping: bool = False
    paddleocr_vl_use_chart_recognition: bool = False

    # ─── Parser defaults (drives /upload; UI can override per request) ───
    # NOTE: pdf_parser / enable_ocr / enable_vlm / image_provider / auto_index
    # are kept as legacy fields for backward compat with old deployments, but
    # the modern upload pipeline auto-decides everything from sniff + VLM.
    pdf_parser: Literal["auto", "pymupdf", "paddleocr_vl"] = "auto"
    enable_ocr: bool = False
    enable_vlm: bool = False
    image_provider: Literal["lite", "sentence_transformers"] = "lite"
    auto_index: bool = True

    # ─── Upload preview safety limits ─────────────────────────────────────
    upload_max_file_bytes: int = 50 * 1024 * 1024
    upload_max_batch_bytes: int = 200 * 1024 * 1024
    upload_max_pdf_pages: int = 500
    upload_max_image_pixels: int = 50_000_000
    upload_slug_max_len: int = 80
    preview_cache_ttl_seconds: int = 24 * 60 * 60

    # ─── OCR / VLM HTTP backends (optional) ──────────────────────────────
    ocr_http_url: str | None = None
    ocr_http_timeout: float = 60.0

    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_model: str | None = None
    vlm_temperature: float = 0.1
    vlm_max_tokens: int = 2000
    vlm_timeout: float = 120.0

    # ─── Auto-extracted metadata (VLM-driven) ────────────────────────────
    # When enabled, the upload pipeline calls the VLM during the preview
    # phase to extract title / description / tags / dominant_objects in one
    # round trip. Disable on deployments where VLM cost is a concern or when
    # the model is unreliable for the corpus.
    auto_meta_enabled: bool = True
    auto_meta_timeout: float = 30.0
    auto_meta_max_tokens: int = 800
    auto_meta_max_concurrency: int = 3
    auto_meta_image_prompt: str | None = None
    auto_meta_pdf_prompt: str | None = None
    auto_meta_pdf_max_pages: int = 100
    auto_meta_pdf_render_dpi: int = 120
    auto_meta_pdf_max_render_pixels: int = 8_000_000

    # ─── Contextual Retrieval ─────────────────────────────────────────────
    # Anthropic-style chunk context (2024, -49% retrieval failure rate):
    # each chunk gets a short LLM-generated preamble situating it within its
    # document, prepended to the embedding/BM25 input so dense + sparse
    # channels can disambiguate generic terms ("diffusion" → DDPM vs Stable
    # Diffusion). opt-in — disabled by default because it costs ~1 LLM call
    # per chunk (4158 PDF chunks on the bundled corpus ≈ 9.4M tokens).
    # The shared ``OPENAI_*`` triple is reused; ``contextual_model`` only
    # overrides the model name. Generated at parse time and cached under
    # ``parsed/<id>/context.jsonl`` so ``mmrag reindex`` reuses it without
    # re-calling the LLM. Enable via ``mmrag parse --contextual``.
    contextual_enabled: bool = False
    contextual_model: str | None = None
    contextual_concurrency: int = 4
    contextual_chunk_max_chars: int = 8000
    contextual_timeout: float = 60.0

    # ─── Derived properties ───────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        """Resolve ``$MM_ASSET_RAG_HOME`` or fall back to ``~/.mm_asset_rag``."""
        if self.mm_asset_rag_home:
            return Path(self.mm_asset_rag_home).expanduser()
        return Path.home() / ".mm_asset_rag"

    @property
    def has_llm(self) -> bool:
        """Whether the LLM triple is complete enough to issue real requests.

        True when *either* the ``OPENAI_*`` triple is set *or* the ``VLM_*``
        triple is set (the LLM channel falls back to VLM credentials — see
        :attr:`llm_creds`). This lets a deployment configure a single
        multimodal model under ``VLM_*`` and have it serve both the image
        caption / auto-meta path *and* the ``/answer`` / ``/chat`` text path.
        """
        return bool(
            (self.openai_api_key and self.openai_base_url and self.openai_model)
            or (self.vlm_api_key and self.vlm_base_url and self.vlm_model)
        )

    @property
    def llm_creds(self) -> tuple[str | None, str | None, str | None]:
        """Return ``(base_url, api_key, model)`` for the chat LLM channel.

        ``OPENAI_*`` is preferred (it is the canonical chat triple); when
        any of the three is missing the ``VLM_*`` triple is used as
        fallback so a deployment that only configured a multimodal VLM
        still gets a working ``/answer`` / ``/chat``. Returns ``(None,
        None, None)`` when neither triple is complete — callers then fall
        back to the evidence-summary path.
        """
        if self.openai_api_key and self.openai_base_url and self.openai_model:
            return self.openai_base_url, self.openai_api_key, self.openai_model
        if self.vlm_api_key and self.vlm_base_url and self.vlm_model:
            return self.vlm_base_url, self.vlm_api_key, self.vlm_model
        return None, None, None

    @property
    def vlm_creds(self) -> tuple[str | None, str | None, str | None]:
        """Return ``(base_url, api_key, model)`` for the VLM channel.

        ``VLM_*`` is preferred; falls back to ``OPENAI_*`` when unset
        (preserves the long-standing "configure once under OPENAI_*"
        convenience for image caption / auto-meta).
        """
        if self.vlm_api_key and self.vlm_base_url and self.vlm_model:
            return self.vlm_base_url, self.vlm_api_key, self.vlm_model
        if self.openai_api_key and self.openai_base_url and self.openai_model:
            return self.openai_base_url, self.openai_api_key, self.openai_model
        return None, None, None

    @property
    def text_embedding_creds(self) -> tuple[str | None, str | None, str | None]:
        """Return ``(api_key, base_url, model)`` falling back to OPENAI_* for creds."""
        return (
            self.embedding_api_key or self.openai_api_key,
            self.embedding_base_url or self.openai_base_url,
            self.embedding_model,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    ``lru_cache`` is used so repeated calls are cheap and a single
    ``Settings()`` is constructed per process. Tests that need to
    override environment should call ``get_settings.cache_clear()``
    and then ``Settings()`` again (or use ``monkeypatch.setenv`` and a
    fresh ``get_settings()``).
    """
    return Settings()
