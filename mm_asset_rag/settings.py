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
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_batch_size: int = 5
    embedding_request_interval: float = 0.25
    embedding_retry_count: int = 5
    embedding_timeout: float = 120.0
    embedding_max_input_chars: int = 8192

    # ─── Image embedding (CLIP, optional) ────────────────────────────────
    clip_model: str = "clip-ViT-B-32"

    # ─── Qdrant ──────────────────────────────────────────────────────────
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_text_collection: str = "multimodal_text"
    qdrant_image_collection: str = "multimodal_image"
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
    pdf_parser: Literal["auto", "pymupdf", "paddleocr_vl"] = "auto"
    enable_ocr: bool = False
    enable_vlm: bool = False
    image_provider: Literal["lite", "sentence_transformers"] = "lite"
    auto_index: bool = True

    # ─── OCR / VLM HTTP backends (optional) ──────────────────────────────
    ocr_http_url: str | None = None
    ocr_http_timeout: float = 60.0

    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_model: str | None = None
    vlm_temperature: float = 0.1
    vlm_timeout: float = 120.0

    # ─── Derived properties ───────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        """Resolve ``$MM_ASSET_RAG_HOME`` or fall back to ``~/.mm_asset_rag``."""
        if self.mm_asset_rag_home:
            return Path(self.mm_asset_rag_home).expanduser()
        return Path.home() / ".mm_asset_rag"

    @property
    def has_llm(self) -> bool:
        """Whether the LLM triple is complete enough to issue real requests."""
        return bool(self.openai_api_key and self.openai_base_url and self.openai_model)

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
