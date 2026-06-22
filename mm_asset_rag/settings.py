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
        return bool(
            self.openai_api_key and self.openai_base_url and self.openai_model
        )

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