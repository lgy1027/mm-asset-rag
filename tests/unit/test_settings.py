"""Tests for ``mm_asset_rag.settings``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Each test gets a fresh Settings singleton."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_applied_when_no_env(monkeypatch):
    # Strip any .env-inherited values so defaults are tested in isolation.
    for key in (
        "MM_ASSET_RAG_HOME",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "PDF_PARSER",
        "IMAGE_PROVIDER",
        "ENABLE_OCR",
        "ENABLE_VLM",
        "AUTO_INDEX",
        "QDRANT_UPSERT_BATCH_SIZE",
        "LLM_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)

    s = Settings(_env_file=None)  # bypass .env so only defaults apply
    assert s.pdf_parser == "auto"
    assert s.enable_ocr is False
    assert s.enable_vlm is False
    assert s.image_provider == "lite"
    assert s.auto_index is True
    assert s.qdrant_upsert_batch_size == 16
    assert s.llm_timeout == 120.0
    assert s.clip_model == "clip-ViT-B-32"
    assert s.upload_max_file_bytes == 50 * 1024 * 1024
    assert s.upload_max_batch_bytes == 200 * 1024 * 1024
    assert s.upload_max_pdf_pages == 500
    assert s.upload_max_image_pixels == 50_000_000
    assert s.upload_slug_max_len == 80
    assert s.auto_meta_pdf_max_pages == 100
    assert s.auto_meta_pdf_render_dpi == 120
    assert s.auto_meta_pdf_max_render_pixels == 8_000_000
    assert s.preview_cache_ttl_seconds == 24 * 60 * 60
    assert s.auto_meta_max_concurrency == 3


def test_data_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("MM_ASSET_RAG_HOME", raising=False)
    s = Settings(_env_file=None)
    assert s.data_dir == Path.home() / ".mm_asset_rag"


def test_data_dir_uses_mm_asset_rag_home(monkeypatch):
    monkeypatch.setenv("MM_ASSET_RAG_HOME", "/tmp/custom-home")
    s = Settings(_env_file=None)
    assert s.data_dir == Path("/tmp/custom-home")


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.delenv("PDF_PARSER", raising=False)
    monkeypatch.setenv("PDF_PARSER", "paddleocr_vl")
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setenv("ENABLE_VLM", "1")
    monkeypatch.setenv("QDRANT_UPSERT_BATCH_SIZE", "64")
    s = Settings(_env_file=None)
    assert s.pdf_parser == "paddleocr_vl"
    assert s.enable_ocr is True
    assert s.enable_vlm is True
    assert s.qdrant_upsert_batch_size == 64


def test_case_insensitive_env(monkeypatch):
    """Pydantic-settings lower-cases env var names by default."""
    monkeypatch.setenv("pdf_parser", "pymupdf")
    s = Settings(_env_file=None)
    assert s.pdf_parser == "pymupdf"


def test_has_llm_requires_full_triple(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert Settings(_env_file=None).has_llm is False

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert Settings(_env_file=None).has_llm is False  # still missing BASE_URL+MODEL
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    assert Settings(_env_file=None).has_llm is False  # still missing MODEL
    monkeypatch.setenv("OPENAI_MODEL", "gpt")
    assert Settings(_env_file=None).has_llm is True


def test_text_embedding_creds_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embed-3-small")
    s = Settings(_env_file=None)
    api_key, base_url, model = s.text_embedding_creds
    assert api_key == "k1"
    assert base_url == "http://llm"
    assert model == "text-embed-3-small"


def test_text_embedding_creds_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embed")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-model")
    s = Settings(_env_file=None)
    api_key, base_url, model = s.text_embedding_creds
    assert api_key == "embed-key"
    assert base_url == "http://embed"
    assert model == "custom-model"


def test_pdf_parser_validates_choice(monkeypatch):
    """Invalid PDF_PARSER values are rejected by Pydantic at construction."""
    monkeypatch.setenv("PDF_PARSER", "totally-bogus")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_get_settings_returns_singleton(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "first")
    a = get_settings()
    b = get_settings()
    assert a is b  # lru_cache

    monkeypatch.setenv("OPENAI_MODEL", "second")
    # Cache miss only if cleared.
    c = Settings(_env_file=None)
    assert c.openai_model == "second"


def test_get_settings_cache_clear_reflects_new_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "first")
    a = get_settings()
    assert a.openai_model == "first"

    monkeypatch.setenv("OPENAI_MODEL", "second")
    get_settings.cache_clear()
    b = get_settings()
    assert b.openai_model == "second"


def test_env_bool_coercion(monkeypatch):
    """Each true-y string should map to True; everything else (except 1) to False."""
    monkeypatch.setenv("ENABLE_OCR", "true")
    assert Settings(_env_file=None).enable_ocr is True
    monkeypatch.setenv("ENABLE_OCR", "FALSE")
    assert Settings(_env_file=None).enable_ocr is False
    monkeypatch.setenv("ENABLE_OCR", "yes")
    assert Settings(_env_file=None).enable_ocr is True
