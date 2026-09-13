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
    # Covers every field asserted below; a host with a real .env (e.g. a
    # developer's CLIP_MODEL=OFA-Sys/...) would otherwise leak through
    # ``Settings(_env_file=None)`` (it still reads os.environ).
    for key in (
        "MM_ASSET_RAG_HOME",
        "PDF_PARSER",
        "IMAGE_PROVIDER",
        "ENABLE_OCR",
        "ENABLE_VLM",
        "AUTO_INDEX",
        "QDRANT_UPSERT_BATCH_SIZE",
        "LLM_TIMEOUT",
        "CLIP_MODEL",
        "DOCUMENT_PARSER",
        "MMRAG_API_HOST",
        "MMRAG_API_PORT",
    ):
        monkeypatch.delenv(key, raising=False)

    s = Settings(_env_file=None)  # bypass .env so only defaults apply
    assert s.pdf_parser == "auto"
    assert s.document_parser == "markitdown"
    assert s.enable_ocr is False
    assert s.enable_vlm is False
    assert s.image_provider == "clip"
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
    assert s.mmrag_api_host == "127.0.0.1"
    assert s.mmrag_api_port == 8011


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


def test_api_bind_address_can_be_configured(monkeypatch):
    monkeypatch.setenv("MMRAG_API_HOST", "0.0.0.0")
    monkeypatch.setenv("MMRAG_API_PORT", "18011")

    settings = Settings(_env_file=None)

    assert settings.mmrag_api_host == "0.0.0.0"
    assert settings.mmrag_api_port == 18011


def test_case_insensitive_env(monkeypatch):
    """Pydantic-settings lower-cases env var names by default."""
    monkeypatch.setenv("pdf_parser", "pymupdf")
    s = Settings(_env_file=None)
    assert s.pdf_parser == "pymupdf"


def test_has_llm_requires_full_triple(monkeypatch):
    # ``has_llm`` accepts either ``OPENAI_*`` or ``VLM_*`` (LLM channel
    # falls back to VLM credentials), so the negative case must clear both
    # triples — otherwise a sibling test that set ``VLM_*`` would leak in.
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert Settings(_env_file=None).has_llm is False

    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
    assert Settings(_env_file=None).has_llm is False  # still missing BASE_URL+MODEL
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
    assert Settings(_env_file=None).has_llm is False  # still missing MODEL
    monkeypatch.setenv("LLM_MODEL", "gpt")
    assert Settings(_env_file=None).has_llm is True


def test_text_embedding_uses_common_connection_but_requires_own_model(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k1")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://provider")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-v1")
    s = Settings(_env_file=None)
    assert s.text_embedding_creds == ("k1", "http://provider", "embed-v1")


def test_text_embedding_creds_overrides_take_precedence(monkeypatch):
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
    monkeypatch.setenv("LLM_MODEL", "first")
    a = get_settings()
    b = get_settings()
    assert a is b  # lru_cache

    monkeypatch.setenv("LLM_MODEL", "second")
    # Cache miss only if cleared.
    c = Settings(_env_file=None)
    assert c.llm_model == "second"


def test_get_settings_cache_clear_reflects_new_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "first")
    a = get_settings()
    assert a.llm_model == "first"

    monkeypatch.setenv("LLM_MODEL", "second")
    get_settings.cache_clear()
    b = get_settings()
    assert b.llm_model == "second"


def test_env_bool_coercion(monkeypatch):
    """Each true-y string should map to True; everything else (except 1) to False."""
    monkeypatch.setenv("ENABLE_OCR", "true")
    assert Settings(_env_file=None).enable_ocr is True
    monkeypatch.setenv("ENABLE_OCR", "FALSE")
    assert Settings(_env_file=None).enable_ocr is False
    monkeypatch.setenv("ENABLE_OCR", "yes")
    assert Settings(_env_file=None).enable_ocr is True
