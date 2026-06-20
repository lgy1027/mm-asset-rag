# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- Mock embedding provider and `EMBEDDING_PROVIDER=mock` path. Every embedding call now goes to a real OpenAI-compatible backend; missing configuration raises `EmbeddingConfigError` instead of silently producing hash-based fake vectors.
- Graceful fallback in `EmbeddingProvider` and `configure_embedding` for missing API keys.
- Bundled mini fixture (`tests/fixtures/`) with a synthetic 1-page PDF and 32x32 PNG. Tests now use the real `examples/data/chapter11_assets/` sample set (10 PDFs + 20 images) by copying into a per-test tmp directory.
- `MOCK_EMBEDDING_DIM` environment variable.

### Changed
- Tests in `test_providers_embedding.py` now exercise the real OpenAI provider via `responses` HTTP mocking (vs. the previous `_mock_embedding` SHA-256 path).
- Test count: 56 → 53 (dropped two mock-only assertions; added one batch-size test).

## [0.1.0] - 2026-XX-XX

### Added
- Initial extraction from `llamaindex-tutorial` project.
- Multimodal asset RAG pipeline: PDF parsing (PyMuPDF + PaddleOCR-VL), image OCR/VLM caption, text + image embedding, hybrid retrieval, LLM-grounded answering.
- Two vector backends: Qdrant (default, local or server) and LlamaIndex (lightweight fallback).
- Retrieval modes: text, text-to-image, image-to-image, hybrid.
- CLI (`mmrag`) and FastAPI server (`mmrag-api`).
- Offline-friendly defaults: every external service has a graceful fallback.

### Changed
- Removed the self-implemented lite-image-feature backend; only Qdrant and LlamaIndex remain.
- Image embedding provider is now exclusively `sentence-transformers` (optional `[clip]` extra).

### Fixed
- `requirements.txt` originally missed `llama-index-embeddings-openai`; corrected in `pyproject.toml`.