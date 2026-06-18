# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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