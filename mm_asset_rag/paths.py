"""Resolve on-disk paths for mm-asset-rag.

All data lives under a single directory pointed to by ``MM_ASSET_RAG_HOME``
(or ``~/.mm_asset_rag`` if the variable is not set). Layout::

    $MM_ASSET_RAG_HOME/
    ├── assets/                  # user-supplied PDFs / images (auto-sniffed)
    │   ├── pdfs/                # PDFs uploaded via /upload/confirm
    │   └── images/              # images uploaded via /upload/confirm
    ├── .preview-cache/<id>/     # short-lived cache for /upload/preview
    ├── parsed/<asset_id>/       # PDF page-level markdown / image OCR JSON
    ├── captions/<asset_id>.json # VLM captions
    ├── indexes/
    │   ├── text/                # LlamaIndex storage
    │   └── qdrant/              # Qdrant local persistence
    ├── documents.jsonl          # unified ParsedDocument store
    ├── tasks.jsonl              # background task history
    ├── asset_index.jsonl        # content-hash → asset_id index (append-only)
    └── eval_report.json
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "MM_ASSET_RAG_HOME"


def get_data_dir() -> Path:
    home = os.environ.get(_ENV_VAR)
    path = Path(home).expanduser() if home else Path.home() / ".mm_asset_rag"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_assets_dir() -> Path:
    return get_data_dir() / "assets"


def get_pdf_assets_dir() -> Path:
    """Subdirectory for PDFs (``assets/pdfs/``)."""
    path = get_assets_dir() / "pdfs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_assets_dir() -> Path:
    """Subdirectory for images (``assets/images/``)."""
    path = get_assets_dir() / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_preview_cache_dir() -> Path:
    """Short-lived cache directory for the upload pipeline preview phase.

    Created on first access; cleaned up by ``UploadPipeline.confirm`` /
    ``UploadPipeline.discard_cache``.
    """
    path = get_data_dir() / ".preview-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_parsed_dir() -> Path:
    return get_data_dir() / "parsed"


def get_captions_dir() -> Path:
    path = get_data_dir() / "captions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_indexes_dir() -> Path:
    return get_data_dir() / "indexes"


def get_text_index_dir() -> Path:
    return get_indexes_dir() / "text"


def get_qdrant_path() -> Path:
    return get_indexes_dir() / "qdrant"


def get_documents_jsonl() -> Path:
    return get_data_dir() / "documents.jsonl"


def get_eval_report() -> Path:
    return get_data_dir() / "eval_report.json"


def get_asset_index_path() -> Path:
    """Append-only JSONL index that maps content SHA256 to ``Asset`` metadata.

    See :mod:`mm_asset_rag.asset_index` for the row schema. The file is
    created lazily on first write, not on read.
    """
    return get_data_dir() / "asset_index.jsonl"
