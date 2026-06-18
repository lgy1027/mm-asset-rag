"""Resolve on-disk paths for mm-asset-rag.

All data lives under a single directory pointed to by ``MM_ASSET_RAG_HOME``
(or ``~/.mm_asset_rag`` if the variable is not set). Layout::

    $MM_ASSET_RAG_HOME/
    ├── assets/                  # user-supplied PDFs / images + asset_manifest.json
    ├── parsed/<asset_id>/       # PDF page-level markdown / image OCR JSON
    ├── captions/<asset_id>.json # VLM captions
    ├── indexes/
    │   ├── text/                # LlamaIndex storage
    │   └── qdrant/              # Qdrant local persistence
    ├── documents.jsonl          # unified ParsedDocument store
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


def get_manifest_path() -> Path:
    return get_assets_dir() / "asset_manifest.json"
