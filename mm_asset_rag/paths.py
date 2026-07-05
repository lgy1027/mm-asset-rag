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


# Suffixes allowed for served/embedded parsed images. Kept tight so a
# crafted ``filename`` cannot exfiltrate arbitrary files. Shared by the
# ``/parsed-image`` HTTP endpoint and the tier-3 multimodal answer path.
PARSED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


def safe_parsed_image_path(asset_id: str, filename: str) -> Path | None:
    """Resolve ``parsed/<asset_id>/images/<filename>`` with traversal guard.

    Returns the absolute :class:`Path` when ``filename`` is a bare base
    name with an image suffix, ``asset_id`` has no path separators, the
    resolved path stays inside the asset's ``images/`` dir, and the file
    exists on disk. Returns ``None`` otherwise — callers (the
    ``/parsed-image`` endpoint and the tier-3 answer image loader) turn
    that into a 404 / a skipped image without raising.

    Centralised here so the endpoint and the answer path apply identical
    validation: a hit's ``images`` list is untrusted payload data that
    must not reach the filesystem unfiltered.
    """
    import re

    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        return None
    if not asset_id or "/" in asset_id or "\\" in asset_id or asset_id in (".", ".."):
        return None
    # Reject any path component that resolves above the images dir, even
    # via unusual encodings — the containment check below is the real
    # backstop, but a bare base name is the only thing we ever accept.
    if re.search(r"[\x00-\x1f]", filename) or re.search(r"[\x00-\x1f]", asset_id):
        return None
    suffix = Path(filename).suffix.lower()
    if suffix not in PARSED_IMAGE_SUFFIXES:
        return None
    base = (get_parsed_dir() / asset_id / "images").resolve()
    candidate = (base / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None
