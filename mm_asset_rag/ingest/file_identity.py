"""Shared filename → document-ID identity rules.

The upload pipeline and the evaluation tooling must derive document IDs
from filenames *exactly* the same way, so the slug rules live here in a
single module rather than being re-implemented (and drifting) inline.
"""

from __future__ import annotations

import re
from pathlib import Path

_DANGEROUS_FILENAME_CHARS = re.compile(r"[<>:\"|?*\x00-\x1f]+")
_TRAILING_SUFFIX = re.compile(r"\.[^./\\]+$")

# Asset extensions recognised when stripping a trailing filename suffix.
# Anything else (``.tar``, ``.env``, ...) is left alone so document IDs
# derived from multi-dot or dotfile names stay stable.
_KNOWN_ASSET_SUFFIXES = {
    # images
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg",
    # documents
    ".pdf", ".doc", ".docx", ".txt", ".md",
    # audio
    ".mp3", ".wav", ".flac", ".m4a", ".ogg",
    # video
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
}


def slugify_filename_stem(value: str, *, max_len: int | None = None) -> str:
    """Normalise a filename stem so it's safe to embed in a path.

    Keeps unicode (so Chinese titles stay readable), removes path
    separators and control characters, collapses whitespace.
    """
    cleaned = _strip_known_suffix(value)
    cleaned = re.sub(r"[\\/]+", " ", cleaned).strip()
    cleaned = _DANGEROUS_FILENAME_CHARS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .") or "asset"
    if max_len is not None and max_len > 0 and len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" .") or "asset"
    return cleaned


def _strip_known_suffix(value: str) -> str:
    match = _TRAILING_SUFFIX.search(value)
    if match and match.group(0).lower() in _KNOWN_ASSET_SUFFIXES:
        return value[: match.start()]
    return value


def document_id_from_filename(filename: str | Path, *, max_len: int | None = None) -> str:
    """Derive a stable document ID from a filename, mirroring upload rules.

    Strips the extension in two layers: ``Path.stem`` removes the final
    dotted suffix (any extension), then ``slugify_filename_stem`` removes a
    known asset suffix if one remains (e.g. ``archive.tar.gz`` → stem
    ``archive.tar`` → ``archive.tar``, since ``.tar`` is not an asset
    extension). Both layers are idempotent for plain stems.
    """
    return slugify_filename_stem(Path(filename).stem, max_len=max_len)
