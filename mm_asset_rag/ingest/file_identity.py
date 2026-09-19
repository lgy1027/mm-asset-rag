"""Shared filename → document-ID identity rules.

The upload pipeline and the evaluation tooling must derive document IDs
from filenames *exactly* the same way, so the slug rules live here in a
single module rather than being re-implemented (and drifting) inline.
"""

from __future__ import annotations

import re
from pathlib import Path

_DANGEROUS_FILENAME_CHARS = re.compile(r"[<>:\"|?*\x00-\x1f]+")
# Trailing dotted suffix on a raw filename ("poster.jpg"). Callers normally
# pass an already-stemmed name; stripping here keeps direct calls on full
# filenames consistent with the upload pipeline.
_FILENAME_SUFFIX = re.compile(r"\.[^./\\]*$")


def slugify_filename_stem(value: str, *, max_len: int | None = None) -> str:
    """Normalise a filename stem so it's safe to embed in a path.

    Keeps unicode (so Chinese titles stay readable), removes path
    separators and control characters, collapses whitespace.
    """
    cleaned = _FILENAME_SUFFIX.sub("", value)
    cleaned = re.sub(r"[\\/]+", " ", cleaned).strip()
    cleaned = _DANGEROUS_FILENAME_CHARS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .") or "asset"
    if max_len is not None and max_len > 0 and len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" .") or "asset"
    return cleaned


def document_id_from_filename(filename: str | Path, *, max_len: int | None = None) -> str:
    """Derive a stable document ID from a filename, mirroring upload rules."""
    return slugify_filename_stem(Path(filename).stem, max_len=max_len)
