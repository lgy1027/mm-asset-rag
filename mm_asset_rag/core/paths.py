"""Resolve on-disk paths for mm-asset-rag.

All data lives under a single directory pointed to by ``MM_ASSET_RAG_HOME``
(or ``~/.mm_asset_rag`` if the variable is not set). Layout::

    $MM_ASSET_RAG_HOME/
    ├── assets/                  # user-supplied files (auto-sniffed on confirm)
    │   ├── pdfs/                # PDFs uploaded via /upload/confirm
    │   ├── images/              # images uploaded via /upload/confirm
    │   └── documents/           # office/text (docx/pptx/xlsx/html/md)
    ├── .preview-cache/<id>/     # short-lived cache for /upload/preview
    ├── parsed/<cache_id>/       # PDF page-level markdown / image OCR JSON
    ├── captions/<cache_id>.jsonl  # VLM captions (image asset: .json single-object)
    ├── indexes/
    │   └── qdrant/              # Qdrant local persistence
    ├── documents.jsonl          # unified ParsedChunk store
    ├── tasks.db                 # background task history (SQLite)
    ├── asset_index.jsonl        # content-hash → asset_id index (append-only)
    └── eval_report.json
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

_ENV_VAR = "MM_ASSET_RAG_HOME"


def physical_cache_id(relative_path: str) -> str:
    """Return a stable filesystem-safe identity for the complete asset path."""
    path = PurePosixPath(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise ValueError("relative_path must be a safe relative POSIX path")
    return hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()


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


def get_asr_dir() -> Path:
    """Transcript cache for audio/video ASR (``asr/``).

    Lives outside ``parsed/<id>/`` (which force re-parse clears) so
    re-parsing reuses the transcript without re-running ASR; the cache
    entries key on source mtime/size to invalidate stale media.
    """
    path = get_data_dir() / "asr"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_indexes_dir() -> Path:
    return get_data_dir() / "indexes"


def get_qdrant_path() -> Path:
    return get_indexes_dir() / "qdrant"


def get_documents_jsonl() -> Path:
    return get_data_dir() / "documents.jsonl"


def _report_slug(collection: str | None) -> str:
    """Filesystem-safe slug for collection-scoped report names."""
    slug = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(collection or ""))
    return slug[:48]


def get_eval_report(collection: str | None = None) -> Path:
    """Eval report path, scoped per collection when one is known.

    A single global ``eval_report.json`` meant an eval run on one
    knowledge base silently clobbered the previous collection's report.
    ``None`` / ``"default"`` keeps the legacy filename; named collections
    get their own ``eval_report_<slug>.json``.
    """
    base = get_data_dir() / "eval_report"
    if not collection or collection == "default":
        return base.with_suffix(".json")
    return base.with_name(f"eval_report_{_report_slug(collection)}.json")


def get_answer_eval_report(collection: str | None = None) -> Path:
    """Output path for ``mmrag eval --answer-quality`` (coverage / citation
    / LLM-judge faithfulness scores). Sits next to ``eval_report.json`` /
    ``eval_report_v2.json`` and is ``answer_v1``-versioned in the payload
    so dashboards can tell reports apart after a schema bump.
    """
    return get_eval_report(collection).with_name(
        f"eval_report_answer{_report_slug(collection) and '_' + _report_slug(collection) or ''}.json"
    )


def get_eval_cases_dir() -> Path:
    """Directory for user-supplied eval case JSONs accepted by ``cases_path``.

    ``/eval`` and ``--cases`` restrict user-supplied case files to this dir
    (or the repo ``examples/`` dir) so an untrusted ``cases_path`` can't read
    arbitrary files. Created lazily on first access.
    """
    path = get_data_dir() / "eval_cases"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_cases_path(value: str | Path | None) -> Path | None:
    """Resolve a user-supplied ``cases_path`` against the allowed dirs.

    ``None`` → ``None`` (the caller uses the bundled default /
    ``EVAL_CASES_PATH``, resolved server-side and trusted). A non-None path
    is validated (relative, ``.json`` suffix, no ``..`` / absolute
    traversal), then resolved under ``$MM_ASSET_RAG_HOME/eval_cases/``
    first and the repo ``examples/`` dir second. The resolved path must
    exist on disk and stay inside one of those dirs — "lands inside an
    allowed dir" is not enough; the file has to be there so a typo'd path
    raises rather than silently falling through.

    Shared by the API (``POST /eval``) and CLI (``--cases``) so the two
    surfaces can't drift. Raises :class:`ValueError` on a malformed path
    and :class:`FileNotFoundError` when the file isn't in either dir;
    callers translate those to 422 / ``SystemExit``.

    Note: the repo ``examples/`` dir only exists for source / editable
    installs (it's excluded from the wheel); a wheel install's user should
    put case files in ``eval_cases/`` or use the bundled default.
    """
    if value is None:
        return None
    raw = Path(str(value)).expanduser()
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"cases_path must be a relative path inside eval_cases/: {value!r}")
    suffix = raw.suffix.lower()
    if suffix and suffix != ".json":
        raise ValueError(f"cases_path must point at a .json file: {value!r}")

    candidate = raw
    # The README / docs use the ``examples/<name>`` form (e.g.
    # ``--cases examples/eval_cases_chapter11_v1.json``). ``repo_examples``
    # *is* the ``examples/`` dir, so joining it with ``examples/<name>``
    # would double the prefix (``examples/examples/<name>``) and miss.
    # Strip a leading ``examples/`` segment for the examples/ root only;
    # eval_cases/ keeps the raw candidate so a file literally named that
    # under eval_cases/ still wins (and a bare ``<name>`` works on both).
    parts = candidate.parts
    candidate_examples = (
        Path(*parts[1:]) if len(parts) > 1 and parts[0] == "examples" else candidate
    )

    def _inside(existing_base: Path, cand: Path) -> Path | None:
        base = existing_base.resolve()
        resolved = (base / cand).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            return None
        return resolved if resolved.is_file() else None

    # eval_cases/ under the data dir — the user's own case file location.
    found = _inside(get_eval_cases_dir(), candidate)
    if found is not None:
        return found
    # Repo examples/ — chapter11 baselines ship there (source installs only).
    repo_examples = Path(__file__).resolve().parents[2] / "examples"
    found = _inside(repo_examples, candidate_examples)
    if found is not None:
        return found
    raise FileNotFoundError(f"cases_path not found in eval_cases/ or examples/: {value!r}")


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
