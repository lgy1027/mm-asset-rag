"""Two-phase upload pipeline: preview -> confirm.

This module owns the lifecycle between the moment a user drops a file
into the web UI and the moment it lands in Qdrant. There are two
distinct stages:

* **preview** (``/upload/preview``) — files are streamed into a short-
  lived cache directory (``$MM_ASSET_RAG_HOME/.preview-cache/<id>``),
  sniffed for format, and (optionally) sent to a vision-language model
  to extract ``title`` / ``description`` / ``tags``. The caller gets a
  list of ``AssetPreview`` records with everything they need to render
  editable cards in the UI. Nothing under ``assets/`` is touched yet,
  no Qdrant calls are made.

* **confirm** (``/upload/confirm``) — takes the previews plus the
  user's edits, moves the cached files into their final home under
  ``assets/pdfs/`` or ``assets/images/``, and returns a list of
  ``Asset`` objects ready to hand to the ingest service. The ingest
  step itself (parsing + indexing) still runs asynchronously via
  ``IngestService``; this module's job ends at "files are on disk
  with sensible metadata".

Why two phases? Letting the user correct VLM-hallucinated tags before
the file gets indexed is much cheaper than re-indexing later. It also
gives us a natural place to reject unsupported file types (``source_type
== "unknown"``) before any embedding cost is paid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import asset_index, auto_meta
from .assets import Asset, from_sniffed
from .auto_meta import AutoMeta
from .settings import get_settings
from .sniff import SniffedAsset, sniff

log = logging.getLogger(__name__)

_CACHE_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_DANGEROUS_FILENAME_CHARS = re.compile(r"[<>:\"|?*\x00-\x1f]+")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
SUPPORTED_MANIFEST_VERSIONS = {1}


class UploadManifestError(ValueError):
    """Preview cache manifest is invalid or no longer matches the cache."""


class UploadCommitError(RuntimeError):
    """Moving cached files into assets failed after validation."""


@dataclass(frozen=True)
class _PreparedAsset:
    source_path: Path
    target_path: Path
    asset: Asset
    sha256: str


# ─── Preview DTOs ──────────────────────────────────────────────────────


@dataclass
class UserEdits:
    """What the user changed on a preview card.

    Every field is optional — ``None`` means "keep the auto-extracted
    value". The preview cache id is also returned so the confirm step
    can look the file back up.
    """

    preview_id: str
    title: str | None = None
    tags: list[str] | str | None = None
    description: str | None = None
    rejected: bool = False  # user explicitly skipped this file


@dataclass
class AssetPreview:
    """One preview card sent back to the web UI."""

    preview_id: str
    cache_id: str
    sniff: SniffedAsset
    # Absolute on-disk path the file is currently sitting at. Needed by
    # auto_meta so the VLM call can read the bytes without the manifest.
    source_path: Path = field(default_factory=lambda: Path("."))
    auto_meta: AutoMeta | None = None
    # Convenience fields the UI renders directly without unpacking
    # ``sniff`` / ``auto_meta``. Populated by ``UploadPipeline.preview``.
    effective_title: str = ""
    effective_tags: list[str] = field(default_factory=list)
    effective_description: str = ""
    rejected_reason: str | None = None
    # Content-hash of the staged bytes, computed in ``preview()``. Empty
    # string means we haven't hashed yet (e.g. an unsupported file).
    sha256: str = ""
    # When the content hash matches a non-deleted entry in the asset
    # index, this is the existing ``asset_id``. ``None`` means "no
    # duplicate" — the new upload will get a fresh asset.
    existing_asset_id: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.sniff.source_type in {"pdf", "image"} and self.rejected_reason is None


# ─── Helpers ───────────────────────────────────────────────────────────


def _slugify(value: str, *, max_len: int | None = None) -> str:
    """Normalise a filename stem so it's safe to embed in a path.

    Keeps unicode (so Chinese titles stay readable), removes path
    separators and control characters, collapses whitespace.
    """
    cleaned = re.sub(r"[\\/]+", " ", value).strip()
    cleaned = _DANGEROUS_FILENAME_CHARS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .") or "asset"
    if max_len is not None and max_len > 0 and len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" .") or "asset"
    return cleaned


def _target_subdir(source_type: str) -> str:
    if source_type == "pdf":
        return "pdfs"
    if source_type == "image":
        return "images"
    raise ValueError(f"unsupported source_type: {source_type!r}")


def _parse_tags(raw: str | list[str] | None) -> list[str]:
    """Normalise a tags field.

    Accepts either a list (preferred) or a comma-separated string (for
    the simple ``<input>`` field in the web UI). Returns a deduplicated
    list of trimmed, non-empty strings.
    """
    if raw is None:
        return []
    parts = re.split(r"[,，;；\n]+", raw) if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for p in parts:
        s = str(p).strip()
        if s and s not in out:
            out.append(s)
    return out


def _validate_cache_id(cache_id: str) -> None:
    if not _CACHE_ID_RE.fullmatch(cache_id):
        raise UploadManifestError(f"invalid preview cache id: {cache_id!r}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_cached_file(cache_dir: Path, cached_name: str) -> Path:
    if not cached_name:
        raise UploadManifestError("preview cache entry is missing cached_name")
    rel = Path(cached_name)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 1:
        raise UploadManifestError(f"invalid cached file name: {cached_name!r}")
    cache_root = cache_dir.resolve()
    candidate = (cache_dir / rel).resolve()
    if not _is_relative_to(candidate, cache_root):
        raise UploadManifestError(f"cached file escapes preview cache: {cached_name!r}")
    if not candidate.exists() or not candidate.is_file():
        raise UploadManifestError(f"cached file missing: {cached_name!r}")
    return candidate


def _manifest_tags(raw: object) -> list[str]:
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return _parse_tags(raw)
        return _parse_tags(loaded if isinstance(loaded, (list, str)) else None)
    return _parse_tags(raw if isinstance(raw, list) else None)


def _resource_rejected_reason(sniffed: SniffedAsset) -> str | None:
    settings = get_settings()
    if sniffed.file_size and sniffed.file_size > settings.upload_max_file_bytes:
        return f"file is larger than upload_max_file_bytes ({settings.upload_max_file_bytes})"
    if (
        sniffed.source_type == "pdf"
        and sniffed.page_count is not None
        and sniffed.page_count > settings.upload_max_pdf_pages
    ):
        return f"PDF has too many pages ({sniffed.page_count} > {settings.upload_max_pdf_pages})"
    if sniffed.source_type == "image" and sniffed.width and sniffed.height:
        pixels = sniffed.width * sniffed.height
        if pixels > settings.upload_max_image_pixels:
            return f"image has too many pixels ({pixels} > {settings.upload_max_image_pixels})"
    return None


def _suffix_for(source_path: Path, source_type: str) -> str:
    suffix = source_path.suffix.lower()
    if source_type == "pdf":
        return ".pdf"
    if source_type == "image":
        return suffix if suffix in _IMAGE_SUFFIXES else ".png"
    return suffix or ".bin"


def _unique_target_path(target_dir: Path, asset_id: str, suffix: str, reserved: set[Path]) -> Path:
    target = target_dir / f"{asset_id}{suffix}"
    if target not in reserved and not target.exists():
        reserved.add(target)
        return target
    index = 2
    while True:
        candidate = target_dir / f"{asset_id}_{index}{suffix}"
        if candidate not in reserved and not candidate.exists():
            reserved.add(candidate)
            return candidate
        index += 1


# ─── Pipeline ──────────────────────────────────────────────────────────


class UploadPipeline:
    """Two-phase upload coordinator.

    Constructed with the resolved ``$MM_ASSET_RAG_HOME`` directory.
    Holds no state beyond that — every preview gets a fresh
    ``preview_id`` and the cache is keyed off it.
    """

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.cache_root = self.home / ".preview-cache"
        self.assets_root = self.home / "assets"

    # ── Phase 1 ─────────────────────────────────────────────────────────

    def preview(self, files: Iterable[tuple[str, Path]]) -> list[AssetPreview]:
        """Sniff + VLM each uploaded file. Returns preview cards.

        ``files`` is a sequence of ``(display_name, local_path)`` tuples
        — the local path is where FastAPI streamed the bytes to during
        the multipart parse. Preview copies those bytes into the cache
        so confirm can safely move from cache to assets without mutating
        the caller's temp file.
        """
        cache_id = uuid.uuid4().hex[:12]
        cache_dir = self.cache_root / cache_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, dict[str, object]] = {}

        previews: list[AssetPreview] = []
        for display_name, path in files:
            path = Path(path)
            cached_path = cache_dir / Path(display_name).name
            if cached_path.exists():
                digest = hashlib.md5(str(path).encode()).hexdigest()[:6]
                cached_path = cache_dir / f"{cached_path.stem}_{digest}{cached_path.suffix}"
            shutil.copy2(path, cached_path)

            sniffed = sniff(cached_path)
            preview_id = uuid.uuid4().hex[:12]
            rejected_reason = None
            if sniffed.source_type == "unknown":
                rejected_reason = sniffed.error or "unsupported file"
            else:
                rejected_reason = _resource_rejected_reason(sniffed)
            sha256 = self._sha256_file(cached_path) if rejected_reason is None else ""
            existing = (
                asset_index.find_by_sha256(sha256) if sha256 and rejected_reason is None else None
            )
            preview = AssetPreview(
                preview_id=preview_id,
                cache_id=cache_id,
                sniff=sniffed,
                source_path=cached_path,
                effective_title=sniffed.title,
                rejected_reason=rejected_reason,
                sha256=sha256,
                existing_asset_id=existing.asset_id if existing else None,
            )
            previews.append(preview)
            manifest[preview_id] = {
                "display_name": display_name,
                "cached_name": cached_path.name,
                "source_type": sniffed.source_type,
                "sha256": sha256,
                "existing_asset_id": preview.existing_asset_id,
            }

        # Persist the manifest so confirm() can look files back up.
        manifest["__meta__"] = {"created_at": time.time(), "version": 1}
        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # VLM calls are independent. Each is wrapped to swallow network
        # errors so a single bad image can't kill the whole batch.
        self._fill_auto_meta(previews)

        for preview in previews:
            if preview.auto_meta is not None:
                if preview.auto_meta.title:
                    preview.effective_title = preview.auto_meta.title
                if preview.auto_meta.tags:
                    preview.effective_tags = list(preview.auto_meta.tags)
                if preview.auto_meta.description:
                    preview.effective_description = preview.auto_meta.description

            entry = manifest[preview.preview_id]
            entry["effective_title"] = preview.effective_title
            entry["effective_tags"] = preview.effective_tags
            entry["effective_description"] = preview.effective_description

        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return previews

    def _fill_auto_meta(self, previews: list[AssetPreview]) -> None:
        """Call auto-meta helpers with bounded concurrency."""
        candidates = [
            preview
            for preview in previews
            if preview.rejected_reason is None and preview.sniff.source_type in {"pdf", "image"}
        ]
        if not candidates:
            return
        max_workers = max(1, int(get_settings().auto_meta_max_concurrency))
        if max_workers == 1 or len(candidates) == 1:
            for preview in candidates:
                preview.auto_meta = self._auto_meta_for_preview(preview)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._auto_meta_for_preview, preview): preview
                for preview in candidates
            }
            for future in as_completed(futures):
                preview = futures[future]
                try:
                    preview.auto_meta = future.result()
                except Exception as exc:
                    log.warning("auto_meta failed for %s: %s", preview.preview_id, exc)
                    preview.auto_meta = None

    def _auto_meta_for_preview(self, preview: AssetPreview) -> AutoMeta | None:
        try:
            if preview.sniff.source_type == "pdf":
                return auto_meta.auto_meta_pdf_first_page(preview.source_path)
            if preview.sniff.source_type == "image":
                return auto_meta.auto_meta_image(preview.source_path)
        except Exception as exc:
            log.warning("auto_meta failed for %s: %s", preview.preview_id, exc)
        return None

    # ── Phase 2 ─────────────────────────────────────────────────────────

    def confirm(
        self,
        cache_id: str,
        edits: list[UserEdits],
    ) -> list[Asset]:
        """Move cached files into ``assets/{pdfs,images}/`` and build Assets.

        Files marked ``rejected=True`` are deleted with the cache and
        skipped. Files whose ``preview_id`` isn't in the cache raise
        ``KeyError`` so the API layer can surface a 400.
        """
        _validate_cache_id(cache_id)
        cache_dir = self.cache_root / cache_id
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"unknown preview cache id: {cache_id}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UploadManifestError(f"invalid preview manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise UploadManifestError("invalid preview manifest: expected object")
        self._validate_manifest_version(manifest, source=cache_id)

        edits_by_id = {e.preview_id: e for e in edits}
        unknown_edits = sorted(set(edits_by_id) - set(manifest))
        if unknown_edits:
            raise KeyError(f"unknown preview id: {unknown_edits[0]}")

        prepared = self._prepare_confirm(cache_dir, manifest, edits_by_id)
        moved: list[tuple[Path, Path]] = []
        try:
            for item in prepared:
                shutil.move(str(item.source_path), str(item.target_path))
                moved.append((item.source_path, item.target_path))
        except Exception as exc:
            self._rollback_moves(moved)
            raise UploadCommitError(f"failed to move uploaded file into assets: {exc}") from exc

        for item in prepared:
            try:
                asset_index.upsert_entry(
                    asset_index.AssetIndexEntry(
                        asset_id=item.asset.asset_id,
                        sha256=item.sha256,
                        source_type=item.asset.source_type,
                        relative_path=item.asset.relative_path,
                        asset_title=item.asset.title,
                        tags=list(item.asset.tags),
                    )
                )
            except OSError as exc:
                log.warning("asset_index upsert failed for %s: %s", item.asset.asset_id, exc)

        # Preserve the cache when the user rejected every preview: the
        # web UI's 400 response tells them to "re-edit" and we want
        # the same ``cache_id`` to remain valid for the next confirm.
        # An empty ``prepared`` for a *non-empty* manifest means every
        # preview was rejected or unsupported — never delete in that
        # case. A genuinely empty manifest (corrupt cache) is still
        # safe to clean up since nothing useful was there.
        if not prepared and any(pid != "__meta__" for pid in manifest):
            log.info(
                "confirm: all previews rejected or unsupported; keeping cache %s",
                cache_dir,
            )
            return []
        shutil.rmtree(cache_dir, ignore_errors=True)
        return [item.asset for item in prepared]

    def _prepare_confirm(
        self,
        cache_dir: Path,
        manifest: dict[str, object],
        edits_by_id: dict[str, UserEdits],
    ) -> list[_PreparedAsset]:
        settings = get_settings()
        self.assets_root.mkdir(parents=True, exist_ok=True)
        reserved: set[Path] = set()
        prepared: list[_PreparedAsset] = []

        for preview_id, raw_entry in manifest.items():
            if preview_id == "__meta__":
                continue
            if not isinstance(preview_id, str) or not isinstance(raw_entry, dict):
                raise UploadManifestError("invalid preview manifest entry")
            cached_name = raw_entry.get("cached_name")
            if not isinstance(cached_name, str):
                raise UploadManifestError(f"preview {preview_id} is missing cached_name")
            source_path = _resolve_cached_file(cache_dir, cached_name)

            edit = edits_by_id.get(preview_id)
            if edit is not None and edit.rejected:
                continue

            sniffed = sniff(source_path)
            if sniffed.source_type not in {"pdf", "image"}:
                continue
            resource_reason = _resource_rejected_reason(sniffed)
            if resource_reason is not None:
                log.warning("confirm: rejecting %s: %s", source_path.name, resource_reason)
                continue

            subdir = _target_subdir(sniffed.source_type)
            target_dir = self.assets_root / subdir
            target_dir.mkdir(parents=True, exist_ok=True)

            effective_title = str(raw_entry.get("effective_title") or sniffed.title)
            effective_tags = _manifest_tags(raw_entry.get("effective_tags"))
            display_title = (
                edit.title if edit and edit.title else effective_title
            ) or sniffed.title
            safe_stem = _slugify(display_title, max_len=settings.upload_slug_max_len)
            base_asset_id = f"{safe_stem}_{preview_id[:8]}"
            suffix = _suffix_for(source_path, sniffed.source_type)
            content_sha = self._sha256_file(source_path)
            existing = asset_index.find_by_sha256(content_sha)
            if existing is not None:
                target = self.assets_root / existing.relative_path
                asset_id = existing.asset_id
                relative_path = existing.relative_path
            else:
                target = _unique_target_path(target_dir, base_asset_id, suffix, reserved)
                asset_id = target.stem
                relative_path = str(target.relative_to(self.assets_root))
            normalized_edit = None
            if edit is not None:
                normalized_edit = UserEdits(
                    preview_id=edit.preview_id,
                    title=edit.title,
                    tags=_parse_tags(edit.tags) if edit.tags is not None else None,
                    description=edit.description,
                    rejected=edit.rejected,
                )
            asset = from_sniffed(
                sniffed,
                relative_path,
                asset_dir=self.assets_root,
                user_edits=normalized_edit,
                auto_title=effective_title,
                auto_tags=effective_tags,
                asset_id_override=asset_id,
                title_override=display_title,
            )
            prepared.append(
                _PreparedAsset(
                    source_path=source_path,
                    target_path=target,
                    asset=asset,
                    sha256=content_sha,
                )
            )

        return prepared

    def _rollback_moves(self, moved: list[tuple[Path, Path]]) -> None:
        for source_path, target_path in reversed(moved):
            try:
                if target_path.exists():
                    if not source_path.exists():
                        shutil.move(str(target_path), str(source_path))
                    else:
                        target_path.unlink()
            except OSError as exc:
                log.warning("rollback failed for %s -> %s: %s", target_path, source_path, exc)

    def cleanup_expired_caches(self, now: float | None = None) -> int:
        """Delete preview caches older than ``preview_cache_ttl_seconds``.

        Only cache directories with the generated 12-hex id are eligible;
        incoming upload scratch dirs and unexpected names are left alone.
        """
        ttl = get_settings().preview_cache_ttl_seconds
        if ttl <= 0 or not self.cache_root.exists():
            return 0
        cutoff = (time.time() if now is None else now) - ttl
        removed = 0
        for child in self.cache_root.iterdir():
            if not child.is_dir() or not _CACHE_ID_RE.fullmatch(child.name):
                continue
            marker = child / "manifest.json"
            try:
                mtime = marker.stat().st_mtime if marker.exists() else child.stat().st_mtime
            except OSError as exc:
                log.warning("preview cache stat failed for %s: %s", child, exc)
                continue
            created_at: float | None = None
            unknown_version = False
            if marker.exists():
                try:
                    meta_obj = json.loads(marker.read_text(encoding="utf-8"))
                    if isinstance(meta_obj, dict):
                        meta = meta_obj.get("__meta__")
                        if isinstance(meta, dict):
                            if isinstance(meta.get("created_at"), (int, float)):
                                created_at = float(meta["created_at"])
                            version = meta.get("version")
                            if "version" in meta and (
                                not isinstance(version, int)
                                or version not in SUPPORTED_MANIFEST_VERSIONS
                            ):
                                unknown_version = True
                except (OSError, json.JSONDecodeError):
                    created_at = None
            if unknown_version:
                log.warning(
                    "preview cache %s has unsupported manifest version; leaving for explicit discard",
                    child,
                )
                continue
            effective_mtime = created_at if created_at is not None else mtime
            if effective_mtime >= cutoff:
                continue
            try:
                shutil.rmtree(child)
                removed += 1
            except OSError as exc:
                log.warning("preview cache cleanup failed for %s: %s", child, exc)
        return removed

    def _sha256_file(self, path: Path) -> str:
        """Stream a file through SHA-256 so we can dedup on content."""
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_manifest_version(
        self, manifest: dict[str, object], *, source: str | Path
    ) -> None:
        """Reject preview caches whose ``__meta__.version`` is unsupported.

        Manifests without ``__meta__`` entirely are tolerated (legacy
        caches from before version metadata existed) and are left alone
        by ``cleanup_expired_caches``. Manifests that explicitly carry
        ``__meta__`` but a ``version`` we don't recognise must be
        rejected so the user can re-upload rather than silently corrupt
        the asset directory.
        """
        meta = manifest.get("__meta__") if isinstance(manifest, dict) else None
        if meta is None:
            return
        if not isinstance(meta, dict):
            raise UploadManifestError(
                f"preview cache {source} has malformed __meta__: expected object"
            )
        version = meta.get("version")
        if not isinstance(version, int) or version not in SUPPORTED_MANIFEST_VERSIONS:
            raise UploadManifestError(
                f"preview cache {source} has unsupported manifest version: {version!r}"
            )

    def discard_cache(self, cache_id: str) -> None:
        """Best-effort cleanup for a preview the user abandoned."""
        if not _CACHE_ID_RE.fullmatch(cache_id):
            return
        shutil.rmtree(self.cache_root / cache_id, ignore_errors=True)


# ─── Module-level helpers ──────────────────────────────────────────────


def get_pipeline() -> UploadPipeline:
    """Default pipeline rooted at ``$MM_ASSET_RAG_HOME``.

    Used by the FastAPI handlers so the test suite can swap the home
    directory via ``monkeypatch.setenv`` without touching call sites.
    """
    from .settings import get_settings

    return UploadPipeline(get_settings().data_dir)
