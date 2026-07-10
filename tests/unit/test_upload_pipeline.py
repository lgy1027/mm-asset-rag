"""Tests for ``mm_asset_rag.upload_pipeline``.

Pipeline is exercised end-to-end against a tmp ``$MM_ASSET_RAG_HOME``:
real PNG / PDF fixtures get snipped, fake VLM responses are injected
via ``monkeypatch``, then ``confirm`` moves the files into ``assets/``
and yields ``Asset`` objects the ingest service can pick up.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from mm_asset_rag import auto_meta
from mm_asset_rag.upload_pipeline import (
    UploadManifestError,
    UploadPipeline,
    UserEdits,
    _parse_tags,
    _slugify,
)

# ─── fixtures & helpers ────────────────────────────────────────────────


@pytest.fixture()
def png_file(tmp_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = tmp_path / "beach.png"
    Image.new("RGB", (10, 10), color=(0, 0, 255)).save(p)
    return p


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    p = tmp_path / "stable_diffusion.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(p))
    doc.close()
    return p


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


@pytest.fixture()
def pipeline(home: Path) -> UploadPipeline:
    return UploadPipeline(home)


def _stub_vlm_image(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    monkeypatch.setattr(
        auto_meta,
        "auto_meta_image",
        lambda path: auto_meta.AutoMeta(
            title=payload.get("title"),
            description=payload.get("description"),
            tags=payload.get("tags", []),
            dominant_objects=payload.get("dominant_objects", []),
        ),
    )


def _stub_vlm_pdf(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    monkeypatch.setattr(
        auto_meta,
        "auto_meta_pdf_first_page",
        lambda path: auto_meta.AutoMeta(
            title=payload.get("title"),
            description=payload.get("description"),
            tags=payload.get("tags", []),
            page_summary=payload.get("page_summary"),
        ),
    )


# ─── _parse_tags ───────────────────────────────────────────────────────


def test_parse_tags_from_string_comma() -> None:
    assert _parse_tags("beach, sunset, ocean") == ["beach", "sunset", "ocean"]


def test_parse_tags_from_string_chinese_sep() -> None:
    assert _parse_tags("海,滩;日落\n海洋") == ["海", "滩", "日落", "海洋"]


def test_parse_tags_from_list_dedup() -> None:
    assert _parse_tags(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_parse_tags_empty() -> None:
    assert _parse_tags("") == []
    assert _parse_tags(None) == []


# ─── _slugify ──────────────────────────────────────────────────────────


def test_slugify_collapses_separators() -> None:
    assert _slugify("foo/bar\\baz qux") == "foo bar baz qux"


def test_slugify_empty_falls_back() -> None:
    assert _slugify("   /  \\  ") == "asset"


# ─── preview ───────────────────────────────────────────────────────────


def test_preview_image_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    png_file: Path,
) -> None:
    _stub_vlm_image(
        monkeypatch,
        {
            "title": "Beach Sunset",
            "description": "A sunny beach at sunset.",
            "tags": ["beach", "sunset"],
            "dominant_objects": ["ocean"],
        },
    )
    previews = pipeline.preview([(png_file.name, png_file)])
    assert len(previews) == 1
    p = previews[0]
    assert p.sniff.source_type == "image"
    assert p.is_supported
    assert p.effective_title == "Beach Sunset"
    assert p.effective_tags == ["beach", "sunset"]
    assert p.rejected_reason is None


def test_preview_pdf_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    pdf_file: Path,
) -> None:
    _stub_vlm_pdf(
        monkeypatch,
        {
            "title": "Stable Diffusion",
            "description": "Latent diffusion paper.",
            "tags": ["diffusion", "image-generation"],
            "page_summary": "Abstract",
        },
    )
    previews = pipeline.preview([(pdf_file.name, pdf_file)])
    assert len(previews) == 1
    p = previews[0]
    assert p.sniff.source_type == "pdf"
    assert p.effective_title == "Stable Diffusion"
    assert "diffusion" in p.effective_tags


def test_preview_unknown_file_rejected(pipeline: UploadPipeline, home: Path) -> None:
    bogus = home / "garbage.bin"
    bogus.write_bytes(b"not really anything")
    previews = pipeline.preview([("garbage.bin", bogus)])
    assert previews[0].sniff.source_type == "unknown"
    assert not previews[0].is_supported
    assert previews[0].rejected_reason is not None


def test_preview_vlm_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    png_file: Path,
) -> None:
    monkeypatch.setattr(auto_meta, "auto_meta_image", lambda path: None)
    previews = pipeline.preview([(png_file.name, png_file)])
    assert previews[0].auto_meta is None
    # Falls back to sniff-derived title.
    assert previews[0].effective_title == "Beach"
    assert previews[0].effective_tags == []


def test_preview_creates_cache_manifest(
    pipeline: UploadPipeline, home: Path, png_file: Path
) -> None:
    pipeline.preview([(png_file.name, png_file)])
    # Preview doesn't expose the cache id, but the manifest file should exist.
    cache_dirs = list((home / ".preview-cache").iterdir())
    assert len(cache_dirs) == 1
    assert (cache_dirs[0] / "manifest.json").exists()


# ─── confirm ───────────────────────────────────────────────────────────


def _get_cache_id(home: Path) -> str:
    return next(iter((home / ".preview-cache").iterdir())).name


def test_confirm_moves_files_into_assets(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
    png_file: Path,
) -> None:
    _stub_vlm_image(monkeypatch, {"title": "Beach"})
    pipeline.preview([(png_file.name, png_file)])
    cache_id = _get_cache_id(home)

    edits = []
    # Re-read the manifest to find the preview_id we generated.
    import json as _json

    manifest = _json.loads((home / ".preview-cache" / cache_id / "manifest.json").read_text())
    preview_id = next(iter(manifest))
    edits.append(UserEdits(preview_id=preview_id))

    assets = pipeline.confirm(cache_id, edits)
    assert len(assets) == 1
    a = assets[0]
    assert a.source_type == "image"
    assert a.title == "Beach"
    # File now lives under assets/images/
    target = home / "assets" / "images" / a.relative_path.split("/")[-1]
    assert target.exists()
    # Preview copies into cache, so the original temp/source file is left intact.
    assert png_file.exists()
    # Cache directory removed.
    assert not (home / ".preview-cache" / cache_id).exists()


def test_confirm_moves_document_into_assets_documents(
    pipeline: UploadPipeline,
    home: Path,
    tmp_path: Path,
) -> None:
    """A docx (source_type=document) is supported end-to-end: preview
    marks it supported, confirm routes it to ``assets/documents/`` with
    the original extension preserved (so the docling adapter can pick
    its backend by extension)."""
    import zipfile

    docx = tmp_path / "report.docx"
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")

    previews = pipeline.preview([(docx.name, docx)])
    assert len(previews) == 1
    p = previews[0]
    assert p.sniff.source_type == "document"
    assert p.is_supported  # document is now a first-class supported type

    cache_id = _get_cache_id(home)
    import json as _json

    manifest = _json.loads((home / ".preview-cache" / cache_id / "manifest.json").read_text())
    preview_id = next(iter(manifest))
    assets = pipeline.confirm(cache_id, [UserEdits(preview_id=preview_id)])

    assert len(assets) == 1
    a = assets[0]
    assert a.source_type == "document"
    # Routed under assets/documents/ with the .docx extension preserved.
    target = home / "assets" / "documents" / a.relative_path.split("/")[-1]
    assert target.exists()
    assert target.suffix == ".docx"


def test_confirm_with_user_edits(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
    png_file: Path,
) -> None:
    _stub_vlm_image(monkeypatch, {"title": "Auto Title", "tags": ["auto"]})
    pipeline.preview([(png_file.name, png_file)])
    cache_id = _get_cache_id(home)

    import json as _json

    manifest = _json.loads((home / ".preview-cache" / cache_id / "manifest.json").read_text())
    preview_id = next(iter(manifest))
    edits = [
        UserEdits(
            preview_id=preview_id,
            title="My Edited Title",
            tags="custom, manual, tags",
        )
    ]
    assets = pipeline.confirm(cache_id, edits)
    a = assets[0]
    assert a.title == "My Edited Title"
    assert a.tags == ["custom", "manual", "tags"]


def test_confirm_rejected_skipped(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
    png_file: Path,
) -> None:
    _stub_vlm_image(monkeypatch, {"title": "X"})
    pipeline.preview([(png_file.name, png_file)])
    cache_id = _get_cache_id(home)

    import json as _json

    manifest = _json.loads((home / ".preview-cache" / cache_id / "manifest.json").read_text())
    preview_id = next(iter(manifest))
    edits = [UserEdits(preview_id=preview_id, rejected=True)]
    assets = pipeline.confirm(cache_id, edits)
    assert assets == []
    # Rejected preview deletes only the cached copy; original stays untouched.
    assert png_file.exists()


def test_confirm_unknown_cache_id_raises(pipeline: UploadPipeline) -> None:
    with pytest.raises(UploadManifestError, match="invalid preview cache id"):
        pipeline.confirm("does_not_exist", [])


def test_confirm_missing_valid_cache_id_raises(pipeline: UploadPipeline) -> None:
    with pytest.raises(KeyError, match="unknown preview cache id"):
        pipeline.confirm("0" * 12, [])


def test_confirm_dedupes_repeat_upload(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
    png_file: Path,
) -> None:
    """Two uploads of byte-identical files resolve to the same asset."""
    _stub_vlm_image(monkeypatch, {"title": "Beach"})
    pipeline.preview([(png_file.name, png_file)])
    cache_id_1 = _get_cache_id(home)
    import json as _json

    preview_id_1 = next(
        iter(
            e
            for e in _json.loads(
                (home / ".preview-cache" / cache_id_1 / "manifest.json").read_text()
            )
            if e != "__meta__"
        )
    )
    assets_1 = pipeline.confirm(cache_id_1, [UserEdits(preview_id=preview_id_1)])
    first_relative = assets_1[0].relative_path
    first_asset_id = assets_1[0].asset_id

    pipeline.preview([(png_file.name, png_file)])
    cache_id_2 = _get_cache_id(home)
    preview_id_2 = next(
        e
        for e in _json.loads((home / ".preview-cache" / cache_id_2 / "manifest.json").read_text())
        if e != "__meta__"
    )
    assets_2 = pipeline.confirm(cache_id_2, [UserEdits(preview_id=preview_id_2)])
    # Content-hash dedup: same asset_id and relative_path.
    assert assets_2[0].asset_id == first_asset_id
    assert assets_2[0].relative_path == first_relative


def test_confirm_collision_with_different_content(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
    png_file: Path,
) -> None:
    """Different bytes produce different relative_paths even with the same title."""
    from PIL import Image

    other = home / "different.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(other)
    _stub_vlm_image(monkeypatch, {"title": "Beach"})
    pipeline.preview([(png_file.name, png_file)])
    cache_id_1 = _get_cache_id(home)
    import json as _json

    preview_id_1 = next(
        e
        for e in _json.loads((home / ".preview-cache" / cache_id_1 / "manifest.json").read_text())
        if e != "__meta__"
    )
    assets_1 = pipeline.confirm(cache_id_1, [UserEdits(preview_id=preview_id_1)])

    pipeline.preview([(other.name, other)])
    cache_id_2 = _get_cache_id(home)
    preview_id_2 = next(
        e
        for e in _json.loads((home / ".preview-cache" / cache_id_2 / "manifest.json").read_text())
        if e != "__meta__"
    )
    assets_2 = pipeline.confirm(cache_id_2, [UserEdits(preview_id=preview_id_2)])
    assert assets_2[0].relative_path != assets_1[0].relative_path


# ─── discard_cache ────────────────────────────────────────────────────


def test_discard_cache_removes_directory(
    pipeline: UploadPipeline, home: Path, png_file: Path
) -> None:
    pipeline.preview([(png_file.name, png_file)])
    cache_id = _get_cache_id(home)
    assert (home / ".preview-cache" / cache_id).exists()
    pipeline.discard_cache(cache_id)
    assert not (home / ".preview-cache" / cache_id).exists()


def test_discard_cache_unknown_is_noop(pipeline: UploadPipeline) -> None:
    pipeline.discard_cache("never_existed")  # should not raise


# ─── cleanup_expired_caches ──────────────────────────────────────────


def test_cleanup_removes_expired_caches(
    pipeline: UploadPipeline, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PREVIEW_CACHE_TTL_SECONDS", "60")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()

    pipeline.preview([("a.png", _write_png(home, "a.png"))])
    pipeline.preview([("b.png", _write_png(home, "b.png"))])
    caches = sorted((home / ".preview-cache").iterdir())
    assert len(caches) == 2

    # Mark first cache as very old via manifest.json mtime.
    old_cache = caches[0]
    manifest = old_cache / "manifest.json"
    manifest.write_text("{}")
    import os

    os.utime(manifest, (1_000_000, 1_000_000))

    removed = pipeline.cleanup_expired_caches(now=2_000_000)
    assert removed == 1
    remaining = sorted((home / ".preview-cache").iterdir())
    assert remaining == [caches[1]]


def test_cleanup_skips_unmatched_dirs(pipeline: UploadPipeline, home: Path) -> None:
    cache_dir = home / ".preview-cache"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "incoming_aaaa").mkdir()
    (cache_dir / "garbage").mkdir()
    (cache_dir / "not-a-file.txt").write_text("noop")

    assert pipeline.cleanup_expired_caches(now=10_000_000) == 0
    assert (cache_dir / "incoming_aaaa").exists()
    assert (cache_dir / "garbage").exists()


def test_preview_manifest_has_created_at(pipeline: UploadPipeline, png_file: Path) -> None:
    import json as _json

    pipeline.preview([(png_file.name, png_file)])
    cache_id = next(iter((pipeline.cache_root).iterdir())).name
    manifest = _json.loads((pipeline.cache_root / cache_id / "manifest.json").read_text())
    assert "__meta__" in manifest
    assert "created_at" in manifest["__meta__"]


def test_vlm_rewrite_preserves_created_at(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    png_file: Path,
) -> None:
    import json as _json
    import time as _time

    _stub_vlm_image(monkeypatch, {"title": "Beach"})
    pipeline.preview([(png_file.name, png_file)])
    cache_id = next(iter((pipeline.cache_root).iterdir())).name
    manifest_path = pipeline.cache_root / cache_id / "manifest.json"
    first = _json.loads(manifest_path.read_text())
    created = first["__meta__"]["created_at"]
    _time.sleep(0.05)
    pipeline.preview([(png_file.name, png_file)])
    second = _json.loads(manifest_path.read_text())
    assert second["__meta__"]["created_at"] == created


def test_confirm_rejects_meta_without_version(
    pipeline: UploadPipeline, home: Path, png_file: Path
) -> None:
    import json as _json

    pipeline.preview([(png_file.name, png_file)])
    cache_id = next(iter((pipeline.cache_root).iterdir())).name
    manifest = _json.loads((pipeline.cache_root / cache_id / "manifest.json").read_text())
    manifest["__meta__"] = {"created_at": 1.0}  # no version field
    (pipeline.cache_root / cache_id / "manifest.json").write_text(_json.dumps(manifest))
    with pytest.raises(UploadManifestError, match="unsupported manifest version"):
        pipeline.confirm(cache_id, [])


def test_confirm_rejects_unknown_manifest_version(
    pipeline: UploadPipeline, home: Path, png_file: Path
) -> None:
    import json as _json

    pipeline.preview([(png_file.name, png_file)])
    cache_id = next(iter((pipeline.cache_root).iterdir())).name
    manifest = _json.loads((pipeline.cache_root / cache_id / "manifest.json").read_text())
    manifest["__meta__"] = {"created_at": 1.0, "version": 99}
    (pipeline.cache_root / cache_id / "manifest.json").write_text(_json.dumps(manifest))
    with pytest.raises(UploadManifestError, match="unsupported manifest version"):
        pipeline.confirm(cache_id, [])


def test_cleanup_skips_unversioned_cache(
    pipeline: UploadPipeline, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json

    monkeypatch.setenv("PREVIEW_CACHE_TTL_SECONDS", "60")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()

    cache_dir = home / ".preview-cache"
    cache_dir.mkdir(exist_ok=True)
    bad = cache_dir / "abc123456789"
    bad.mkdir()
    (bad / "manifest.json").write_text(
        _json.dumps({"__meta__": {"version": 99}, "preview_id": {"cached_name": "x"}})
    )
    import os as _os

    _os.utime(bad / "manifest.json", (1, 1))
    removed = pipeline.cleanup_expired_caches(now=2_000_000)
    assert removed == 0
    assert bad.exists()


def test_preview_returns_sha256_and_existing_id(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
    png_file: Path,
) -> None:
    from mm_asset_rag import asset_index
    from mm_asset_rag.asset_index import AssetIndexEntry

    asset_index.upsert_entry(
        AssetIndexEntry(
            asset_id="prior-asset",
            sha256="x" * 64,
            source_type="image",
            relative_path="images/prior.png",
        )
    )
    _stub_vlm_image(monkeypatch, {"title": "X"})
    # Compute the real hash using the same helper.
    from mm_asset_rag.upload_pipeline import UploadPipeline as _UP

    digest = _UP._sha256_file(pipeline, png_file)
    asset_index.upsert_entry(
        AssetIndexEntry(
            asset_id="prior-asset",
            sha256=digest,
            source_type="image",
            relative_path="images/prior.png",
        )
    )
    previews = pipeline.preview([(png_file.name, png_file)])
    assert previews[0].sha256 == digest
    assert previews[0].existing_asset_id == "prior-asset"


def test_cleanup_uses_created_at_not_mtime(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    png_file: Path,
) -> None:
    monkeypatch.setenv("PREVIEW_CACHE_TTL_SECONDS", "60")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    pipeline.preview([(png_file.name, png_file)])
    cache_id = next(iter((pipeline.cache_root).iterdir())).name
    cache_dir = pipeline.cache_root / cache_id
    manifest = cache_dir / "manifest.json"
    import os as _os

    _os.utime(manifest, (1_000_000, 1_000_000))
    assert pipeline.cleanup_expired_caches(now=2_000_000) == 0
    assert cache_dir.exists()


def test_cleanup_disabled_when_ttl_zero(
    pipeline: UploadPipeline, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PREVIEW_CACHE_TTL_SECONDS", "0")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()

    cache_dir = home / ".preview-cache"
    cache_dir.mkdir(exist_ok=True)
    stale = cache_dir / "0123456789ab"
    stale.mkdir()
    (stale / "manifest.json").write_text("{}")
    import os

    os.utime(stale / "manifest.json", (1, 1))

    assert pipeline.cleanup_expired_caches(now=10_000_000) == 0
    assert stale.exists()


def _write_png(home: Path, name: str) -> Path:
    from PIL import Image

    p = home / name
    Image.new("RGB", (8, 8), color=(0, 200, 0)).save(p)
    return p


# ─── VLM auto-meta concurrency ───────────────────────────────────────


def test_fill_auto_meta_respects_concurrency_limit(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
) -> None:
    from mm_asset_rag import auto_meta, upload_pipeline
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("AUTO_META_MAX_CONCURRENCY", "2")
    get_settings.cache_clear()

    pngs = [_write_png(home, f"img_{i}.png") for i in range(4)]
    in_flight = 0
    peak = 0
    barrier_lock = threading.Lock()
    barrier = threading.Event()

    def stub_image(_path: Path) -> auto_meta.AutoMeta:
        nonlocal in_flight, peak
        with barrier_lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            barrier.wait(timeout=1.0)
        finally:
            with barrier_lock:
                in_flight -= 1
        return auto_meta.AutoMeta(title="t", tags=["x"])

    monkeypatch.setattr(upload_pipeline.auto_meta, "auto_meta_image", stub_image)

    pipeline.preview([(p.name, p) for p in pngs])
    barrier.set()
    assert peak <= 2


def test_fill_auto_meta_keeps_order_and_skips_rejected(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
) -> None:
    from mm_asset_rag import auto_meta, upload_pipeline
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("AUTO_META_MAX_CONCURRENCY", "4")
    get_settings.cache_clear()

    good = _write_png(home, "good.png")
    bogus = home / "bad.bin"
    bogus.write_bytes(b"plain bytes")
    other = _write_png(home, "other.png")

    counter = [0]

    def stub_image(_path: Path) -> auto_meta.AutoMeta:
        counter[0] += 1
        return auto_meta.AutoMeta(title=f"T{counter[0]}", tags=[])

    monkeypatch.setattr(upload_pipeline.auto_meta, "auto_meta_image", stub_image)

    previews = pipeline.preview([(good.name, good), (bogus.name, bogus), (other.name, other)])
    titles = [p.effective_title for p in previews]
    assert titles[0] == "T1"
    assert titles[2] == "T2"
    assert previews[1].rejected_reason is not None


def test_fill_auto_meta_swallows_single_failure(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: UploadPipeline,
    home: Path,
) -> None:
    from mm_asset_rag import upload_pipeline
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("AUTO_META_MAX_CONCURRENCY", "2")
    get_settings.cache_clear()

    p1 = _write_png(home, "one.png")
    p2 = _write_png(home, "two.png")

    def boom(_path: Path) -> auto_meta.AutoMeta:
        raise RuntimeError("network down")

    monkeypatch.setattr(upload_pipeline.auto_meta, "auto_meta_image", boom)
    previews = pipeline.preview([(p1.name, p1), (p2.name, p2)])
    for p in previews:
        assert p.auto_meta is None
