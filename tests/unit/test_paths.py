"""Tests for mm_asset_rag.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag import paths


def test_get_data_dir_uses_env_var(tmp_home: Path) -> None:
    assert paths.get_data_dir() == tmp_home
    assert tmp_home.exists()


def test_get_data_dir_creates_directory(tmp_path, monkeypatch) -> None:
    target = tmp_path / "does" / "not" / "exist"
    monkeypatch.setenv("MM_ASSET_RAG_HOME", str(target))
    assert paths.get_data_dir() == target
    assert target.is_dir()


def test_get_data_dir_falls_back_to_home(monkeypatch) -> None:
    monkeypatch.delenv("MM_ASSET_RAG_HOME", raising=False)
    monkeypatch.setenv("HOME", "/tmp/fake-home-for-test")
    assert paths.get_data_dir() == Path("/tmp/fake-home-for-test/.mm_asset_rag")


@pytest.mark.parametrize(
    "func, suffix",
    [
        (paths.get_assets_dir, "assets"),
        (paths.get_parsed_dir, "parsed"),
        (paths.get_captions_dir, "captions"),
        (paths.get_indexes_dir, "indexes"),
        (paths.get_qdrant_path, "indexes/qdrant"),
    ],
)
def test_subdirs_resolve_under_data_dir(tmp_home: Path, func, suffix: str) -> None:
    assert func() == tmp_home / suffix


def test_get_documents_jsonl(tmp_home: Path) -> None:
    assert paths.get_documents_jsonl() == tmp_home / "documents.jsonl"


def test_get_eval_report(tmp_home: Path) -> None:
    assert paths.get_eval_report() == tmp_home / "eval_report.json"


def test_physical_cache_id_uses_the_complete_relative_path() -> None:
    cache_id = getattr(paths, "physical_cache_id", None)

    assert callable(cache_id)
    pdf_id = cache_id("pdfs/shared.pdf")
    document_id = cache_id("documents/shared.pdf")
    assert pdf_id != document_id
    assert len(pdf_id) == 64
    assert pdf_id == cache_id("pdfs/shared.pdf")


def test_safe_parsed_image_path_keeps_same_stem_assets_separate(tmp_home: Path) -> None:
    pdf_id = paths.physical_cache_id("pdfs/shared.pdf")
    document_id = paths.physical_cache_id("documents/shared.pdf")
    pdf_image = tmp_home / "parsed" / pdf_id / "images" / "figure.png"
    document_image = tmp_home / "parsed" / document_id / "images" / "figure.png"
    pdf_image.parent.mkdir(parents=True)
    document_image.parent.mkdir(parents=True)
    pdf_image.write_bytes(b"pdf")
    document_image.write_bytes(b"document")

    assert paths.safe_parsed_image_path(pdf_id, "figure.png") == pdf_image
    assert paths.safe_parsed_image_path(document_id, "figure.png") == document_image
