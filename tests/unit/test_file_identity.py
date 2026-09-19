from pathlib import Path

from mm_asset_rag.ingest.file_identity import document_id_from_filename, slugify_filename_stem


def test_document_id_from_filename_uses_upload_slug_rules() -> None:
    assert document_id_from_filename("Caltech Airplanes 01_9fe67b3f.jpg") == (
        "Caltech Airplanes 01_9fe67b3f"
    )


def test_slugify_filename_stem_removes_dangerous_characters() -> None:
    assert slugify_filename_stem('a<b>:"c"|d?.jpg') == "a b c d"


def test_slugify_filename_stem_truncates_without_trailing_dot_or_space() -> None:
    assert slugify_filename_stem("abcdefghijklmnopqrstuvwxyz.jpg", max_len=10) == "abcdefghij"


def test_document_id_accepts_path_objects() -> None:
    assert document_id_from_filename(Path("images") / "poster.jpg") == "poster"
