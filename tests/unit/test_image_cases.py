"""Tests for the image-eval manifest loader and deterministic case builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_asset_rag.eval.evaluation_v2 import load_cases
from mm_asset_rag.eval.image_cases import (
    build_image_eval_cases,
    image_eval_corpus_files,
    load_image_eval_manifest,
    write_image_eval_cases,
)


def _write_manifest(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir(parents=True)
    for name in (
        "Cat 01_a.jpg",
        "Cat 02_b.jpg",
        "Cat 03_c.jpg",
        "Dog 01_d.jpg",
        "Dog 02_e.jpg",
        "Dog 03_f.jpg",
    ):
        (images / name).write_bytes(b"image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "image-eval.v1",
                "corpus_root": "images",
                "categories": [
                    {
                        "id": "cat",
                        "document_files": [
                            "Cat 01_a.jpg",
                            "Cat 02_b.jpg",
                            "Cat 03_c.jpg",
                        ],
                        "text_queries": [
                            {
                                "query_id": "tti-cat-zh-01",
                                "group": "text_to_image_zh",
                                "query": "猫",
                            }
                        ],
                    },
                    {
                        "id": "dog",
                        "document_files": [
                            "Dog 01_d.jpg",
                            "Dog 02_e.jpg",
                            "Dog 03_f.jpg",
                        ],
                        "text_queries": [],
                    },
                ],
                "image_queries": [
                    {
                        "query_id": "iti-cat-01",
                        "image": "Cat 01_a.jpg",
                        "relevant_categories": ["cat"],
                    }
                ],
                "negatives": [{"query_id": "negative-001", "query": "汽车"}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_manifest_rejects_category_with_not_exactly_three_documents(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["categories"][1]["document_files"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly three"):
        load_image_eval_manifest(manifest_path)


def _write_json(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_builds_case_relative_image_path_and_self_excluding_i2i_qrels(tmp_path):
    cases = build_image_eval_cases(_write_manifest(tmp_path))

    assert cases["groups"]["image_to_image"][0]["image_path"] == "images/Cat 01_a.jpg"
    assert cases["qrels"]["iti-cat-01"] == {"Cat 02_b": 1, "Cat 03_c": 1}
    assert cases["qrels"]["negative-001"] == {}
    assert load_cases(_write_json(tmp_path, cases), version="v2")["text_to_image_zh"]


def test_manifest_rejects_duplicate_query_ids_across_sections(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["negatives"][0]["query_id"] = "iti-cat-01"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate query_id"):
        load_image_eval_manifest(manifest_path)


def test_checked_in_image_case_file_matches_manifest():
    repo = Path(__file__).resolve().parents[2]
    generated = build_image_eval_cases(repo / "examples" / "image_eval_manifest_v1.json")
    checked_in = json.loads(
        (repo / "examples" / "eval_cases_images_v2.json").read_text(encoding="utf-8")
    )

    assert generated == checked_in
    assert len(image_eval_corpus_files(repo / "examples" / "image_eval_manifest_v1.json")) == 48


def test_write_image_eval_cases_round_trips(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    output_path = tmp_path / "out" / "cases.json"

    cases = write_image_eval_cases(manifest_path, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == cases
