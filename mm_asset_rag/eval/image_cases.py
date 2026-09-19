"""Image-eval manifest loading and deterministic v2 case generation.

The manifest (``image-eval.v1``) is the semantic source of truth: it names
the corpus image files, the semantic categories, and the text / image /
negative queries.  ``build_image_eval_cases`` converts it into the existing
qrels-only v2 case schema so the image benchmark shares ``load_cases`` and
the v2 runners with the text evaluations.

The builder derives document IDs through the same filename-identity helper
used by upload ingest, so generated qrels always match the logical document
IDs produced by ``mmrag ingest-image-eval``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.settings import get_settings
from ..ingest.file_identity import document_id_from_filename

MANIFEST_VERSION = "image-eval.v1"

TEXT_TO_IMAGE_ZH = "text_to_image_zh"
TEXT_TO_IMAGE_EN = "text_to_image_en"
IMAGE_TO_IMAGE = "image_to_image"
NEGATIVE = "negative"

GROUP_ORDER = (TEXT_TO_IMAGE_ZH, TEXT_TO_IMAGE_EN, IMAGE_TO_IMAGE, NEGATIVE)
_ALLOWED_TEXT_GROUPS = (TEXT_TO_IMAGE_ZH, TEXT_TO_IMAGE_EN)


@dataclass(frozen=True)
class ImageEvalTextQuery:
    query_id: str
    group: str
    query: str


@dataclass(frozen=True)
class ImageEvalImageQuery:
    query_id: str
    image: str
    relevant_categories: tuple[str, ...]


@dataclass(frozen=True)
class ImageEvalNegative:
    query_id: str
    query: str


@dataclass(frozen=True)
class ImageEvalCategory:
    id: str
    document_files: tuple[str, ...]
    text_queries: tuple[ImageEvalTextQuery, ...]


@dataclass(frozen=True)
class ImageEvalManifest:
    version: str
    corpus_root: Path
    categories: tuple[ImageEvalCategory, ...]
    image_queries: tuple[ImageEvalImageQuery, ...]
    negatives: tuple[ImageEvalNegative, ...]

    def category(self, category_id: str) -> ImageEvalCategory:
        for category in self.categories:
            if category.id == category_id:
                return category
        raise KeyError(category_id)


def _require_str(payload: dict[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: expected a non-empty string field {field!r}")
    return value


def _require_str_list(payload: dict[str, Any], field: str, *, context: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{context}: expected a list of non-empty strings in field {field!r}")
    return tuple(value)


def load_image_eval_manifest(path: str | Path) -> ImageEvalManifest:
    """Load and strictly validate an ``image-eval.v1`` manifest."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest {manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"manifest {manifest_path}: expected a top-level JSON object.")

    version = _require_str(raw, "version", context=f"manifest {manifest_path}")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"manifest {manifest_path}: unsupported version {version!r} (expected {MANIFEST_VERSION!r})."
        )

    corpus_root = manifest_path.parent / _require_str(
        raw, "corpus_root", context=f"manifest {manifest_path}"
    )

    raw_categories = raw.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError(f"manifest {manifest_path}: expected a non-empty 'categories' list.")

    categories: list[ImageEvalCategory] = []
    seen_category_ids: set[str] = set()
    seen_query_ids: dict[str, str] = {}

    def register_query_id(query_id: str, context: str) -> None:
        if query_id in seen_query_ids:
            raise ValueError(
                f"manifest {manifest_path}: duplicate query_id {query_id!r} "
                f"({seen_query_ids[query_id]} and {context})."
            )
        seen_query_ids[query_id] = context

    for index, raw_category in enumerate(raw_categories):
        context = f"manifest {manifest_path}: category #{index}"
        if not isinstance(raw_category, dict):
            raise ValueError(f"{context}: expected a JSON object.")
        category_id = _require_str(raw_category, "id", context=context)
        if category_id in seen_category_ids:
            raise ValueError(f"{context}: duplicate category id {category_id!r}.")
        seen_category_ids.add(category_id)

        document_files = _require_str_list(raw_category, "document_files", context=context)
        if len(document_files) != 3 or len(set(document_files)) != 3:
            raise ValueError(
                f"{context} ({category_id!r}): exactly three unique document files are required, "
                f"got {len(document_files)}."
            )
        for document_file in document_files:
            if not (corpus_root / document_file).is_file():
                raise ValueError(
                    f"{context} ({category_id!r}): corpus file not found: {corpus_root / document_file}"
                )

        raw_text_queries = raw_category.get("text_queries")
        if not isinstance(raw_text_queries, list):
            raise ValueError(f"{context}: expected a 'text_queries' list.")
        text_queries: list[ImageEvalTextQuery] = []
        for raw_query in raw_text_queries:
            query_context = f"{context}: text query"
            if not isinstance(raw_query, dict):
                raise ValueError(f"{query_context}: expected a JSON object.")
            query_id = _require_str(raw_query, "query_id", context=query_context)
            group = _require_str(raw_query, "group", context=query_context)
            if group not in _ALLOWED_TEXT_GROUPS:
                raise ValueError(
                    f"{query_context} ({query_id!r}): group must be one of "
                    f"{_ALLOWED_TEXT_GROUPS}, got {group!r}."
                )
            register_query_id(query_id, f"text query {query_id!r} in category {category_id!r}")
            text_queries.append(
                ImageEvalTextQuery(
                    query_id=query_id,
                    group=group,
                    query=_require_str(raw_query, "query", context=query_context),
                )
            )

        categories.append(
            ImageEvalCategory(
                id=category_id,
                document_files=document_files,
                text_queries=tuple(text_queries),
            )
        )

    raw_image_queries = raw.get("image_queries")
    if not isinstance(raw_image_queries, list):
        raise ValueError(f"manifest {manifest_path}: expected an 'image_queries' list.")
    image_queries: list[ImageEvalImageQuery] = []
    for raw_query in raw_image_queries:
        context = f"manifest {manifest_path}: image query"
        if not isinstance(raw_query, dict):
            raise ValueError(f"{context}: expected a JSON object.")
        query_id = _require_str(raw_query, "query_id", context=context)
        register_query_id(query_id, f"image query {query_id!r}")
        relevant_categories = _require_str_list(raw_query, "relevant_categories", context=context)
        unknown = [c for c in relevant_categories if c not in seen_category_ids]
        if unknown:
            raise ValueError(
                f"{context} ({query_id!r}): unknown relevant categories: {', '.join(unknown)}."
            )
        image = _require_str(raw_query, "image", context=context)
        if not (corpus_root / image).is_file():
            raise ValueError(
                f"{context} ({query_id!r}): image file not found: {corpus_root / image}"
            )
        image_queries.append(
            ImageEvalImageQuery(
                query_id=query_id,
                image=image,
                relevant_categories=relevant_categories,
            )
        )

    raw_negatives = raw.get("negatives")
    if not isinstance(raw_negatives, list):
        raise ValueError(f"manifest {manifest_path}: expected a 'negatives' list.")
    negatives: list[ImageEvalNegative] = []
    for raw_query in raw_negatives:
        context = f"manifest {manifest_path}: negative"
        if not isinstance(raw_query, dict):
            raise ValueError(f"{context}: expected a JSON object.")
        query_id = _require_str(raw_query, "query_id", context=context)
        register_query_id(query_id, f"negative {query_id!r}")
        negatives.append(
            ImageEvalNegative(
                query_id=query_id,
                query=_require_str(raw_query, "query", context=context),
            )
        )

    return ImageEvalManifest(
        version=version,
        corpus_root=corpus_root,
        categories=tuple(categories),
        image_queries=tuple(image_queries),
        negatives=tuple(negatives),
    )


def _document_ids(document_files: tuple[str, ...]) -> dict[str, int]:
    max_len = get_settings().upload_slug_max_len
    return {document_id_from_filename(name, max_len=max_len): 1 for name in document_files}


def build_image_eval_cases(
    manifest_path: str | Path, *, case_dir: str | Path | None = None
) -> dict[str, object]:
    """Build the v2 qrels-only case payload from an image-eval manifest.

    ``corpus_root`` is resolved against the manifest directory; relative
    ``image_path`` values are computed against ``case_dir`` (the manifest
    directory by default) so the generated case file is cwd-independent.
    """
    manifest = load_image_eval_manifest(manifest_path)
    output_dir = Path(case_dir) if case_dir is not None else Path(manifest_path).parent

    groups: dict[str, list[dict[str, str]]] = {group: [] for group in GROUP_ORDER}
    qrels: dict[str, dict[str, int]] = {}
    for category in manifest.categories:
        category_qrels = _document_ids(category.document_files)
        for text_query in category.text_queries:
            groups[text_query.group].append(
                {"query_id": text_query.query_id, "query": text_query.query}
            )
            qrels[text_query.query_id] = dict(category_qrels)
    for image_query in manifest.image_queries:
        image_path = manifest.corpus_root / image_query.image
        groups[IMAGE_TO_IMAGE].append(
            {
                "query_id": image_query.query_id,
                "image_path": os.path.relpath(image_path, output_dir),
            }
        )
        query_document_id = document_id_from_filename(
            image_query.image, max_len=get_settings().upload_slug_max_len
        )
        qrels[image_query.query_id] = {
            document_id: 1
            for category_id in image_query.relevant_categories
            for document_id in _document_ids(manifest.category(category_id).document_files)
            if document_id != query_document_id
        }
    for negative in manifest.negatives:
        groups[NEGATIVE].append({"query_id": negative.query_id, "query": negative.query})
        qrels[negative.query_id] = {}

    return {"version": "v2", "groups": groups, "qrels": qrels}


def write_image_eval_cases(manifest_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """Generate the v2 case payload from a manifest and write it to disk."""
    cases = build_image_eval_cases(manifest_path, case_dir=Path(output_path).parent)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cases


def image_eval_corpus_files(manifest_path: str | Path) -> list[Path]:
    """Return the deterministic corpus file list declared by the manifest."""
    manifest = load_image_eval_manifest(manifest_path)
    return [
        manifest.corpus_root / document_file
        for category in manifest.categories
        for document_file in category.document_files
    ]
