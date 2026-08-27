"""Transport-neutral application service for retrieval requests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .paths import get_assets_dir
from .protocols import SearchBackend
from .query_rewrite import hybrid_search_with_rewrite, text_search_with_rewrite
from .registry import get_backend
from .schema import SearchHit


class SearchMode(str, Enum):
    """Supported retrieval routes."""

    TEXT = "text"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SearchCommand:
    """A transport-neutral request to retrieve matching assets."""

    query: str
    mode: SearchMode = SearchMode.HYBRID
    image_path: str | Path | None = None
    top_k: int = 5
    min_score: float | None = None


class SearchInputError(ValueError):
    """An invalid input supplied to the search application service."""


def resolve_sandboxed_image_path(image_path: str | Path | None) -> Path | None:
    """Resolve an image path strictly inside the configured ``assets/`` directory."""
    if not image_path:
        return None
    assets_dir = get_assets_dir().resolve()
    raw = Path(image_path)
    if raw.is_absolute():
        raise SearchInputError("image_path must be relative to assets/")
    try:
        resolved = (assets_dir / raw).resolve()
    except OSError as exc:
        raise SearchInputError(f"image_path cannot be resolved: {exc}") from exc
    if not resolved.is_relative_to(assets_dir):
        raise SearchInputError("image_path resolves outside assets/")
    if not resolved.is_file():
        raise SearchInputError(f"image_path not found or not a regular file: {raw}")
    return resolved


def coerce_search_mode(mode: str | SearchMode) -> SearchMode:
    """Parse a route mode while retaining the legacy input-error message."""
    try:
        return SearchMode(mode)
    except ValueError as exc:
        raise SearchInputError(
            f"unknown mode {mode!r}; expected one of "
            "'text', 'text-to-image', 'image-to-image', 'hybrid'"
        ) from exc


class SearchService:
    """Execute typed retrieval commands through the active search backend."""

    def __init__(self, backend: SearchBackend | None = None) -> None:
        self._backend = backend if backend is not None else get_backend("qdrant")

    def execute(self, command: SearchCommand) -> list[SearchHit]:
        mode = coerce_search_mode(command.mode)
        image_path = (
            resolve_sandboxed_image_path(command.image_path)
            if mode in {SearchMode.IMAGE_TO_IMAGE, SearchMode.HYBRID}
            else None
        )
        if mode is SearchMode.TEXT:
            return text_search_with_rewrite(
                command.query,
                top_k=command.top_k,
                min_score=command.min_score,
                backend=self._backend,
            )
        if mode is SearchMode.TEXT_TO_IMAGE:
            return self._backend.search_text_to_image(query=command.query, top_k=command.top_k)
        if mode is SearchMode.IMAGE_TO_IMAGE:
            if image_path is None:
                raise SearchInputError("image_path required for image-to-image")
            return self._backend.search_image(image_path=image_path, top_k=command.top_k)
        return hybrid_search_with_rewrite(
            command.query,
            image_path=image_path,
            top_k=command.top_k,
            min_score=command.min_score,
            backend=self._backend,
        )


def get_search_service() -> SearchService:
    """Return the production search application service."""
    return SearchService()
