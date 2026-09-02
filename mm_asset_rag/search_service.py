"""Transport-neutral application service for retrieval requests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from .paths import get_assets_dir
from .protocols import SearchBackend, SearchFilter
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
    collection: str | None = None
    metadata_filter: dict[str, object] | None = None
    principal: str | None = None


class SearchInputError(ValueError):
    """An invalid input supplied to the search application service."""


def _matches_access_policy(hit: SearchHit, command: SearchCommand) -> bool:
    metadata = hit.metadata
    collection = command.collection or "default"
    if metadata.get("collection") != collection:
        return False
    policy_metadata = metadata.get("metadata")
    if not isinstance(policy_metadata, dict):
        policy_metadata = {}
    for key, expected in (command.metadata_filter or {}).items():
        if policy_metadata.get(key) != expected:
            return False
    principals = metadata.get("allowed_principals")
    if not isinstance(principals, (list, tuple)):
        return False
    return not principals or (command.principal is not None and command.principal in principals)


def _aggregate_documents(hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Collapse chunk/version hits to their logical document without copying corpus data."""
    best: dict[str, SearchHit] = {}
    for hit in hits:
        document_id = hit.metadata.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            continue
        current = best.get(document_id)
        if current is None or hit.score > current.score:
            metadata = {**hit.metadata, "document_id": document_id}
            best[document_id] = replace(hit, asset_id=document_id, metadata=metadata)
    return sorted(
        best.values(), key=lambda hit: (-hit.score, str(hit.metadata.get("document_id", "")))
    )[:top_k]


def _apply_knowledge_policy(hits: list[SearchHit], command: SearchCommand) -> list[SearchHit]:
    return _aggregate_documents(
        [hit for hit in hits if _matches_access_policy(hit, command)], command.top_k
    )


class _PolicyFilteredBackend:
    """Attach one immutable native policy filter to every backend route."""

    def __init__(self, backend: SearchBackend, search_filter: SearchFilter) -> None:
        self._backend = backend
        self._search_filter = search_filter
        self.name = getattr(backend, "name", "filtered")

    def search_text(self, *, query: str, top_k: int) -> list[SearchHit]:
        return self._backend.search_text(
            query=query, top_k=top_k, search_filter=self._search_filter
        )

    def search_text_to_image(self, *, query: str, top_k: int) -> list[SearchHit]:
        return self._backend.search_text_to_image(
            query=query, top_k=top_k, search_filter=self._search_filter
        )

    def search_image(self, *, image_path: Path, top_k: int) -> list[SearchHit]:
        return self._backend.search_image(
            image_path=image_path, top_k=top_k, search_filter=self._search_filter
        )


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
        native_backend = _PolicyFilteredBackend(
            self._backend,
            SearchFilter(
                collection=command.collection or "default",
                metadata=command.metadata_filter or {},
                principal=command.principal,
            ),
        )
        if mode is SearchMode.TEXT:
            hits = text_search_with_rewrite(
                command.query,
                top_k=command.top_k,
                min_score=command.min_score,
                backend=native_backend,
            )
            return _apply_knowledge_policy(hits, command)
        if mode is SearchMode.TEXT_TO_IMAGE:
            return _apply_knowledge_policy(
                native_backend.search_text_to_image(query=command.query, top_k=command.top_k),
                command,
            )
        if mode is SearchMode.IMAGE_TO_IMAGE:
            if image_path is None:
                raise SearchInputError("image_path required for image-to-image")
            return _apply_knowledge_policy(
                native_backend.search_image(image_path=image_path, top_k=command.top_k), command
            )
        hits = hybrid_search_with_rewrite(
            command.query,
            image_path=image_path,
            top_k=command.top_k,
            min_score=command.min_score,
            backend=native_backend,
        )
        return _apply_knowledge_policy(hits, command)


def get_search_service() -> SearchService:
    """Return the production search application service."""
    return SearchService()
