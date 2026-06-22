"""Runtime registry for parsers / embedders / backends.

The registries are the single source of truth for what implementations are
available at runtime. Each entry is keyed by an implementation-specific
identifier:

* ``parsers``: ``(source_type, name)`` → ``Parser`` — same ``source_type``
  can have multiple implementations (e.g. ``"pdf" / "pymupdf"`` and
  ``"pdf" / "paddleocr_vl"``).
* ``embedders``: ``(modality, name)`` → ``Embedder`` — each modality can
  have a default plus experimental variants.
* ``backends``: ``name`` → ``VectorBackend`` — at most one backend per name;
  the active backend is selected by ``VECTOR_BACKEND`` env or constructor arg.

Registration is idempotent: re-registering an existing key replaces the
binding (useful for tests). Production code calls ``register_*`` exactly
once per implementation, usually in the package ``__init__.py``.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from .protocols import Embedder, Parser, VectorBackend

T = TypeVar("T")


class Registry(Generic[T]):
    """Minimal keyed registry with descriptive error messages."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._items: dict[tuple[str, str] | str, T] = {}

    def register(self, key: tuple[str, str] | str, item: T, *, replace: bool = False) -> None:
        if not replace and key in self._items:
            raise ValueError(f"{self.label} {key!r} already registered")
        self._items[key] = item

    def get(self, key: tuple[str, str] | str) -> T:
        if key not in self._items:
            available = sorted(str(k) for k in self._items)
            raise KeyError(
                f"{self.label} {key!r} not registered; available: {available}"
            )
        return self._items[key]

    def all(self) -> dict[tuple[str, str] | str, T]:
        return dict(self._items)

    def keys(self) -> list[tuple[str, str] | str]:
        return list(self._items)

    def __contains__(self, key: tuple[str, str] | str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


parsers: Registry[Parser] = Registry("parser")
embedders: Registry[Embedder] = Registry("embedder")
backends: Registry[VectorBackend] = Registry("backend")


# ─── Convenience helpers ────────────────────────────────────────────────


def register_parser(parser: Parser, *, replace: bool = False) -> None:
    """Register ``parser`` under ``(parser.source_type, parser.name)``."""
    parsers.register((parser.source_type, parser.name), parser, replace=replace)


def register_embedder(embedder: Embedder, *, replace: bool = False) -> None:
    """Register ``embedder`` under ``(embedder.modality, embedder.name)``."""
    embedders.register((embedder.modality, embedder.name), embedder, replace=replace)


def register_backend(backend: VectorBackend, *, replace: bool = False) -> None:
    """Register ``backend`` under ``backend.name``."""
    backends.register(backend.name, backend, replace=replace)


def get_parser(source_type: str, name: str) -> Parser:
    return parsers.get((source_type, name))


def get_embedder(modality: str, name: str) -> Embedder:
    return embedders.get((modality, name))


def get_backend(name: str) -> VectorBackend:
    return backends.get(name)