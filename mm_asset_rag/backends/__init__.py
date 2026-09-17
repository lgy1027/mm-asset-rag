"""Registered vector-store backend adapters."""

from __future__ import annotations

from ..core.registry import backends as backend_registry
from ..core.registry import register_backend
from .qdrant import QdrantBackend


def register_builtin_backends() -> None:
    """Ensure the built-in adapters are present in the backend registry."""
    if "qdrant" not in backend_registry:
        register_backend(QdrantBackend())


register_builtin_backends()

__all__ = ["QdrantBackend", "register_builtin_backends"]
