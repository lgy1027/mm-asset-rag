"""Registered vector-store backend adapters."""

from __future__ import annotations

from ..registry import register_backend
from .qdrant import QdrantBackend

register_backend(QdrantBackend())

__all__ = ["QdrantBackend"]
