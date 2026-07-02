"""Pluggable extension points for ``mm-asset-rag``.

Three Protocol classes describe the contract each backend / parser / embedder
must satisfy. The runtime ``Registry`` (in ``registry.py``) provides a
single source of truth for what implementations are available.

Adding a new modality (e.g. audio) is a three-line change in this codebase:

1. Write ``parsers/audio_parser.py`` whose class satisfies ``Parser``.
2. In ``mm_asset_rag/parsers/__init__.py`` (or a dedicated module), call
   ``register_parser(AudioParser())``.
3. The CLI ``--pdf-parser`` analog ``--audio-parser`` slot auto-appears in
   argparse because ``parsers.all()`` is queried.

No central dispatch table needs editing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .assets import Asset
from .schema import ParsedDocument

# ─── Parser ──────────────────────────────────────────────────────────────


@runtime_checkable
class Parser(Protocol):
    """Parse a single source asset into a flat list of ``ParsedDocument``.

    Each ``ParsedDocument`` carries modality-neutral ``metadata`` — callers
    store ``asset_id`` / ``source_type`` / ``page`` / ``parser`` / etc. and
    downstream embedding / retrieval steps don't need to special-case the
    parser that produced them.

    ``name`` distinguishes implementations of the same ``source_type`` (e.g.
    ``"pymupdf"`` vs ``"paddleocr_vl"`` for ``source_type="pdf"``); the
    registry keys on ``(source_type, name)``.
    """

    name: str
    source_type: str  # "pdf" | "image" | "audio" | "video"

    def parse(self, asset: Asset, **options: object) -> list[ParsedDocument]: ...


# ─── Embedder ────────────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    """Embed content of a single modality into a fixed-dim vector.

    ``dim`` and ``modality`` drive collection naming in any
    ``VectorBackend``: the active collection becomes
    ``{base}_{modality}_{dim}d`` (or similar). ``name`` is the model
    identifier used for logging / debugging.

    Modality-specific helpers live in :class:`TextEmbedderProtocol`
    and :class:`ImageEmbedderProtocol`; an embedder opts into either
    or both as appropriate. ``qdrant_backend`` uses ``isinstance``
    to check the capability at the call site, so a custom audio
    embedder that implements only :class:`Embedder` is still a
    valid participant in the registry.
    """

    name: str
    modality: str  # "text" | "image" | "audio" | "video_frame"

    def dim(self) -> int: ...

    def embed(self, content: object) -> list[float]: ...

    def embed_batch(self, contents: list[object]) -> list[list[float]]: ...


@runtime_checkable
class TextEmbedderProtocol(Protocol):
    """Mixin protocol an :class:`Embedder` implements to handle text.

    ``qdrant_text_search`` and ``build_qdrant_text_index`` call
    ``embed_text`` directly when the registered embedder conforms to
    this Protocol; otherwise they fall back to the generic
    ``embed_batch`` path.
    """

    def embed_text(self, text: str) -> list[float]: ...


@runtime_checkable
class ImageEmbedderProtocol(Protocol):
    """Mixin protocol an :class:`Embedder` implements to handle images.

    ``qdrant_image_to_image_search`` / ``build_qdrant_image_index`` use
    ``embed_image`` / ``embed_image_batch`` when the registered
    embedder conforms; the helper returns ``None`` for files the
    underlying model cannot process (e.g. non-image paths), letting
    the caller skip them without a hard error.
    """

    def embed_image(self, path: Path) -> list[float] | None: ...

    def embed_image_batch(self, paths: list[Path]) -> list[list[float] | None]: ...


# ─── VectorBackend ───────────────────────────────────────────────────────


@runtime_checkable
class VectorBackend(Protocol):
    """A vector store backend (qdrant, milvus, pinecone, …).

    Collections are identified by a string ``name``. Dense and sparse
    vectors are passed as a single ``vector: dict[str, list[float] |
    SparseVector]`` mapping so the backend doesn't need to know the
    embedding model layout.
    """

    name: str  # "qdrant"

    def ensure_collection(
        self,
        *,
        name: str,
        dim: int,
        sparse: bool = False,
    ) -> None: ...

    def drop_collection(self, name: str) -> None: ...

    def upsert(
        self,
        *,
        collection: str,
        points: list[object],
        wait: bool = True,
    ) -> int: ...

    def retrieve_existing_ids(self, *, collection: str, ids: list[str]) -> set[str]: ...

    def search_points(
        self,
        *,
        collection: str,
        query_vector: list[float],
        sparse_vector: object | None,
        vector_name_dense: str,
        vector_name_sparse: str,
        top_k: int,
    ) -> list[object]: ...

    def search_image_to_image(
        self,
        *,
        collection: str,
        image_path: Path,
        top_k: int,
    ) -> list[object]: ...
