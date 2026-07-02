"""Asset dataclass + factory for the auto-ingest upload pipeline.

This module used to host the manifest loader and atomic manifest
writer. Those responsibilities moved out: assets are no longer
declared up-front in ``asset_manifest.json`` — they're constructed on
the fly by the upload pipeline from sniff + VLM + user edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .paths import get_assets_dir

if TYPE_CHECKING:
    from .sniff import SniffedAsset
    from .upload_pipeline import UserEdits


@dataclass(frozen=True)
class Asset:
    """A unit of content the pipeline can parse and index.

    ``source_type`` is ``"pdf"`` or ``"image"``. ``relative_path`` is
    always relative to ``asset_dir`` (which defaults to
    ``$MM_ASSET_RAG_HOME/assets``) so the same Asset object survives
    a move of the home directory.
    """

    asset_id: str
    title: str
    source_type: str
    relative_path: str
    source_url: str = ""
    tags: list[str] = field(default_factory=list)
    asset_dir: Path = field(default_factory=get_assets_dir)
    page_count: int | None = None

    @property
    def file_path(self) -> Path:
        return self.asset_dir / self.relative_path


def from_sniffed(
    sniffed: SniffedAsset,
    relative_path: str,
    *,
    asset_dir: Path,
    user_edits: UserEdits | None = None,
    auto_title: str | None = None,
    auto_tags: list[str] | None = None,
    asset_id_override: str | None = None,
    title_override: str | None = None,
) -> Asset:
    """Construct an ``Asset`` from a ``SniffedAsset`` plus optional metadata.

    Resolution order for each user-facing field:

    * ``title``  → explicit override → user edit → VLM title → sniff default
    * ``tags``   → user edit (if set) → VLM tags → empty list
    * ``source_type`` and ``relative_path`` come straight from sniff
      and the caller (the pipeline knows where the file ended up).
    """
    title = (
        title_override
        or (user_edits.title if user_edits and user_edits.title else None)
        or auto_title
        or sniffed.title
    )
    tags = (
        (user_edits.tags if user_edits and user_edits.tags is not None else None)
        if user_edits and user_edits.tags is not None
        else auto_tags
        if auto_tags
        else []
    )

    return Asset(
        asset_id=asset_id_override or sniffed.asset_id,
        title=title,
        source_type=sniffed.source_type,
        relative_path=relative_path,
        source_url="",
        tags=list(tags),
        asset_dir=asset_dir,
        page_count=sniffed.page_count,
    )
