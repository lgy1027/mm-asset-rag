from dataclasses import dataclass, field

from .knowledge_models import AccessPolicy, Asset, Chunk, Document, DocumentVersion, Source

__all__ = [
    "AccessPolicy",
    "Asset",
    "Chunk",
    "Document",
    "DocumentVersion",
    "ParsedDocument",
    "SearchHit",
    "Source",
]


@dataclass
class ParsedDocument:
    """Transient parser output during the v2 parser migration.

    This compatibility DTO is deliberately not accepted by
    :mod:`mm_asset_rag.document_store`; only ``Chunk`` values are persisted.
    """

    text: str
    metadata: dict[str, object]

    def to_json(self) -> dict[str, object]:
        """Serialize the transient parser DTO for non-persistence callers."""
        return {"text": self.text, "metadata": dict(self.metadata)}


@dataclass
class SearchHit:
    route: str
    score: float
    asset_id: str
    title: str
    source_type: str
    source_path: str
    evidence: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    images: list = field(default_factory=list)
    cache_id: str = ""

    def key(self) -> str:
        page = self.metadata.get("page", "")
        return f"{self.asset_id}:{page}:{self.route}"
