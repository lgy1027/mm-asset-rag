from dataclasses import dataclass, field

from .knowledge_models import AccessPolicy, Asset, Chunk, Document, Source

__all__ = [
    "AccessPolicy",
    "Asset",
    "Chunk",
    "Document",
    "ParsedChunk",
    "SearchHit",
    "Source",
]


@dataclass
class ParsedChunk:
    """Parser-owned text and metadata before document identity is assigned."""

    text: str
    metadata: dict[str, object]

    def to_json(self) -> dict[str, object]:
        """Serialize parser output for in-process enrichment stages."""
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
