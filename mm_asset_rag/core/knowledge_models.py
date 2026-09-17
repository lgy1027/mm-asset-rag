"""Immutable identities for the knowledge-base schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class Source:
    """Origin of a logical document."""

    source_id: str
    uri: str = ""
    provider: str = "upload"

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id is required")

    def to_record(self) -> dict[str, object]:
        return {"source_id": self.source_id, "uri": self.uri, "provider": self.provider}


@dataclass(frozen=True)
class AccessPolicy:
    """The collection and ACL required to read a record."""

    collection: str
    allowed_principals: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.collection.strip():
            raise ValueError("collection is required")
        if any(not principal for principal in self.allowed_principals):
            raise ValueError("allowed_principals cannot contain an empty principal")
        object.__setattr__(self, "allowed_principals", tuple(self.allowed_principals))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    def allows(self, principal: str | None) -> bool:
        return not self.allowed_principals or principal in self.allowed_principals

    def to_record(self) -> dict[str, object]:
        return {
            "collection": self.collection,
            "allowed_principals": list(self.allowed_principals),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Asset:
    """Immutable physical bytes backing the current document.

    ``content_hash`` is a content descriptor, not a public document identity.
    """

    content_hash: str
    source_type: str
    relative_path: str

    def __post_init__(self) -> None:
        if not self.content_hash:
            raise ValueError("content_hash is required")
        if not self.source_type:
            raise ValueError("source_type is required")
        if not self.relative_path:
            raise ValueError("relative_path is required")

    def to_record(self) -> dict[str, str]:
        return {
            "content_hash": self.content_hash,
            "source_type": self.source_type,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class Document:
    """Stable public identity for a knowledge-base document."""

    document_id: str
    title: str
    source: Source
    access_policy: AccessPolicy

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id is required")

    def to_record(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source.to_record(),
        }


@dataclass(frozen=True)
class Chunk:
    """A retrievable, deterministically identified current-document segment."""

    chunk_id: str
    document: Document
    asset: Asset
    ordinal: int
    text: str
    source: Source
    access_policy: AccessPolicy
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        expected = f"{self.document.document_id}:{self.ordinal}"
        if self.chunk_id != expected:
            raise ValueError("chunk_id does not match document chunk identity")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def document_id(self) -> str:
        return self.document.document_id

    @classmethod
    def create(
        cls,
        *,
        document: Document,
        asset: Asset,
        ordinal: int,
        text: str,
        source: Source,
        access_policy: AccessPolicy,
        metadata: Mapping[str, object] | None = None,
    ) -> Chunk:
        return cls(
            chunk_id=f"{document.document_id}:{ordinal}",
            document=document,
            asset=asset,
            ordinal=ordinal,
            text=text,
            source=source,
            access_policy=access_policy,
            metadata=metadata or {},
        )

    def to_record(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document.to_record(),
            "asset": self.asset.to_record(),
            "ordinal": self.ordinal,
            "text": self.text,
            "source": self.source.to_record(),
            "access_policy": self.access_policy.to_record(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, payload: object) -> Chunk:
        """Decode a complete current-document chunk row."""
        if not isinstance(payload, dict) or "asset_id" in payload or "document_version" in payload:
            raise ValueError("legacy chunk rows are unsupported")
        document_data = payload.get("document")
        asset_data = payload.get("asset")
        source_data = payload.get("source")
        policy_data = payload.get("access_policy")
        if not all(
            isinstance(value, dict)
            for value in (document_data, asset_data, source_data, policy_data)
        ):
            raise ValueError("chunk is missing identity fields")
        principals = policy_data.get("allowed_principals")
        policy_metadata = policy_data.get("metadata", {})
        chunk_metadata = payload.get("metadata", {})
        if (
            not isinstance(principals, list)
            or not isinstance(policy_metadata, dict)
            or not isinstance(chunk_metadata, dict)
        ):
            raise ValueError("chunk policy or metadata is invalid")
        return cls(
            chunk_id=str(payload["chunk_id"]),
            document=Document(
                document_id=str(document_data["document_id"]),
                title=str(document_data.get("title", "")),
                source=Source(
                    source_id=str(document_data["source"]["source_id"]),
                    uri=str(document_data["source"].get("uri", "")),
                    provider=str(document_data["source"].get("provider", "upload")),
                ),
                access_policy=AccessPolicy(
                    collection=str(policy_data["collection"]),
                    allowed_principals=tuple(str(value) for value in principals),
                    metadata=policy_metadata,
                ),
            ),
            asset=Asset(
                content_hash=str(asset_data["content_hash"]),
                source_type=str(asset_data["source_type"]),
                relative_path=str(asset_data["relative_path"]),
            ),
            ordinal=int(payload["ordinal"]),
            text=str(payload["text"]),
            source=Source(
                source_id=str(source_data["source_id"]),
                uri=str(source_data.get("uri", "")),
                provider=str(source_data.get("provider", "upload")),
            ),
            access_policy=AccessPolicy(
                collection=str(policy_data["collection"]),
                allowed_principals=tuple(str(value) for value in principals),
                metadata=policy_metadata,
            ),
            metadata=chunk_metadata,
        )
