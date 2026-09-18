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
        """Decode a persisted chunk row, migrating ``document_version``-era rows."""
        if isinstance(payload, dict) and "document_version" in payload:
            return cls._from_legacy_record(payload)
        if not isinstance(payload, dict) or "asset_id" in payload:
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

    @classmethod
    def _from_legacy_record(cls, payload: dict) -> Chunk:
        """Best-effort migration of a ``document_version``-era chunk row.

        Those rows carry every identity field the current model needs,
        but their ``chunk_id`` used the ``doc@version-hash:ordinal``
        format, which violates the current ``document_id:ordinal``
        invariant — so the id is rebuilt from document id + ordinal.
        Older ``asset_id``-era rows predate the persisted access policy
        and stay unsupported.
        """
        version = payload.get("document_version")
        if not isinstance(version, dict):
            raise ValueError("legacy chunk row has no document version")
        document_id = str(version.get("document_id") or "")
        if not document_id:
            raise ValueError("legacy chunk row has no document id")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("legacy chunk metadata is invalid")
        source_data = payload.get("source") or {}
        asset_data = payload.get("asset") or {}
        policy_data = payload.get("access_policy") or {}
        principals = policy_data.get("allowed_principals")
        if not isinstance(principals, list):
            raise ValueError("legacy chunk policy is invalid")
        ordinal = int(payload.get("ordinal", 0))
        policy = AccessPolicy(
            collection=str(policy_data.get("collection") or "default"),
            allowed_principals=tuple(str(value) for value in principals),
            metadata=policy_data.get("metadata") or {},
        )
        source = Source(
            source_id=str(source_data.get("source_id") or f"upload:{document_id}"),
            uri=str(source_data.get("uri", "")),
            provider=str(source_data.get("provider", "upload")),
        )
        return cls(
            chunk_id=f"{document_id}:{ordinal}",
            document=Document(
                document_id=document_id,
                title=str(metadata.get("asset_title") or document_id),
                source=source,
                access_policy=policy,
            ),
            asset=Asset(
                content_hash=str(
                    asset_data.get("content_hash") or version.get("content_hash") or ""
                ),
                source_type=str(asset_data.get("source_type", "")),
                relative_path=str(asset_data.get("relative_path", "")),
            ),
            ordinal=ordinal,
            text=str(payload.get("text", "")),
            source=source,
            access_policy=policy,
            metadata={key: value for key, value in metadata.items() if key != "asset_id"},
        )
