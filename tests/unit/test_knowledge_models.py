from dataclasses import FrozenInstanceError

import pytest

from mm_asset_rag.knowledge_models import (
    AccessPolicy,
    Asset,
    Chunk,
    Document,
    DocumentVersion,
    Source,
)


def _document() -> Document:
    return Document(
        document_id="handbook",
        title="Handbook",
        source=Source(source_id="upload:handbook"),
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )


def test_document_version_and_chunk_identities_are_deterministic() -> None:
    document = _document()
    first = DocumentVersion.create(document, "a" * 64, version_number=1)
    repeated = DocumentVersion.create(document, "a" * 64, version_number=1)
    chunk = Chunk.create(
        document_version=first,
        asset=Asset(content_hash="a" * 64, source_type="pdf", relative_path="pdfs/a.pdf"),
        ordinal=3,
        text="The handbook body.",
        source=document.source,
        access_policy=document.access_policy,
    )

    assert repeated == first
    assert first.version_id == f"handbook@1-{'a' * 64}"
    assert chunk.chunk_id == f"handbook@1-{'a' * 64}:3"
    assert chunk.document_id == "handbook"


def test_document_version_advance_is_content_sensitive_and_immutable() -> None:
    document = _document()
    first = DocumentVersion.create(document, "a" * 64, version_number=1)
    second = first.advance("b" * 64)

    assert second.version_number == 2
    assert second.version_id == f"handbook@2-{'b' * 64}"
    with pytest.raises(FrozenInstanceError):
        second.version_number = 4  # type: ignore[misc]


def test_identity_does_not_collapse_hashes_with_the_same_prefix() -> None:
    document = _document()
    left_hash = "a" * 12 + "1" * 52
    right_hash = "a" * 12 + "2" * 52

    left = DocumentVersion.create(document, left_hash)
    right = DocumentVersion.create(document, right_hash)

    assert left.version_id != right.version_id
    assert left.version_id.endswith(left_hash)
    assert right.version_id.endswith(right_hash)


def test_persisted_identity_ids_are_recomputed_and_validated() -> None:
    document = _document()
    with pytest.raises(ValueError, match="version_id"):
        DocumentVersion(
            document_id=document.document_id,
            version_number=1,
            content_hash="a" * 64,
            version_id="tampered",
        )

    version = DocumentVersion.create(document, "a" * 64)
    with pytest.raises(ValueError, match="chunk_id"):
        Chunk(
            chunk_id="tampered",
            document_version=version,
            asset=Asset(content_hash="a" * 64, source_type="pdf", relative_path="a.pdf"),
            ordinal=0,
            text="body",
            source=document.source,
            access_policy=document.access_policy,
        )


def test_access_policy_requires_collection_and_explicit_acl() -> None:
    with pytest.raises(ValueError, match="collection"):
        AccessPolicy(collection="", allowed_principals=())

    policy = AccessPolicy(collection="team", allowed_principals=("alice",))
    assert policy.allows("alice")
    assert not policy.allows("bob")
    assert not policy.allows(None)
