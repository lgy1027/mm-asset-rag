from dataclasses import FrozenInstanceError

import pytest

from mm_asset_rag.core.knowledge_models import AccessPolicy, Asset, Chunk, Document, Source


def _document() -> Document:
    return Document(
        document_id="handbook",
        title="Handbook",
        source=Source(source_id="upload:handbook"),
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )


def test_document_asset_and_chunk_identities_are_deterministic() -> None:
    document = _document()
    chunk = Chunk.create(
        document=document,
        asset=Asset(content_hash="a" * 64, source_type="pdf", relative_path="pdfs/a.pdf"),
        ordinal=3,
        text="The handbook body.",
        source=document.source,
        access_policy=document.access_policy,
    )

    assert chunk.chunk_id == "handbook:3"
    assert chunk.document_id == "handbook"
    assert chunk.asset.content_hash == "a" * 64


def test_chunk_identity_is_not_content_or_version_based() -> None:
    document = _document()
    with pytest.raises(ValueError, match="chunk_id"):
        Chunk(
            chunk_id="handbook@2-deadbeef:0",
            document=document,
            asset=Asset(content_hash="a" * 64, source_type="pdf", relative_path="a.pdf"),
            ordinal=0,
            text="body",
            source=document.source,
            access_policy=document.access_policy,
        )


def test_domain_models_are_immutable() -> None:
    document = _document()
    with pytest.raises(FrozenInstanceError):
        document.title = "Other"  # type: ignore[misc]


def test_access_policy_requires_collection_and_explicit_acl() -> None:
    with pytest.raises(ValueError, match="collection"):
        AccessPolicy(collection="", allowed_principals=())

    policy = AccessPolicy(collection="team", allowed_principals=("alice",))
    assert policy.allows("alice")
    assert not policy.allows("bob")
    assert not policy.allows(None)
