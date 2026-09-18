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


def test_chunk_from_record_migrates_document_version_era_row() -> None:
    """Pre-v2 rows carried every identity field but a different chunk_id
    format; from_record must rebuild them instead of crashing so a
    reindex no longer silently drops the old corpus."""
    legacy_row = {
        "chunk_id": "旧文档@1-deadbeef:3",
        "document_version": {
            "document_id": "旧文档",
            "version_number": 1,
            "content_hash": "deadbeef",
            "version_id": "旧文档@1-deadbeef",
        },
        "asset": {
            "content_hash": "deadbeef",
            "source_type": "pdf",
            "relative_path": "pdfs/旧文档_deadbeef.pdf",
        },
        "ordinal": 3,
        "text": "正文内容",
        "source": {"source_id": "upload:旧文档", "uri": "", "provider": "upload"},
        "access_policy": {
            "collection": "manual-test",
            "allowed_principals": ["alice"],
            "metadata": {},
        },
        "metadata": {"asset_title": "旧文档", "page": 3, "chunk_index": 3, "asset_id": "legacy"},
    }

    chunk = Chunk.from_record(legacy_row)

    assert chunk.chunk_id == "旧文档:3"  # rebuilt to the current invariant
    assert chunk.document.document_id == "旧文档"
    assert chunk.document.title == "旧文档"
    assert chunk.asset.relative_path == "pdfs/旧文档_deadbeef.pdf"
    assert chunk.access_policy.collection == "manual-test"
    assert chunk.access_policy.allowed_principals == ("alice",)
    assert chunk.text == "正文内容"
    assert chunk.metadata["page"] == 3
    assert "asset_id" not in chunk.metadata


def test_chunk_from_record_still_rejects_asset_id_era_row() -> None:
    import pytest

    with pytest.raises(ValueError, match="legacy chunk rows are unsupported"):
        Chunk.from_record({"asset_id": "ancient", "text": "x"})
