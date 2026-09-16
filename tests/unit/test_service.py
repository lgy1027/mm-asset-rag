from pathlib import Path

from mm_asset_rag import asset_index
from mm_asset_rag.asset_index import DocumentRecord, load_records, upsert_record
from mm_asset_rag.document_store import read_documents, write_documents
from mm_asset_rag.knowledge_models import AccessPolicy, Asset, Chunk, Document, Source
from mm_asset_rag.service import IngestService, _remove_document_rows_from_documents_jsonl


def _record(document_id: str = "guide") -> DocumentRecord:
    document = Document(
        document_id=document_id,
        title="Guide",
        source=Source(source_id=f"upload:{document_id}"),
        access_policy=AccessPolicy(collection="team", allowed_principals=("alice",)),
    )
    return DocumentRecord(
        document=document,
        asset=Asset("a" * 64, "pdf", f"pdfs/{document_id}.pdf"),
    )


def _chunk(record: DocumentRecord) -> Chunk:
    return Chunk.create(
        document=record.document,
        asset=record.asset,
        ordinal=0,
        text="guide body",
        source=record.document.source,
        access_policy=record.document.access_policy,
    )


def test_remove_document_rows_keeps_other_documents(tmp_home: Path) -> None:
    guide, other = _record(), _record("other")
    write_documents([_chunk(guide), _chunk(other)])

    assert _remove_document_rows_from_documents_jsonl("guide") == 1
    assert [chunk.document_id for chunk in read_documents()] == ["other"]


def test_delete_document_removes_current_index_record(tmp_home: Path, monkeypatch) -> None:
    record = _record()
    upsert_record(record)
    write_documents([_chunk(record)])
    class Backend:
        def delete_documents(self, document_ids):
            return {"text": 0, "image": 0}

        def invalidate_caches(self):
            return None

    report = IngestService(backend=Backend()).delete_document("guide")

    assert report.was_known
    assert report.documents_removed == 1
    assert report.chunks_removed == 1
    assert load_records() == []


def test_delete_document_delegates_vector_cleanup_to_injected_backend(
    tmp_home: Path, monkeypatch
) -> None:
    record = _record()
    upsert_record(record)
    write_documents([_chunk(record)])

    class Backend:
        name = "stub"

        def delete_documents(self, document_ids):
            assert document_ids == {"guide"}
            return {"text": 2, "image": 1}

        def invalidate_caches(self):
            return None

    backend = Backend()
    service = IngestService(backend=backend)

    report = service.delete_document("guide")

    assert report.text_collections_scanned == 2
    assert report.image_collections_scanned == 1


def test_replacing_current_document_index_never_keeps_history(tmp_home: Path) -> None:
    old = _record()
    new = DocumentRecord(
        document=old.document,
        asset=Asset("b" * 64, "pdf", "pdfs/guide-new.pdf"),
    )
    upsert_record(old)
    upsert_record(new)

    assert asset_index.load_records() == [new]
