"""Tests for the transport-neutral search application service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_asset_rag import search_service
from mm_asset_rag.protocols import SearchFilter
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.search_service import (
    SearchCommand,
    SearchInputError,
    SearchMode,
    SearchService,
)


def test_search_service_has_no_concrete_qdrant_dependency() -> None:
    source = Path("mm_asset_rag/search_service.py").read_text(encoding="utf-8")

    assert "backends.qdrant_backend" not in source
    assert "qdrant_text_search" not in source


def test_dispatch_search_builds_one_transport_neutral_command(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(
        search_service, "get_search_service", lambda: SimpleNamespace(execute=commands.append)
    )

    assert (
        search_service.dispatch_search(query="needle", mode="hybrid", image_path=None, top_k=3)
        is None
    )
    assert commands == [SearchCommand(query="needle", mode=SearchMode.HYBRID, top_k=3)]


def test_execute_forwards_per_call_min_score_to_text_and_hybrid_rewrite(monkeypatch) -> None:
    calls: list[tuple] = []
    backend = SimpleNamespace()

    monkeypatch.setattr(
        search_service,
        "text_search_with_rewrite",
        lambda query, *, top_k, min_score, backend: (
            calls.append(("text", query, top_k, min_score, backend)) or []
        ),
    )
    monkeypatch.setattr(
        search_service,
        "hybrid_search_with_rewrite",
        lambda query, *, image_path, top_k, min_score, backend: (
            calls.append(("hybrid", query, image_path, top_k, min_score, backend)) or []
        ),
    )
    search = SearchService(backend=backend)

    assert (
        search.execute(SearchCommand(query="needle", mode=SearchMode.TEXT, top_k=3, min_score=0.01))
        == []
    )
    assert search.execute(SearchCommand(query="needle", top_k=2, min_score=0.02)) == []
    assert calls[0][:4] == ("text", "needle", 3, 0.01)
    assert calls[1][:5] == ("hybrid", "needle", None, 2, 0.02)
    assert calls[0][-1]._backend is backend
    assert calls[1][-1]._backend is backend


def test_execute_uses_backend_for_image_modes_and_requires_image_for_image_to_image() -> None:
    calls: list[tuple] = []
    backend = SimpleNamespace(
        search_text_to_image=lambda *, query, top_k, search_filter: (
            calls.append(("text-to-image", query, top_k)) or []
        ),
        search_image=lambda *, image_path, top_k, search_filter: (
            calls.append(("image-to-image", image_path, top_k)) or []
        ),
    )
    search = SearchService(backend=backend)

    assert (
        search.execute(SearchCommand(query="needle", mode=SearchMode.TEXT_TO_IMAGE, top_k=3)) == []
    )
    with pytest.raises(SearchInputError, match="image_path required for image-to-image"):
        search.execute(SearchCommand(query="needle", mode=SearchMode.IMAGE_TO_IMAGE, top_k=3))
    assert calls == [("text-to-image", "needle", 3)]


def test_auto_mode_routes_picture_requests_to_image_search(monkeypatch) -> None:
    calls: list[str] = []
    backend = SimpleNamespace(
        search_text_to_image=lambda **kwargs: calls.append("image") or [],
    )
    monkeypatch.setattr(
        search_service,
        "hybrid_search_with_rewrite",
        lambda *args, **kwargs: calls.append("hybrid") or [],
    )
    search = SearchService(backend=backend)

    search.execute(SearchCommand(query="找 KO 活动照片", mode=SearchMode.AUTO))
    search.execute(SearchCommand(query="联宝发展史是什么", mode=SearchMode.AUTO))

    assert calls == ["hybrid", "hybrid"]


def test_search_service_logs_route_latency_candidates_and_empty_reason(monkeypatch, caplog) -> None:
    import logging

    monkeypatch.setattr(search_service, "hybrid_search_with_rewrite", lambda *_args, **_kwargs: [])
    with caplog.at_level(logging.INFO, logger="mm_asset_rag.search_service"):
        result = SearchService(backend=SimpleNamespace()).execute(
            SearchCommand(
                query="不存在", mode=SearchMode.AUTO, collection="team", principal="alice"
            )
        )

    assert result == []
    assert "retrieval_event" in caplog.text
    assert "route=hybrid" in caplog.text
    assert "candidates=0" in caplog.text
    assert "reason=no_candidates" in caplog.text


def test_execute_filters_policy_and_aggregates_chunks_by_document(monkeypatch) -> None:
    hits = [
        SearchHit(
            route="qdrant_text",
            score=0.8,
            asset_id="report_v2",
            title="Report",
            source_type="pdf",
            source_path="pdfs/report.pdf",
            evidence="best",
            metadata={
                "document_id": "report",
                "collection": "team",
                "metadata": {"department": "eng"},
                "allowed_principals": ["alice"],
            },
        ),
        SearchHit(
            route="qdrant_text",
            score=0.4,
            asset_id="report_v1",
            title="Report",
            source_type="pdf",
            source_path="pdfs/report.pdf",
            evidence="older",
            metadata={
                "document_id": "report",
                "collection": "team",
                "metadata": {"department": "eng"},
                "allowed_principals": ["alice"],
            },
        ),
        SearchHit(
            route="qdrant_text",
            score=0.9,
            asset_id="private",
            title="Private",
            source_type="pdf",
            source_path="pdfs/private.pdf",
            evidence="no",
            metadata={
                "document_id": "private",
                "collection": "team",
                "metadata": {"department": "eng"},
                "allowed_principals": ["bob"],
            },
        ),
    ]
    monkeypatch.setattr(search_service, "text_search_with_rewrite", lambda *args, **kwargs: hits)

    result = SearchService(backend=SimpleNamespace()).execute(
        SearchCommand(
            query="needle",
            mode=SearchMode.TEXT,
            collection="team",
            metadata_filter={"department": "eng"},
            principal="alice",
        )
    )

    assert [hit.asset_id for hit in result] == ["report"]
    assert result[0].metadata["document_id"] == "report"


def test_execute_fails_closed_for_missing_or_malformed_acl(monkeypatch) -> None:
    """Only an explicit empty ACL is public at the service defense layer."""
    hits = [
        SearchHit(
            route="qdrant_text",
            score=0.9,
            asset_id="public-list",
            title="Public list",
            source_type="pdf",
            source_path="pdfs/public-list.pdf",
            metadata={
                "document_id": "public-list",
                "collection": "team",
                "metadata": {},
                "allowed_principals": [],
            },
        ),
        SearchHit(
            route="qdrant_text",
            score=0.8,
            asset_id="public-tuple",
            title="Public tuple",
            source_type="pdf",
            source_path="pdfs/public-tuple.pdf",
            metadata={
                "document_id": "public-tuple",
                "collection": "team",
                "metadata": {},
                "allowed_principals": (),
            },
        ),
        SearchHit(
            route="qdrant_text",
            score=1.0,
            asset_id="missing-acl",
            title="Missing ACL",
            source_type="pdf",
            source_path="pdfs/missing.pdf",
            metadata={"document_id": "missing-acl", "collection": "team", "metadata": {}},
        ),
        SearchHit(
            route="qdrant_text",
            score=1.0,
            asset_id="malformed-acl",
            title="Malformed ACL",
            source_type="pdf",
            source_path="pdfs/malformed.pdf",
            metadata={
                "document_id": "malformed-acl",
                "collection": "team",
                "metadata": {},
                "allowed_principals": "alice",
            },
        ),
    ]
    monkeypatch.setattr(search_service, "text_search_with_rewrite", lambda *args, **kwargs: hits)

    result = SearchService(backend=SimpleNamespace()).execute(
        SearchCommand(query="needle", mode=SearchMode.TEXT, collection="team")
    )

    assert [hit.asset_id for hit in result] == ["public-list", "public-tuple"]


def test_execute_drops_hit_without_document_identity(monkeypatch) -> None:
    """A physical asset ID must not establish a v2 retrieval identity."""
    hit = SearchHit(
        route="qdrant_text",
        score=1.0,
        asset_id="physical-asset",
        title="Legacy",
        source_type="pdf",
        source_path="pdfs/legacy.pdf",
        metadata={"collection": "team", "metadata": {}, "allowed_principals": []},
    )
    monkeypatch.setattr(search_service, "text_search_with_rewrite", lambda *args, **kwargs: [hit])

    assert (
        SearchService(backend=SimpleNamespace()).execute(
            SearchCommand(query="needle", mode=SearchMode.TEXT, collection="team")
        )
        == []
    )


def test_execute_passes_policy_filter_to_every_native_route(monkeypatch) -> None:
    """SearchService wraps all route calls with the requested native policy filter."""
    calls: list[tuple[str, SearchFilter]] = []

    class Backend:
        name = "capture"

        def search_text(self, *, query, top_k, search_filter):
            calls.append(("text", search_filter))
            return []

        def search_text_to_image(self, *, query, top_k, search_filter):
            calls.append(("text-to-image", search_filter))
            return []

        def search_image(self, *, image_path, top_k, search_filter):
            calls.append(("image-to-image", search_filter))
            return []

    backend = Backend()
    monkeypatch.setattr(
        search_service,
        "text_search_with_rewrite",
        lambda query, *, top_k, backend, **kwargs: backend.search_text(query=query, top_k=top_k),
    )
    monkeypatch.setattr(
        search_service,
        "hybrid_search_with_rewrite",
        lambda query, *, image_path, top_k, backend, **kwargs: (
            backend.search_text(query=query, top_k=top_k)
            + backend.search_text_to_image(query=query, top_k=top_k)
            + (
                backend.search_image(image_path=image_path, top_k=top_k)
                if image_path is not None
                else []
            )
        ),
    )
    monkeypatch.setattr(search_service, "resolve_sandboxed_image_path", lambda _: Path("image.png"))
    command = SearchCommand(
        query="needle",
        mode=SearchMode.TEXT,
        collection="engineering",
        metadata_filter={"department": "search"},
        principal="alice",
    )
    search = SearchService(backend=backend)
    search.execute(command)
    search.execute(SearchCommand(**{**command.__dict__, "mode": SearchMode.TEXT_TO_IMAGE}))
    search.execute(
        SearchCommand(
            **{**command.__dict__, "mode": SearchMode.IMAGE_TO_IMAGE, "image_path": "image.png"}
        )
    )
    search.execute(
        SearchCommand(**{**command.__dict__, "mode": SearchMode.HYBRID, "image_path": "image.png"})
    )

    expected = SearchFilter("engineering", {"department": "search"}, "alice")
    assert calls == [
        ("text", expected),
        ("text-to-image", expected),
        ("image-to-image", expected),
        ("text", expected),
        ("text-to-image", expected),
        ("image-to-image", expected),
    ]
