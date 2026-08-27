"""Tests for the transport-neutral search application service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mm_asset_rag import search_service, service
from mm_asset_rag.search_service import (
    SearchCommand,
    SearchInputError,
    SearchMode,
    SearchService,
)


def test_dispatch_search_builds_one_transport_neutral_command(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(
        service, "get_search_service", lambda: SimpleNamespace(execute=commands.append)
    )

    assert service.dispatch_search(query="needle", mode="hybrid", image_path=None, top_k=3) is None
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
    assert calls == [
        ("text", "needle", 3, 0.01, backend),
        ("hybrid", "needle", None, 2, 0.02, backend),
    ]


def test_execute_uses_backend_for_image_modes_and_requires_image_for_image_to_image() -> None:
    calls: list[tuple] = []
    backend = SimpleNamespace(
        search_text_to_image=lambda *, query, top_k: (
            calls.append(("text-to-image", query, top_k)) or []
        ),
        search_image=lambda *, image_path, top_k: (
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
