"""Tests for mm_asset_rag.api (FastAPI)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mm_asset_rag.api import app


@pytest.fixture
def client(examples_home) -> TestClient:
    return TestClient(app)


def test_health_endpoint_reports_status(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_backend"] == "qdrant"
    assert body["assets"] == 30  # full bundled sample set


def test_search_endpoint_text_mode(client: TestClient) -> None:
    with patch(
        "mm_asset_rag.api.qdrant_text_search",
        return_value=[],
    ):
        response = client.post(
            "/search",
            json={"query": "anything", "mode": "text", "top_k": 3},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "anything"
    assert body["hits"] == []


def test_search_endpoint_image_to_image_requires_path(client: TestClient) -> None:
    """image-to-image without image_path returns a 400 from the API."""
    response = client.post(
        "/search",
        json={"query": "x", "mode": "image-to-image"},
    )
    assert response.status_code == 400
    assert "image_path" in response.json()["detail"]


def test_answer_endpoint_returns_fallback(client: TestClient) -> None:
    with patch(
        "mm_asset_rag.api.answer_question",
        return_value={
            "question": "q",
            "answer": "no LLM configured",
            "sources": [],
        },
    ):
        response = client.post("/answer", json={"question": "q", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "no LLM configured"


def test_eval_endpoint_runs_cases(client: TestClient) -> None:
    with patch(
        "mm_asset_rag.api.run_eval",
        return_value=[],
    ):
        response = client.post("/eval")
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_ingest_endpoint_runs_parse_and_index(client: TestClient) -> None:
    with (
        patch("mm_asset_rag.api.command_parse") as mock_parse,
        patch("mm_asset_rag.api.command_index") as mock_index,
    ):
        response = client.post(
            "/ingest",
            json={"limit": 0, "pdf_parser": "pymupdf"},
        )
    assert response.status_code == 200
    assert mock_parse.called
    assert mock_index.called
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "qdrant"
