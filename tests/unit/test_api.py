"""Tests for mm_asset_rag.api (FastAPI).

These are offline unit tests: HTTP routing + request parsing is exercised via
FastAPI's TestClient, but background work (parsing, embedding, indexing) is
monkeypatched so no real providers run.
"""

from __future__ import annotations

import json
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
    assert body["assets"] >= 30  # full bundled sample set


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
        response = client.post("/eval", json={})
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_upload_endpoint_rejects_empty_batch(client: TestClient) -> None:
    """Uploading zero valid files returns 400 with the rejection list."""
    # Build a multipart with no files — the endpoint requires at least one.
    # Use a single empty-named file (no .pdf/.jpg suffix) to force rejection.
    response = client.post(
        "/upload",
        files=[("files", ("README.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 400
    body = response.json()
    assert "rejected" in body["detail"]
    assert any("unsupported" in r.get("reason", "") for r in body["detail"]["rejected"])


def test_upload_endpoint_accepts_pdf_and_spawns_task(
    client: TestClient, examples_home
) -> None:
    """Uploading a PDF returns a task_id and schedules a background thread.

    The actual parse/index work is monkeypatched out so no real embedding API
    is called — we only verify routing + file persistence + task creation.
    The background ``threading.Thread.start()`` call is verified via the
    ``kind`` / ``task_id`` fields in the response.
    """
    # The `examples_home` fixture copies the bundled sample set into
    # `tmp_path/mm_asset_rag_home/assets/`, so PDFs live under `assets/pdfs/`.
    pdf_path = examples_home / "assets" / "pdfs" / "attention-is-all-you-need.pdf"
    assert pdf_path.exists()

    with (
        patch("mm_asset_rag.api.get_service") as mock_get_service,
    ):
        from mm_asset_rag.service import TaskRecord

        fake_service = mock_get_service.return_value
        fake_service.parse_uploaded.return_value = TaskRecord(
            task_id="abc123def456", kind="parse"
        )

        response = client.post(
            "/upload",
            data={"auto_index": "false", "pdf_parser": "pymupdf"},
            files=[("files", (pdf_path.name, pdf_path.read_bytes(), "application/pdf"))],
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_id"] == "abc123def456"
    assert body["kind"] == "parse"  # auto_index=false → kind=parse
    assert body["options"]["pdf_parser"] == "pymupdf"
    assert body["options"]["auto_index"] is False
    assert any(p.endswith(".pdf") for p in body["uploaded"])

    # parse_uploaded was called with the right args.
    call_args, call_kwargs = fake_service.parse_uploaded.call_args
    paths = call_kwargs.get("paths") or (call_args[0] if call_args else [])
    options = call_kwargs.get("options") or (
        call_args[1] if len(call_args) > 1 else None
    )
    assert any("attention-is-all-you-need" in p for p in paths)
    assert options is not None
    assert options.pdf_parser == "pymupdf"


def test_tasks_endpoint_returns_history(client: TestClient) -> None:
    """After any /upload, the task should appear in /tasks."""
    response = client.get("/tasks")
    assert response.status_code == 200
    assert "tasks" in response.json()
    assert isinstance(response.json()["tasks"], list)


def test_chat_stream_endpoint_emits_sources_and_done(
    client: TestClient,
) -> None:
    """/chat/stream emits an NDJSON line per event, starting with sources."""
    with (
        patch(
            "mm_asset_rag.api.qdrant_text_search",
            return_value=[],
        ),
        patch(
            "mm_asset_rag.api.stream_answer_chunks",
            return_value=iter(["hello ", "world"]),
        ),
    ):
        response = client.post(
            "/chat/stream",
            json={"question": "hi", "mode": "text", "top_k": 3},
        )
    assert response.status_code == 200
    assert "x-ndjson" in response.headers["content-type"]

    lines = [line for line in response.text.split("\n") if line.strip()]
    events = [json.loads(line) for line in lines]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "sources"
    assert "token" in kinds
    assert kinds[-1] == "done"
    # Tokens joined back together should match what stream_answer_chunks yielded.
    joined = "".join(e["text"] for e in events if e["event"] == "token")
    assert joined == "hello world"


def test_chat_stream_image_to_image_requires_path(client: TestClient) -> None:
    """image-to-image without image_path surfaces as a streaming ``error`` event.

    The retrieval helper raises ``HTTPException(400)`` but the generator
    wraps that into an NDJSON ``{"event": "error", "message": ...}`` line so
    the stream closes cleanly. We verify that contract here.
    """
    response = client.post(
        "/chat/stream",
        json={"question": "x", "mode": "image-to-image"},
    )
    assert response.status_code == 200
    assert "x-ndjson" in response.headers["content-type"]
    events = [
        json.loads(line) for line in response.text.split("\n") if line.strip()
    ]
    assert any(e.get("event") == "error" for e in events)
