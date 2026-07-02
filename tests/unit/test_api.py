"""Tests for mm_asset_rag.api (FastAPI).

Offline unit tests: HTTP routing + request parsing via TestClient, while
background ingest / embeddings / Qdrant are monkeypatched.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mm_asset_rag.api import app


@pytest.fixture
def client(tmp_home) -> TestClient:
    return TestClient(app)


@pytest.fixture
def png_bytes() -> bytes:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    import io

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(0, 0, 255)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def pdf_bytes() -> bytes:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    doc = fitz.open()
    doc.new_page()
    payload = doc.tobytes()
    doc.close()
    return payload


def test_health_endpoint_reports_status(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_backend"] == "qdrant"
    assert body["assets"] == 0
    assert body["version"] == "0.1.0"


def test_root_serves_bundled_ui(client: TestClient) -> None:
    """GET / should serve ``index.html`` plus CSP + security headers."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>" in response.text
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-cache"


def test_search_endpoint_text_mode(client: TestClient) -> None:
    with patch("mm_asset_rag.api.dispatch_search", return_value=[]):
        response = client.post(
            "/search",
            json={"query": "anything", "mode": "text", "top_k": 3},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "anything"
    assert body["hits"] == []


def test_search_endpoint_image_to_image_requires_path(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "x", "mode": "image-to-image"},
    )
    assert response.status_code == 400
    assert "image_path" in response.json()["detail"]


def test_search_endpoint_rejects_absolute_image_path(client: TestClient) -> None:
    """Image path validation rejects absolute paths at the API layer."""
    response = client.post(
        "/search",
        json={"query": "x", "mode": "image-to-image", "image_path": "/etc/passwd"},
    )
    assert response.status_code == 422


def test_search_endpoint_rejects_parent_traversal(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "x", "mode": "hybrid", "image_path": "../escape.png"},
    )
    assert response.status_code == 422


def test_answer_endpoint_returns_fallback(client: TestClient) -> None:
    with patch(
        "mm_asset_rag.api.answer_question",
        return_value={"question": "q", "answer": "no LLM configured", "sources": []},
    ):
        response = client.post("/answer", json={"question": "q", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["answer"] == "no LLM configured"


def test_eval_endpoint_runs_cases(client: TestClient) -> None:
    with patch("mm_asset_rag.api.run_eval", return_value=[]):
        response = client.post("/eval", json={})
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_upload_preview_rejects_empty_batch(client: TestClient) -> None:
    response = client.post("/upload/preview", files=[])
    assert response.status_code == 422  # FastAPI requires at least one file field


def test_upload_preview_rejects_oversized_file(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    png_bytes: bytes,
) -> None:
    monkeypatch.setenv("UPLOAD_MAX_FILE_BYTES", "8")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    response = client.post(
        "/upload/preview",
        files=[("files", ("scene.png", png_bytes, "image/png"))],
    )
    assert response.status_code == 413


def test_upload_preview_accepts_png(client: TestClient, png_bytes: bytes) -> None:
    with patch("mm_asset_rag.auto_meta.auto_meta_image", return_value=None):
        response = client.post(
            "/upload/preview",
            files=[("files", ("scene.png", png_bytes, "image/png"))],
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cache_id"]
    assert len(body["previews"]) == 1
    preview = body["previews"][0]
    assert preview["sniff"]["source_type"] == "image"
    assert preview.get("is_supported", True)


def test_upload_preview_accepts_pdf(client: TestClient, pdf_bytes: bytes) -> None:
    with patch("mm_asset_rag.auto_meta.auto_meta_pdf_first_page", return_value=None):
        response = client.post(
            "/upload/preview",
            files=[("files", ("paper.pdf", pdf_bytes, "application/pdf"))],
        )
    assert response.status_code == 200, response.text
    preview = response.json()["previews"][0]
    assert preview["sniff"]["source_type"] == "pdf"


def test_upload_confirm_spawns_ingest_task(
    client: TestClient,
    png_bytes: bytes,
) -> None:
    with patch("mm_asset_rag.auto_meta.auto_meta_image", return_value=None):
        preview_response = client.post(
            "/upload/preview",
            files=[("files", ("scene.png", png_bytes, "image/png"))],
        )
    body = preview_response.json()
    preview_id = body["previews"][0]["preview_id"]

    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        from mm_asset_rag.service import TaskRecord

        fake_service = mock_get_service.return_value
        fake_service.ingest_assets.return_value = TaskRecord(task_id="abc123def456", kind="ingest")
        response = client.post(
            "/upload/confirm",
            json={
                "cache_id": body["cache_id"],
                "edits": [
                    {
                        "preview_id": preview_id,
                        "title": "Edited Scene",
                        "tags": ["scene", "manual"],
                    }
                ],
            },
        )

    assert response.status_code == 200, response.text
    out = response.json()
    assert out["task_id"] == "abc123def456"
    assert out["kind"] == "ingest"
    args, _kwargs = fake_service.ingest_assets.call_args
    assets = args[0]
    assert assets[0].title == "Edited Scene"
    assert assets[0].tags == ["scene", "manual"]


def test_tasks_endpoint_returns_history(client: TestClient) -> None:
    response = client.get("/tasks")
    assert response.status_code == 200
    assert "tasks" in response.json()
    assert isinstance(response.json()["tasks"], list)


def test_retry_task_endpoint_returns_new_task(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    png_bytes: bytes,
) -> None:
    with patch("mm_asset_rag.auto_meta.auto_meta_image", return_value=None):
        preview = client.post(
            "/upload/preview",
            files=[("files", ("scene.png", png_bytes, "image/png"))],
        )
    body = preview.json()

    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        from mm_asset_rag.service import TaskRecord

        fake_service = mock_get_service.return_value
        fake_service.retry_task.return_value = TaskRecord(
            task_id="newretry01",
            kind="ingest",
            uploaded_files=body["previews"][0]["source_path"].split("/")[-1:]
            if False
            else ["images/scene.png"],
            source="retry",
            origin_task_id="orig0000001",
        )
        response = client.post("/tasks/orig0000001/retry")
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["task_id"] == "newretry01"
    assert out["origin_task_id"] == "orig0000001"
    assert out["source"] == "retry"


def test_retry_task_endpoint_404(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.retry_task.side_effect = KeyError("unknown task missing")
        response = client.post("/tasks/missing/retry")
    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


def test_retry_task_endpoint_400_for_unretryable(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.retry_task.side_effect = ValueError("task x cannot be retried")
        response = client.post("/tasks/abc/retry")
    assert response.status_code == 400
    assert "cannot be retried" in response.json()["detail"]


def test_retry_task_endpoint_accepts_force_and_failed_only(client: TestClient) -> None:
    """``--force`` and ``--failed-only`` compose: the new task re-parses
    only the previously failed assets and clears only their cache.
    """
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        from mm_asset_rag.service import TaskRecord

        fake_service = mock_get_service.return_value
        fake_service.retry_task.return_value = TaskRecord(
            task_id="newretry02",
            kind="ingest",
            uploaded_files=["images/bad.png"],
            source="retry",
            origin_task_id="orig0000002",
            force=True,
            failed_only=True,
        )
        response = client.post("/tasks/orig0000002/retry?force=true&failed_only=true")
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["force"] is True
    assert out["failed_only"] is True
    # Pass-through: both flags reached retry_task.
    kwargs = fake_service.retry_task.call_args.kwargs
    assert kwargs["force"] is True
    assert kwargs["failed_only"] is True


def test_chat_stream_endpoint_emits_sources_and_done(client: TestClient) -> None:
    with (
        patch("mm_asset_rag.api.dispatch_search", return_value=[]),
        patch("mm_asset_rag.api.stream_answer_chunks", return_value=iter(["hello ", "world"])),
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
    joined = "".join(e["text"] for e in events if e["event"] == "token")
    assert joined == "hello world"


def test_chat_stream_image_to_image_requires_path(client: TestClient) -> None:
    response = client.post(
        "/chat/stream",
        json={"question": "x", "mode": "image-to-image"},
    )
    assert response.status_code == 200
    assert "x-ndjson" in response.headers["content-type"]
    events = [json.loads(line) for line in response.text.split("\n") if line.strip()]
    assert any(e.get("event") == "error" for e in events)


def test_chat_stream_runs_through_to_thread(client: TestClient) -> None:
    """``stream_answer_chunks`` is sync; ``chat_stream`` wraps it in
    ``asyncio.to_thread`` so the event loop isn't pinned. Smoke-check
    that the wrapping still produces a full event sequence when the
    producer uses ``time.sleep`` style blocking.
    """
    import time as _time

    def slow_chunks(question, hits):
        for w in ["slow", " ", "stream"]:
            _time.sleep(0.001)
            yield w

    with (
        patch("mm_asset_rag.api.dispatch_search", return_value=[]),
        patch("mm_asset_rag.api.stream_answer_chunks", side_effect=slow_chunks),
    ):
        response = client.post(
            "/chat/stream",
            json={"question": "hi", "mode": "text", "top_k": 3},
        )
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.split("\n") if line.strip()]
    kinds = [e["event"] for e in events]
    assert "sources" in kinds and "token" in kinds and "done" in kinds
    joined = "".join(e["text"] for e in events if e["event"] == "token")
    assert joined == "slow stream"
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.retry_task.side_effect = ValueError("task x cannot be retried")
        response = client.post("/tasks/abc/retry")
    assert response.status_code == 400
    assert "cannot be retried" in response.json()["detail"]


def test_list_assets_endpoint_returns_index(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        from mm_asset_rag.asset_index import AssetIndexEntry

        fake_service = mock_get_service.return_value
        fake_service.list_assets.return_value = [
            AssetIndexEntry(
                asset_id="alpha",
                sha256="abc",
                source_type="image",
                relative_path="images/alpha.png",
                asset_title="Alpha",
            )
        ]
        response = client.get("/assets")
    assert response.status_code == 200
    body = response.json()
    assert len(body["assets"]) == 1
    assert body["assets"][0]["asset_id"] == "alpha"


def test_delete_asset_endpoint_200(client: TestClient) -> None:

    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        from mm_asset_rag.service import DeleteAssetReport

        fake_service = mock_get_service.return_value
        fake_service.delete_asset.return_value = DeleteAssetReport(
            asset_id="alpha", file_deleted=True, was_known=True
        )
        response = client.delete("/assets/alpha")
    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == "alpha"
    assert body["file_deleted"] is True


def test_delete_asset_endpoint_404(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.delete_asset.return_value.was_known = False
        response = client.delete("/assets/missing")
    assert response.status_code == 404
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.delete_asset.return_value.was_known = False
        response = client.delete("/assets/missing")
    assert response.status_code == 404


def test_task_stream_endpoint_emits_initial_and_done(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value

        def fake_stream(task_id: str, **kw):
            yield {"event": "snapshot", "task": {"task_id": task_id, "status": "running"}}
            yield {"event": "done", "status": "done"}

        fake_service.stream_task.side_effect = fake_stream
        response = client.get("/tasks/abc/stream")
    assert response.status_code == 200
    assert "x-ndjson" in response.headers["content-type"]
    lines = [line for line in response.text.split("\n") if line.strip()]
    events = [json.loads(line) for line in lines]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "snapshot"
    assert kinds[-1] == "done"


def test_task_stream_endpoint_emits_nocache_headers(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value

        def fake_stream(task_id: str, **kw):
            yield {"event": "snapshot", "task": {"task_id": task_id, "status": "running"}}
            yield {"event": "done", "status": "done"}

        fake_service.stream_task.side_effect = fake_stream
        response = client.get("/tasks/abc/stream")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache"
    assert response.headers.get("x-accel-buffering") == "no"


def test_task_stream_endpoint_unknown_yields_error(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value

        def fake_stream(task_id: str, **kw):
            yield {"event": "error", "message": f"unknown task {task_id}"}

        fake_service.stream_task.side_effect = fake_stream
        response = client.get("/tasks/missing/stream")
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.split("\n") if line.strip()]
    assert any(e.get("event") == "error" for e in events)
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.delete_asset.return_value.was_known = False
        response = client.delete("/assets/missing")
    assert response.status_code == 404


def test_get_asset_endpoint_returns_detail(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.get_asset_detail.return_value = {
            "asset_id": "alpha",
            "sha256": "abc",
            "source_type": "image",
            "relative_path": "images/alpha.png",
            "title": "Alpha",
            "tags": ["beach"],
            "file_exists": True,
            "parsed_exists": False,
            "captions_exists": False,
        }
        response = client.get("/assets/alpha")
    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == "alpha"
    assert body["tags"] == ["beach"]


def test_get_asset_endpoint_404(client: TestClient) -> None:
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        fake_service = mock_get_service.return_value
        fake_service.get_asset_detail.return_value = None
        response = client.get("/assets/missing")
    assert response.status_code == 404
