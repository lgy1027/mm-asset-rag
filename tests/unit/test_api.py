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
    # base_url pins the Host header to ``127.0.0.1`` so requests clear the
    # TrustedHostMiddleware (which defaults to loopback only). Production
    # callers reach the API on 127.0.0.1/localhost the same way.
    return TestClient(app, base_url="http://127.0.0.1")


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


def test_iter_sync_in_thread_streams_incrementally() -> None:
    """The bridge yields each item as the producer emits it, not buffered.

    Regression guard for the old ``list(stream_answer_chunks(...))`` which
    collected the whole generator before yielding anything. Here the producer
    sets a flag after its first yield; the consumer must observe that flag
    set *before* the producer emits its second item — which only holds if
    the consumer drained the first item promptly instead of waiting for the
    producer to finish.
    """
    import asyncio
    import threading

    from mm_asset_rag.api import _iter_sync_in_thread

    emitted_after_first = threading.Event()

    def producer():
        yield "a"
        # The consumer has received "a" and is now awaiting the next item.
        # Only set the flag now — if the bridge had buffer-all'd, the
        # consumer would not yet have run.
        emitted_after_first.set()
        yield "b"

    async def main():
        bridge = await _iter_sync_in_thread(producer)
        first = await asyncio.to_thread(bridge.get)
        assert first == "a"
        # The producer is parked on its second yield; the flag proves it
        # already passed the first yield (i.e. "a" was delivered live).
        assert emitted_after_first.is_set()
        # Drain the rest + sentinel.
        items = []
        while True:
            item = await asyncio.to_thread(bridge.get)
            from mm_asset_rag.api import _STREAM_DONE

            if item is _STREAM_DONE:
                break
            items.append(item)
        return items

    rest = asyncio.run(main())
    assert rest == ["b"]


def test_iter_sync_in_thread_stop_signals_producer() -> None:
    """Setting ``bridge.stop`` lets the producer exit between yields.

    Models a client disconnect: the consumer stops draining and signals
    stop; the producer, which checks ``stop`` between yields, bails out
    instead of running the rest of the (potentially long, LLM-backed)
    generator to completion.
    """
    import asyncio
    import threading
    import time

    from mm_asset_rag.api import _STREAM_DONE, _iter_sync_in_thread

    past_first = threading.Event()
    producer_exited = threading.Event()
    # Holder for the bridge's stop event; filled by the consumer right after
    # the bridge is constructed, so the producer can read the *same* event
    # the consumer signals on disconnect.
    stop_holder: list[threading.Event] = []

    def producer():
        yield "first"
        past_first.set()
        # Each iteration sleeps briefly (emulating the network wait of a real
        # LLM stream) and checks stop before yielding — so a consumer that
        # sets stop mid-stream makes the producer bail on its next wake
        # instead of running all 1000 iterations to completion.
        for _ in range(1000):
            time.sleep(0.01)
            if stop_holder and stop_holder[0].is_set():
                producer_exited.set()
                return
            yield "more"
        yield "tail"

    async def main():
        bridge = await _iter_sync_in_thread(producer)
        stop_holder.append(bridge.stop)  # type: ignore[attr-defined]

        # The producer sets past_first right after its first yield; wait so
        # we know it is parked inside the loop before we signal stop.
        past_first.wait(2.0)
        bridge.stop.set()  # type: ignore[attr-defined]
        # Drain anything already queued + the sentinel. The producer should
        # observe ``stop`` on its next loop check and return, so the sentinel
        # arrives promptly (well within the 2 s timeout).
        while True:
            item = await asyncio.to_thread(bridge.get, timeout=2.0)
            if item is _STREAM_DONE:
                break

    asyncio.run(main())
    # The producer must have exited *because of* stop, not by running out.
    assert producer_exited.is_set()


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


# ─── Auth (MMRAG_API_TOKEN) ──────────────────────────────────────────────
#
# The token guard is opt-in: unset = zero-config loopback (no auth). When
# set, the destructive + write endpoints require the token; read endpoints
# stay open so the bundled web UI keeps working without one.


def _with_token_env(monkeypatch, token: str | None) -> None:
    """Set/unset ``MMRAG_API_TOKEN`` and clear the settings cache so the
    next ``get_settings()`` sees it (the dependency reads settings per
    request, not at import)."""
    from mm_asset_rag.settings import get_settings

    if token is None:
        monkeypatch.delenv("MMRAG_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MMRAG_API_TOKEN", token)
    get_settings.cache_clear()


def test_auth_disabled_by_default_no_token_required(client: TestClient, monkeypatch) -> None:
    """Zero-config default: no MMRAG_API_TOKEN → guarded endpoints are open."""
    _with_token_env(monkeypatch, None)
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        # /tasks/{id}/retry hits require_token; with no token configured it
        # must pass the guard (then 404 on the unknown task id).
        mock_get_service.return_value.retry_task.side_effect = KeyError("nope")
        response = client.post("/tasks/abc/retry")
    # 404 from the route body, not 401 from the guard — proves the guard passed.
    assert response.status_code == 404


def test_auth_enabled_blocks_write_without_token(client: TestClient, monkeypatch) -> None:
    """With MMRAG_API_TOKEN set, a write endpoint without a token → 401."""
    _with_token_env(monkeypatch, "secret")
    response = client.post("/upload/confirm", json={"edits": []})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") is not None


def test_auth_enabled_blocks_delete_without_token(client: TestClient, monkeypatch) -> None:
    _with_token_env(monkeypatch, "secret")
    response = client.delete("/assets/whatever")
    assert response.status_code == 401


def test_auth_rejects_wrong_token(client: TestClient, monkeypatch) -> None:
    _with_token_env(monkeypatch, "secret")
    response = client.post("/upload/confirm", json={"edits": []}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_auth_accepts_x_api_key_header(client: TestClient, monkeypatch) -> None:
    _with_token_env(monkeypatch, "secret")
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        mock_get_service.return_value.retry_task.side_effect = KeyError("nope")
        response = client.post("/tasks/abc/retry", headers={"X-API-Key": "secret"})
    # Guard passed → 404 from the route body, not 401.
    assert response.status_code == 404


def test_auth_accepts_bearer_header(client: TestClient, monkeypatch) -> None:
    _with_token_env(monkeypatch, "secret")
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        mock_get_service.return_value.retry_task.side_effect = KeyError("nope")
        response = client.post("/tasks/abc/retry", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 404


def test_auth_read_endpoints_stay_open_when_token_set(client: TestClient, monkeypatch) -> None:
    """Read endpoints (/search /answer /assets /tasks) stay open even with a
    token configured — the bundled web UI's same-origin fetches carry no
    Authorization header."""
    _with_token_env(monkeypatch, "secret")
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        mock_get_service.return_value.list_assets.return_value = []
        response = client.get("/assets")
    assert response.status_code == 200


def test_auth_eval_endpoint_guarded(client: TestClient, monkeypatch) -> None:
    _with_token_env(monkeypatch, "secret")
    response = client.post("/eval", json={"top_k": 5})
    assert response.status_code == 401


def test_auth_token_is_case_insensitive_scheme(client: TestClient, monkeypatch) -> None:
    """``Authorization: bearer <t>`` (lowercase scheme) is accepted too."""
    _with_token_env(monkeypatch, "secret")
    with patch("mm_asset_rag.api.get_service") as mock_get_service:
        mock_get_service.return_value.retry_task.side_effect = KeyError("nope")
        response = client.post("/tasks/abc/retry", headers={"Authorization": "bearer secret"})
    assert response.status_code == 404


def test_resolve_trusted_hosts_defaults_to_loopback(monkeypatch) -> None:
    from mm_asset_rag.api import _resolve_trusted_hosts
    from mm_asset_rag.settings import get_settings

    monkeypatch.delenv("MMRAG_TRUSTED_HOSTS", raising=False)
    get_settings.cache_clear()
    hosts = _resolve_trusted_hosts()
    assert "127.0.0.1" in hosts
    assert "localhost" in hosts


def test_resolve_trusted_hosts_configurable(monkeypatch) -> None:
    from mm_asset_rag.api import _resolve_trusted_hosts
    from mm_asset_rag.settings import get_settings

    monkeypatch.setenv("MMRAG_TRUSTED_HOSTS", "rag.example.com, api.example.com")
    get_settings.cache_clear()
    assert _resolve_trusted_hosts() == ["rag.example.com", "api.example.com"]


# ─── Upload / stream safety ──────────────────────────────────────────────


def test_upload_preview_rejects_too_many_files(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    png_bytes: bytes,
) -> None:
    """An upload with more than ``upload_max_files`` files → 413.

    Each previewed file can trigger a VLM auto-meta call, so an unbounded
    file count is a quota-burn vector the byte caps alone don't bound.
    """
    monkeypatch.setenv("UPLOAD_MAX_FILES", "2")
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    files = [("files", (f"img{i}.png", png_bytes, "image/png")) for i in range(3)]
    response = client.post("/upload/preview", files=files)
    assert response.status_code == 413
    assert "upload_max_files" in response.json()["detail"]


def test_upload_preview_body_size_limit_rejects_oversized_batch(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The body-size middleware rejects a POST before Starlette spools it.

    Regression guard for the old behaviour where a 50 GB multipart would
    fill ``/tmp`` before the in-handler byte check fired. The cap is read
    from ``upload_max_batch_bytes`` per-request, so lowering it here makes
    the middleware reject a modestly-sized body.
    """
    monkeypatch.setenv("UPLOAD_MAX_BATCH_BYTES", "1024")  # 1 KiB
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    # A 4 KiB body — well over the 1 KiB cap, under the default file cap.
    big = b"\0" * 4096
    response = client.post(
        "/upload/preview",
        files=[("files", ("big.bin", big, "application/octet-stream"))],
    )
    assert response.status_code == 413


def test_upload_preview_body_size_limit_allows_normal_batch(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    png_bytes: bytes,
) -> None:
    """A normal-sized upload clears the body-size limit (no false positive)."""
    monkeypatch.setenv("UPLOAD_MAX_BATCH_BYTES", str(50 * 1024 * 1024))
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    with patch("mm_asset_rag.auto_meta.auto_meta_image", return_value=None):
        response = client.post(
            "/upload/preview",
            files=[("files", ("scene.png", png_bytes, "image/png"))],
        )
    assert response.status_code == 200, response.text


def test_safe_stream_error_strips_urls_and_caps_length() -> None:
    """``_safe_stream_error`` strips URLs (provider hosts, inlined userinfo)
    and caps the message so a streamed error event leaks no topology."""
    from mm_asset_rag.api import _STREAM_ERR_MAX_CHARS, _safe_stream_error

    # A requests-style error with the full URL — and a userinfo-form URL.
    exc = Exception(
        "404 Client Error: Not Found for url: "
        "https://user:secretpass@10.0.0.5:8080/v1/chat/completions"
    )
    out = _safe_stream_error(exc)
    assert "10.0.0.5" not in out
    assert "secretpass" not in out
    assert "<url>" in out
    assert "404 Client Error" in out

    # Multi-line: only the first line is kept.
    assert "\n" not in _safe_stream_error(Exception("line one\nline two"))

    # Over-cap message is truncated with an ellipsis.
    long_msg = "x" * (_STREAM_ERR_MAX_CHARS + 100)
    truncated = _safe_stream_error(Exception(long_msg))
    assert len(truncated) <= _STREAM_ERR_MAX_CHARS + 1  # +1 for the ellipsis
    assert truncated.endswith("…")

    # Empty message still yields something non-empty.
    assert _safe_stream_error(Exception())
