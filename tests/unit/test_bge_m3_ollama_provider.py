"""Tests for the bge-m3 (ollama OpenAI-compatible) embedding path.

0.4.0 P1 + P2: the user provides an ollama-served ``bge-m3`` model
behind the OpenAI-compatible ``/v1/embeddings`` endpoint. The
existing :class:`TextEmbedder` should hit that endpoint as long as
``EMBEDDING_API_KEY`` / ``EMBEDDING_BASE_URL`` / ``EMBEDDING_MODEL``
are populated. These tests pin the contract and guard against
silent regressions (e.g. someone changing the request shape to a
non-OpenAI schema).
"""

from __future__ import annotations

import pytest
import responses

from mm_asset_rag.embedders.text_embedder import TextEmbedder
from mm_asset_rag.settings import Settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from mm_asset_rag.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _ollama_settings(monkeypatch) -> Settings:
    """Strip OPENAI_* then populate EMBEDDING_* with the ollama bge-m3 triple."""
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "ollama")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    return Settings(_env_file=None)


def test_text_embedder_reads_ollama_bge_m3(monkeypatch) -> None:
    """The provider pulls its creds from the EMBEDDING_* env vars, not OPENAI_*."""
    s = _ollama_settings(monkeypatch)
    emb = TextEmbedder(settings=s)
    assert emb.api_key == "ollama"
    assert emb.base_url == "http://127.0.0.1:11434/v1"
    assert emb.model == "bge-m3"


def test_text_embedder_posts_to_ollama_v1_embeddings(monkeypatch) -> None:
    """``embed_batch`` must POST to ``{base_url}/embeddings`` with
    the OpenAI-compatible request body (model + input array)."""
    s = _ollama_settings(monkeypatch)
    emb = TextEmbedder(settings=s)
    # Disable the per-batch sleep so the test stays under 1 s.
    emb.request_interval = 0.0

    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        rsps.post(
            "http://127.0.0.1:11434/v1/embeddings",
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            },
            status=200,
        )
        out = emb.embed_batch(["hello", "world"])

        # Verify the request shape is OpenAI-compatible (model + input list).
        # Read ``rsps.calls[0]`` *inside* the context manager — the
        # library clears the call list on exit, so accessing it after
        # the ``with`` block raises IndexError.
        request = rsps.calls[0].request
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    import json as _json

    body = _json.loads(request.body)
    assert body == {"model": "bge-m3", "input": ["hello", "world"]}


def test_text_embedder_sends_bearer_auth_header(monkeypatch) -> None:
    """Even with a dummy ``ollama`` key, the provider must still send
    an ``Authorization: Bearer ...`` header — some hosted gateways
    reject the request when the header is missing.
    """
    s = _ollama_settings(monkeypatch)
    emb = TextEmbedder(settings=s)
    emb.request_interval = 0.0
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        rsps.post(
            "http://127.0.0.1:11434/v1/embeddings",
            json={"data": [{"index": 0, "embedding": [0.1]}]},
            status=200,
        )
        emb.embed_batch(["probe"])
        auth = rsps.calls[0].request.headers.get("Authorization", "")
    assert auth.startswith("Bearer ")


def test_text_embedder_dim_matches_bge_m3(monkeypatch) -> None:
    """Sanity: bge-m3 returns 1024-dim vectors. If a deployer swaps
    the model string to something else (e.g. ``nomic-embed-text``)
    and expects a different dim, the active Qdrant collection's
    dim suffix changes — this test guards against accidentally
    hard-coding the wrong dim in the embedder.
    """
    s = _ollama_settings(monkeypatch)
    emb = TextEmbedder(settings=s)
    emb.request_interval = 0.0
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        rsps.post(
            "http://127.0.0.1:11434/v1/embeddings",
            json={
                "data": [
                    {"index": 0, "embedding": [0.0] * 1024},
                ]
            },
            status=200,
        )
        emb.embed("probe")
        # ``dim()`` lazily probes by embedding a probe string (it does not
        # hold a constant), so it issues a *second* HTTP call. Both calls
        # must be inside the mocked scope — outside it, ``dim()`` would hit
        # the real ollama endpoint, which exists locally (test passes by
        # accident) but not on CI (ConnectionRefused).
        assert emb.dim() == 1024
