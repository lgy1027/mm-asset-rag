"""Tests for tier-3 multimodal answer (``answer_with_images``).

Covers the three contracts the feature depends on:
1. ``_user_content`` returns plain text when the toggle is off, and a
   ``[text, image_url...]`` content-parts list when on + images exist.
2. ``_read_image_data_url`` honours the path-traversal guard — a crafted
   ``path`` (``../../tasks.jsonl``) yields ``None`` instead of reading
   the file; a real image yields a ``data:image/...;base64,...`` URL.
3. ``llm_answer`` degrades to a text-only retry when the image-bearing
   request fails (non-vision model rejects ``image_url`` parts).
"""

from __future__ import annotations

from unittest.mock import patch

from mm_asset_rag.answer import (
    _read_image_data_url,
    _user_content,
    llm_answer,
)
from mm_asset_rag.schema import SearchHit
from mm_asset_rag.settings import get_settings


def _hit_with_images(asset_id: str, images: list) -> SearchHit:
    return SearchHit(
        route="text",
        score=0.9,
        asset_id=asset_id,
        title=asset_id,
        source_type="pdf",
        source_path=f"{asset_id}.pdf",
        evidence="evidence text",
        metadata={"page": 1, "images": images},
        images=images,
    )


def test_user_content_text_when_toggle_off(tmp_home, monkeypatch) -> None:
    monkeypatch.delenv("ANSWER_WITH_IMAGES", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    content = _user_content(
        "Q?", "ctx", [_hit_with_images("a", [{"path": "images/x.png"}])], settings
    )
    assert isinstance(content, str)
    assert "Q?" in content


def test_user_content_parts_when_toggle_on(tmp_home, monkeypatch) -> None:
    monkeypatch.setenv("ANSWER_WITH_IMAGES", "true")
    get_settings.cache_clear()
    settings = get_settings()
    # Patch the image reader to return a fake data URL so no real file is needed.
    with patch(
        "mm_asset_rag.answer._read_image_data_url",
        return_value="data:image/png;base64,AAAA",
    ):
        content = _user_content(
            "Q?", "ctx", [_hit_with_images("a", [{"path": "images/x.png"}])], settings
        )
    assert isinstance(content, list)
    # First part is the text, remaining are image_url parts.
    assert content[0]["type"] == "text"
    assert "Q?" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_user_content_text_when_no_images(tmp_home, monkeypatch) -> None:
    """Toggle on but hit has no images → plain text (no empty parts list)."""
    monkeypatch.setenv("ANSWER_WITH_IMAGES", "true")
    get_settings.cache_clear()
    settings = get_settings()
    content = _user_content("Q?", "ctx", [_hit_with_images("a", [])], settings)
    assert isinstance(content, str)


def test_user_content_caps_per_hit_and_total(tmp_home, monkeypatch) -> None:
    monkeypatch.setenv("ANSWER_WITH_IMAGES", "true")
    monkeypatch.setenv("ANSWER_IMAGE_MAX_PER_HIT", "2")
    get_settings.cache_clear()
    settings = get_settings()
    # 3 hits x 5 images each -> per-hit cap 2, global cap 4 -> 4 image parts.
    hits = [
        _hit_with_images(f"a{i}", [{"path": f"images/im{j}.png"} for j in range(5)])
        for i in range(3)
    ]
    with patch(
        "mm_asset_rag.answer._read_image_data_url",
        side_effect=lambda _aid, p: f"data:image/png;base64,{p}",
    ):
        content = _user_content("Q?", "ctx", hits, settings)
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) == 4  # global cap


def test_read_image_data_url_rejects_traversal(tmp_home) -> None:
    # Write a fake non-image file outside images/ to prove the guard blocks it.
    secret = tmp_home / "documents.jsonl"
    secret.write_text("secret")
    # A payload path that tries to escape via ".." — basename is "documents.jsonl"
    # but safe_parsed_image_path rejects: wrong suffix AND escapes images dir.
    assert _read_image_data_url("aid", "images/../../documents.jsonl") is None


def test_read_image_data_url_returns_data_url_for_real_image(tmp_home) -> None:
    import os

    aid = "asset1"
    img_dir = tmp_home / "parsed" / aid / "images"
    img_dir.mkdir(parents=True)
    png = img_dir / "p0_i0.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    url = _read_image_data_url(aid, "images/p0_i0.png")
    assert url is not None
    assert url.startswith("data:image/png;base64,")
    assert os.environ.get("MM_ASSET_RAG_HOME") == str(tmp_home)


def test_llm_answer_degrades_to_text_when_image_request_fails(tmp_home, monkeypatch) -> None:
    """Image-bearing request fails → retry text-only → success."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    monkeypatch.setenv("ANSWER_WITH_IMAGES", "true")
    get_settings.cache_clear()

    calls = {"n": 0}

    class FakeResp:
        def __init__(self, ok: bool) -> None:
            self._ok = ok

        def raise_for_status(self) -> None:
            if not self._ok:
                raise RuntimeError("model does not support images")

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "text-only answer"}}]}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        # First call (with image parts) fails; second (text) succeeds.
        return FakeResp(ok=(calls["n"] == 2))

    with (
        patch(
            "mm_asset_rag.answer._read_image_data_url", return_value="data:image/png;base64,AAAA"
        ),
        patch("mm_asset_rag.answer._post_chat", side_effect=fake_post),
    ):
        result = llm_answer("Q?", [_hit_with_images("a", [{"path": "images/x.png"}])])
    assert calls["n"] == 2  # image attempt + text retry
    assert result["answer"] == "text-only answer"


def test_llm_answer_no_retry_when_text_only_fails(tmp_home, monkeypatch) -> None:
    """When the toggle is off (text-only), a failure must propagate — no retry."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    get_settings.cache_clear()

    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("network down")

    with (
        patch("mm_asset_rag.answer._post_chat", side_effect=fake_post),
    ):
        try:
            llm_answer("Q?", [_hit_with_images("a", [])])
            raised = False
        except RuntimeError:
            raised = True
    assert raised
    assert calls["n"] == 1  # no retry when text-only
