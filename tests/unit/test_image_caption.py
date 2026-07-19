"""Tests for ``mm_asset_rag.image_caption``.

Covers the contracts the parse path relies on:
1. ``enrich_docs_with_image_captions`` appends a VLM caption to a chunk's
   text and stamps ``metadata["images"][*]["caption"]`` so the figure's
   semantics enter the text index and the answer layer can cite it.
2. Captions are cached under ``captions/<asset_id>.jsonl`` so a second pass
   reuses them without re-calling the VLM (figure bytes are stable).
3. Degradation: no-op when ``image_caption_enabled`` is false, when VLM
   creds are unset, or when every VLM call fails — chunk text is untouched.
4. A figure that already carries a caption (PyMuPDF "图N: …") is preserved
   and reused, not overwritten.

The VLM is mocked throughout — no real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mm_asset_rag.image_caption import enrich_docs_with_image_captions
from mm_asset_rag.schema import ParsedDocument


def _doc(text: str, images: list[dict] | None = None) -> ParsedDocument:
    return ParsedDocument(
        text=text,
        metadata={"asset_id": "a1", "chunk_index": 0, "images": images or []},
    )


def _enable(monkeypatch, *, enabled: bool = True, creds=("http://vlm/v1", "sk", "vlm-model")):
    """Flip the caption switch on (or off) and set VLM creds on the cached Settings.

    ``vlm_creds`` is a read-only pydantic property backed by the ``vlm_*``
    fields, so creds are injected by setting those fields directly (not the
    property). The flag is a plain field too. When ``creds`` is the
    "unconfigured" triple we also clear the ``openai_*`` fallback fields —
    ``vlm_creds`` falls back to ``OPENAI_*``, so on a host whose real .env
    has OPENAI_* set, leaving them alone would let the "unconfigured" case
    still look configured (and the test would not be a no-op).
    """
    from mm_asset_rag.settings import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "image_caption_enabled", enabled)
    if creds == (None, None, None):
        monkeypatch.setattr(s, "vlm_base_url", None)
        monkeypatch.setattr(s, "vlm_api_key", None)
        monkeypatch.setattr(s, "vlm_model", None)
        # Clear the OPENAI_* fallback too, or vlm_creds still resolves.
        monkeypatch.setattr(s, "openai_base_url", None)
        monkeypatch.setattr(s, "openai_api_key", None)
        monkeypatch.setattr(s, "openai_model", None)
    else:
        monkeypatch.setattr(s, "vlm_base_url", creds[0])
        monkeypatch.setattr(s, "vlm_api_key", creds[1])
        monkeypatch.setattr(s, "vlm_model", creds[2])
    return s


def _seed_image(tmp_home: Path, asset_id: str, rel: str) -> Path:
    """Write a 1x1 png under parsed/<asset_id>/<rel> so _image_abs_path resolves."""
    abs_path = tmp_home / ".mm_asset_rag" / "parsed" / asset_id / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid PNG header is enough — call_vlm_caption is mocked, the
    # file only needs to exist.
    abs_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return abs_path


def test_caption_appended_to_text_and_metadata(tmp_home, monkeypatch):
    """A figure with no caption gets a VLM description in text + metadata."""
    _seed_image(tmp_home, "a1", "images/fig1.png")
    docs = [_doc("正文内容", images=[{"path": "images/fig1.png", "caption": ""}])]
    cache = tmp_home / ".mm_asset_rag" / "captions" / "a1.jsonl"

    with patch(
        "mm_asset_rag.image_caption._caption_one",
        return_value="一张双碳目标路线图",
    ):
        _enable(monkeypatch)
        enrich_docs_with_image_captions(docs, asset_id="a1", cache_path=cache)

    assert "图片描述: 一张双碳目标路线图" in docs[0].text
    assert docs[0].metadata["images"][0]["caption"] == "一张双碳目标路线图"
    # Cache persisted for reuse.
    assert cache.exists()
    cached = {
        json.loads(line)["path"]: json.loads(line)["caption"]
        for line in cache.read_text().splitlines()
    }
    assert cached["images/fig1.png"] == "一张双碳目标路线图"


def test_cache_reused_no_second_vlm_call(tmp_home, monkeypatch):
    """A second pass reuses the cached caption — the VLM is not called again."""
    _seed_image(tmp_home, "a1", "images/fig1.png")
    cache = tmp_home / ".mm_asset_rag" / "captions" / "a1.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"path": "images/fig1.png", "caption": "缓存描述"}) + "\n")

    docs = [_doc("正文", images=[{"path": "images/fig1.png", "caption": ""}])]
    calls = {"n": 0}

    def _fake_caption(asset_id, path):
        calls["n"] += 1
        return "should-not-be-used"

    with patch("mm_asset_rag.image_caption._caption_one", side_effect=_fake_caption):
        _enable(monkeypatch)
        enrich_docs_with_image_captions(docs, asset_id="a1", cache_path=cache)

    assert calls["n"] == 0  # cache hit, no VLM call
    assert "缓存描述" in docs[0].text
    assert docs[0].metadata["images"][0]["caption"] == "缓存描述"


def test_noop_when_disabled(tmp_home, monkeypatch):
    """image_caption_enabled=false → text and metadata untouched, no VLM call."""
    _seed_image(tmp_home, "a1", "images/fig1.png")
    docs = [_doc("正文", images=[{"path": "images/fig1.png", "caption": ""}])]

    with patch("mm_asset_rag.image_caption._caption_one") as m:
        _enable(monkeypatch, enabled=False)
        enrich_docs_with_image_captions(docs, asset_id="a1", cache_path=tmp_home / "c.jsonl")
        m.assert_not_called()
    assert "图片描述" not in docs[0].text
    assert docs[0].metadata["images"][0]["caption"] == ""


def test_noop_when_vlm_unconfigured(tmp_home, monkeypatch):
    """No VLM creds → enrich is a no-op even when the flag is on."""
    _seed_image(tmp_home, "a1", "images/fig1.png")
    docs = [_doc("正文", images=[{"path": "images/fig1.png", "caption": ""}])]

    with patch("mm_asset_rag.image_caption._caption_one") as m:
        _enable(monkeypatch, creds=(None, None, None))
        enrich_docs_with_image_captions(docs, asset_id="a1", cache_path=tmp_home / "c.jsonl")
        m.assert_not_called()
    assert "图片描述" not in docs[0].text


def test_vlm_failure_leaves_chunk_untouched(tmp_home, monkeypatch):
    """A VLM call returning empty degrades gracefully — text unchanged."""
    _seed_image(tmp_home, "a1", "images/fig1.png")
    docs = [_doc("正文", images=[{"path": "images/fig1.png", "caption": ""}])]

    with patch("mm_asset_rag.image_caption._caption_one", return_value=""):
        _enable(monkeypatch)
        enrich_docs_with_image_captions(docs, asset_id="a1", cache_path=tmp_home / "c.jsonl")
    assert "图片描述" not in docs[0].text
    assert docs[0].metadata["images"][0]["caption"] == ""


def test_existing_caption_preserved_and_reused(tmp_home, monkeypatch):
    """A figure that already has a caption (PyMuPDF "图N: …") is reused, not
    overwritten, and its caption also enters the text index."""
    _seed_image(tmp_home, "a1", "images/fig1.png")
    docs = [_doc("正文", images=[{"path": "images/fig1.png", "caption": "图1: 架构图"}])]

    with patch("mm_asset_rag.image_caption._caption_one") as m:
        _enable(monkeypatch)
        enrich_docs_with_image_captions(docs, asset_id="a1", cache_path=tmp_home / "c.jsonl")
        m.assert_not_called()  # existing caption → no VLM call
    assert "图片描述: 图1: 架构图" in docs[0].text
    assert docs[0].metadata["images"][0]["caption"] == "图1: 架构图"


def test_multiple_figures_in_one_chunk_deduped(tmp_home, monkeypatch):
    """Two distinct figures in one chunk each append their caption once."""
    _seed_image(tmp_home, "a1", "images/fig1.png")
    _seed_image(tmp_home, "a1", "images/fig2.png")
    docs = [
        _doc(
            "正文",
            images=[
                {"path": "images/fig1.png", "caption": ""},
                {"path": "images/fig2.png", "caption": ""},
            ],
        )
    ]
    captions = {"images/fig1.png": "路线图", "images/fig2.png": "产品矩阵"}

    def _fake(asset_id, path):
        return captions[path]

    with patch("mm_asset_rag.image_caption._caption_one", side_effect=_fake):
        _enable(monkeypatch)
        enrich_docs_with_image_captions(docs, asset_id="a1", cache_path=tmp_home / "c.jsonl")
    assert "路线图" in docs[0].text
    assert "产品矩阵" in docs[0].text
    assert docs[0].text.count("图片描述:") == 1  # single appended block


def test_image_abs_path_rejects_traversal(tmp_home) -> None:
    """``_image_abs_path`` confines the resolved path to ``parsed/<asset_id>/`` —
    absolute paths, ``..`` traversal, and symlinks pointing outside are all
    refused (returns None) even though the path may technically exist."""
    from mm_asset_rag.image_caption import _image_abs_path
    from mm_asset_rag.paths import get_parsed_dir

    # Write a legitimate image at the real parsed dir (get_parsed_dir resolves
    # MM_ASSET_RAG_HOME, which tmp_home points at a fresh tmp dir).
    legit = get_parsed_dir() / "a1" / "images" / "fig1.png"
    legit.parent.mkdir(parents=True, exist_ok=True)
    legit.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert _image_abs_path("a1", "images/fig1.png") is not None
    # Absolute path escapes the base dir.
    assert _image_abs_path("a1", "/etc/passwd") is None
    # Parent traversal escapes the base dir.
    assert _image_abs_path("a1", "../other_asset/secret.png") is None
    assert _image_abs_path("a1", "images/../../secret.png") is None
