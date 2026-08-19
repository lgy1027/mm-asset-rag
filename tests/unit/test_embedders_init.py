"""Tests for the image / text embedder factory + registry dispatch in
``mm_asset_rag.embedders``.

The factory + ``_ensure_image_registered`` paths are the production glue
between ``Settings.image_provider`` and the actual embedder implementation;
covering them keeps the dispatch contract honest when the embedder roster
changes (e.g. adding ``cn_clip``).
"""

from __future__ import annotations

import pytest


def test_build_default_image_embedder_picks_cn_clip(monkeypatch) -> None:
    """``IMAGE_PROVIDER=cn_clip`` → ``CnClipImageEmbedder`` instance。"""
    from mm_asset_rag.embedders import build_default_image_embedder
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(CnClipImageEmbedder, "is_available", staticmethod(lambda: True))
    monkeypatch.setattr(get_settings(), "image_provider", "cn_clip")

    emb = build_default_image_embedder()
    assert isinstance(emb, CnClipImageEmbedder)


def test_build_default_image_embedder_picks_image_embedder_by_default(monkeypatch) -> None:
    """默认 ``lite`` / ``sentence_transformers`` → ``ImageEmbedder`` instance。"""
    from mm_asset_rag.embedders import build_default_image_embedder
    from mm_asset_rag.embedders.image_embedder import ImageEmbedder
    from mm_asset_rag.settings import get_settings

    # [clip] 检查注入绕过
    monkeypatch.setattr(ImageEmbedder, "_check_available", staticmethod(lambda: None))

    for provider in ("lite", "sentence_transformers"):
        monkeypatch.setattr(get_settings(), "image_provider", provider)
        emb = build_default_image_embedder()
        assert isinstance(emb, ImageEmbedder)


def test_build_default_image_embedder_missing_cn_clip_deps_raises(monkeypatch) -> None:
    """cn_clip 缺 ``transformers`` 时构造抛 ``CnClipImageUnavailable`` — 不静默吞。"""
    from mm_asset_rag.embedders import build_default_image_embedder
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(CnClipImageEmbedder, "is_available", staticmethod(lambda: False))
    monkeypatch.setattr(get_settings(), "image_provider", "cn_clip")

    with pytest.raises(Exception) as excinfo:
        build_default_image_embedder()
    # 不需要严格匹配具体异常类型,关键是工厂不静默返回。
    msg = str(excinfo.value).lower()
    assert "cn_clip" in msg or "transformers" in msg


def test_ensure_image_registered_swallows_unavailable(monkeypatch, capsys) -> None:
    """``_ensure_image_registered`` 在 image embedder 不可用时 print 原因,不崩。

    直接 ``patch.object`` ``build_default_image_embedder`` 让其抛
    ``ImageEmbeddingUnavailable``,绕开 ``_load()`` 触发的实际 model 加载。
    """
    from mm_asset_rag.embedders import _ensure_image_registered
    from mm_asset_rag.embedders.image_embedder import ImageEmbeddingUnavailable
    from mm_asset_rag.registry import embedders as _embedders
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(get_settings(), "image_provider", "lite")

    # 清空 registry 让 ensure 真正跑(没有 unregister,直接操作 _items)
    _embedders._items.clear()

    def _raise_unavailable() -> None:
        raise ImageEmbeddingUnavailable("[clip] not installed")

    monkeypatch.setattr(
        "mm_asset_rag.embedders.build_default_image_embedder", _raise_unavailable
    )

    _ensure_image_registered()
    captured = capsys.readouterr()
    assert "default image embedder not registered" in captured.out