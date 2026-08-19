"""Tests for ``mm_asset_rag.embedders.cn_clip_embedder.CnClipImageEmbedder``.

The Chinese-CLIP backend is a thin wrapper over ``transformers.ChineseCLIPModel`` +
``ChineseCLIPProcessor`` — neither is guaranteed to be importable in the unit-test
sandbox (torch / transformers are heavy). We therefore mock both at the import
level with ``MagicMock`` + a tiny ``_FakeTensor`` that implements the four ops
``CnClipImageEmbedder`` calls on the returned features (``.norm`` / ``.clamp`` /
``__truediv__`` / ``.__getitem__`` / ``.tolist()``).

Pattern mirrors ``tests/unit/test_image_embedder.py`` (sentence-transformers
backend): same surface, swapped loader.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ── Minimal tensor stand-in ────────────────────────────────────────────────


class _FakeTensor:
    """Stand-in for the torch tensor returned by ``get_*_features``.

    Supports the small set of ops ``CnClipImageEmbedder`` chains on the feature
    tensor: ``.norm(dim, keepdim)`` → scalar tensor of 1.0; ``.clamp(min)`` →
    self; ``__truediv__`` by a scalar tensor; ``__getitem__`` on 2-D for the
    per-row ``tolist()`` call; ``.tolist()`` returns a plain Python list.
    """

    def __init__(self, data: list):
        # ``data`` is 1-D (``list[float]``) or 2-D (``list[list[float]]``).
        self._data = data

    def tolist(self):
        return [list(row) for row in self._data] if self._is_2d() else list(self._data)

    def __getitem__(self, idx):
        rows = self._data
        return _FakeTensor(rows[idx])

    def __truediv__(self, other):
        rows = self._data
        if isinstance(other, _FakeTensor):
            scalar = other._data[0] if not other._is_2d() else 1.0
        else:
            scalar = float(other)
        if self._is_2d():
            return _FakeTensor([[v / scalar for v in row] for row in rows])
        return _FakeTensor([v / scalar for v in rows])

    def norm(self, dim: int = -1, keepdim: bool = False):
        # Pretend the vectors are already unit-length — keeps ``norm.clamp(min)``
        # at 1.0 so the divide in ``embed_text`` / ``embed_image`` is a no-op.
        return _FakeTensor([1.0])

    def clamp(self, min: float | None = None):
        return self

    def _is_2d(self) -> bool:
        return bool(self._data) and isinstance(self._data[0], list)


# ── Fake model + processor ──────────────────────────────────────────────────


class _FakeProcessor:
    """Stand-in for ``ChineseCLIPProcessor``.

    与 ``_FakeModel`` 协同:返回 ``{"_batch_size": N}``,``CnClipImageEmbedder``
    ``**inputs`` 后传给 model,model读出 batch size。真实 processor 返回
    ``input_ids`` / ``attention_mask`` / ``pixel_values``;此处不模拟,
    因为被测代码目前只 ``**inputs`` 透传给 model,未读具体 key。若将来
    embedder 直接 ``inputs["input_ids"]``,此 fake 会失效。
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, text=None, images=None, **kwargs):
        self.calls.append({"text": text, "images": images, **kwargs})
        if text is not None:
            return {"_batch_size": len(text)}
        if images is not None:
            n = len(images) if isinstance(images, list) else 1
            return {"_batch_size": n}
        return {}


class _FakeModel:
    """Stand-in for ``ChineseCLIPModel``. Reads ``_batch_size`` from the inputs
    dict so the returned feature tensor has the expected leading dim."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.eval_called = False
        self.text_feature_calls = 0
        self.image_feature_calls = 0

    def eval(self) -> None:
        self.eval_called = True

    def get_text_features(self, **kwargs):
        self.text_feature_calls += 1
        n = kwargs.get("_batch_size", 1)
        return _FakeTensor([[0.1 * (j + 1) for j in range(self.dim)] for _ in range(n)])

    def get_image_features(self, **kwargs):
        self.image_feature_calls += 1
        n = kwargs.get("_batch_size", 1)
        return _FakeTensor([[0.2 * (j + 1) for j in range(self.dim)] for _ in range(n)])


# ── Fixture ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_cn_clip_embedder(monkeypatch):
    """Build a ``CnClipImageEmbedder`` with the model + processor pre-loaded.

    Bypasses ``_load()`` (which would try to import ``transformers`` and pull a
    real Chinese-CLIP from HF Hub) by injecting fakes directly. The constructor
    still calls ``_check_available()`` so we patch ``is_available`` to return
    ``True``.
    """
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder

    monkeypatch.setattr(CnClipImageEmbedder, "is_available", staticmethod(lambda: True))

    emb = CnClipImageEmbedder.__new__(CnClipImageEmbedder)
    emb.model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    emb._model = _FakeModel(dim=4)
    emb._processor = _FakeProcessor()
    emb._dim = 4
    return emb


# ── Tests ───────────────────────────────────────────────────────────────────


def test_constructor_uses_settings_clip_model(monkeypatch) -> None:
    """``CnClipImageEmbedder()`` 默认读 ``Settings.clip_model``。"""
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(CnClipImageEmbedder, "is_available", staticmethod(lambda: True))
    settings = get_settings()
    monkeypatch.setattr(settings, "clip_model", "OFA-Sys/chinese-clip-vit-base-patch16")

    emb = CnClipImageEmbedder()
    assert emb.model_name == "OFA-Sys/chinese-clip-vit-base-patch16"
    assert emb.modality == "image"
    # lazy: 不调 _load() 就不应有 model / dim
    assert emb._model is None
    assert emb._dim is None


def test_constructor_fails_fast_when_transformers_missing(monkeypatch) -> None:
    """``CnClipImageUnavailable`` 必须构造时就抛,不在 _load() 之后才报。"""
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder, CnClipImageUnavailable

    monkeypatch.setattr(CnClipImageEmbedder, "is_available", staticmethod(lambda: False))
    with pytest.raises(CnClipImageUnavailable) as excinfo:
        CnClipImageEmbedder()
    msg = str(excinfo.value)
    assert "transformers" in msg
    assert "ChineseCLIPModel" in msg


def test_dim_returns_cached_value(fake_cn_clip_embedder) -> None:
    assert fake_cn_clip_embedder.dim() == 4
    # 第二次调用走缓存,不重新算
    assert fake_cn_clip_embedder.dim() == 4
    assert fake_cn_clip_embedder._model.text_feature_calls == 0


def test_dim_settings_override_skips_probe(fake_cn_clip_embedder, monkeypatch) -> None:
    """``Settings.image_embedding_dim`` 覆盖 probe,跳过模型调用。"""
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(get_settings(), "image_embedding_dim", 768)
    fake_cn_clip_embedder._dim = None
    # 让 model 在 probe 时抛:如果走到 probe,会被 catch 成 image unencodable
    # 但更直接的证据是 _model.text_feature_calls 没变
    before = fake_cn_clip_embedder._model.text_feature_calls
    assert fake_cn_clip_embedder.dim() == 768
    assert fake_cn_clip_embedder._model.text_feature_calls == before


def test_dim_settings_override_beats_cached_value(fake_cn_clip_embedder, monkeypatch) -> None:
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(get_settings(), "image_embedding_dim", 1024)
    fake_cn_clip_embedder._dim = 768
    assert fake_cn_clip_embedder.dim() == 1024


def test_dim_falls_back_to_probe_when_settings_unset(fake_cn_clip_embedder, monkeypatch) -> None:
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(get_settings(), "image_embedding_dim", None)
    fake_cn_clip_embedder._dim = None
    assert fake_cn_clip_embedder.dim() == 4


def test_embed_text_calls_chinese_clip_text_encoder(fake_cn_clip_embedder) -> None:
    out = fake_cn_clip_embedder.embed_text("海边的日落")
    assert isinstance(out, list)
    assert len(out) == 4
    assert all(isinstance(v, float) for v in out)
    assert fake_cn_clip_embedder._model.text_feature_calls == 1


def test_embed_image_calls_chinese_clip_image_encoder(
    fake_cn_clip_embedder, tmp_path: Path
) -> None:
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 128, 0)).save(p)

    out = fake_cn_clip_embedder.embed_image(p)
    assert isinstance(out, list)
    assert len(out) == 4
    assert fake_cn_clip_embedder._model.image_feature_calls == 1


def test_embed_text_batch_collapses_to_one_call(fake_cn_clip_embedder) -> None:
    texts = ["海", "日落", "沙滩"]
    out = fake_cn_clip_embedder.embed_text_batch(texts)
    assert len(out) == 3
    assert all(len(v) == 4 for v in out)
    # Processor + model 各被调用一次,不再每条都跑
    assert len(fake_cn_clip_embedder._processor.calls) == 1
    assert fake_cn_clip_embedder._model.text_feature_calls == 1


def test_embed_text_batch_empty_returns_empty(fake_cn_clip_embedder) -> None:
    assert fake_cn_clip_embedder.embed_text_batch([]) == []
    assert fake_cn_clip_embedder._model.text_feature_calls == 0


def test_embed_image_batch_collapses_to_one_call(fake_cn_clip_embedder, tmp_path: Path) -> None:
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (8, 8), color=(i * 30, i * 60, i * 90)).save(p)
        paths.append(p)

    out = fake_cn_clip_embedder.embed_image_batch(paths)
    assert len(out) == 3
    assert len(fake_cn_clip_embedder._processor.calls) == 1
    assert fake_cn_clip_embedder._model.image_feature_calls == 1


def test_embed_image_batch_empty_returns_empty(fake_cn_clip_embedder) -> None:
    assert fake_cn_clip_embedder.embed_image_batch([]) == []
    assert fake_cn_clip_embedder._model.image_feature_calls == 0


def test_embed_image_returns_none_for_non_image_file(fake_cn_clip_embedder, tmp_path: Path) -> None:
    """``embed_image`` 对非图片文件返回 ``None``,不抛 — Protocol 契约要求。"""
    bad = tmp_path / "not_image.txt"
    bad.write_bytes(b"this is not a PNG")
    assert fake_cn_clip_embedder.embed_image(bad) is None
    assert fake_cn_clip_embedder._model.image_feature_calls == 0


def test_embed_image_returns_none_for_missing_file(fake_cn_clip_embedder, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.png"
    assert fake_cn_clip_embedder.embed_image(missing) is None


def test_embed_image_batch_marks_invalid_files_as_none(
    fake_cn_clip_embedder, tmp_path: Path
) -> None:
    """``embed_image_batch`` 跳过不可识别文件,在其槽位放 ``None``。"""
    from PIL import Image

    good = tmp_path / "good.png"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(good)
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"not an image")

    out = fake_cn_clip_embedder.embed_image_batch([bad, good, bad])
    assert len(out) == 3
    assert out[0] is None
    assert isinstance(out[1], list)
    assert len(out[1]) == 4
    assert out[2] is None
    # 只 encode 一次(合法的 image)
    assert fake_cn_clip_embedder._model.image_feature_calls == 1


def test_embed_image_batch_all_invalid_returns_all_none(
    fake_cn_clip_embedder, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"not an image")
    out = fake_cn_clip_embedder.embed_image_batch([bad, bad])
    assert out == [None, None]
    assert fake_cn_clip_embedder._model.image_feature_calls == 0


def test_embed_batch_routes_mixed_text_and_images(fake_cn_clip_embedder, tmp_path: Path) -> None:
    """``embed_batch`` 按类型分流到 text/image batch。"""
    from PIL import Image

    p1 = tmp_path / "img1.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(p1)
    p2 = tmp_path / "img2.png"
    Image.new("RGB", (4, 4), color=(40, 50, 60)).save(p2)

    contents = ["日落", p1, "海浪", p2]
    out = fake_cn_clip_embedder.embed_batch(contents)
    assert len(out) == 4
    assert all(len(v) == 4 for v in out)
    # 一次 text batch + 一次 image batch
    assert fake_cn_clip_embedder._model.text_feature_calls == 1
    assert fake_cn_clip_embedder._model.image_feature_calls == 1


def test_embed_batch_empty_returns_empty(fake_cn_clip_embedder) -> None:
    """``embed_batch([])`` 短路返回,不打 model。"""
    assert fake_cn_clip_embedder.embed_batch([]) == []
    assert fake_cn_clip_embedder._model.text_feature_calls == 0
    assert fake_cn_clip_embedder._model.image_feature_calls == 0


def test_embed_batch_unknown_type_falls_back_to_single_embed(
    fake_cn_clip_embedder, tmp_path: Path
) -> None:
    """``embed_batch`` 遇非 str/Path 类型时逐条 ``embed``(现状会让 ``Path(int)`` 抛 TypeError)。"""
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(p)
    # int 既不是 str 也不是 Path,``embed_batch`` 走 ``return [self.embed(c) for c in contents]``;
    # ``embed(int)`` 里 ``Path(int)`` 会抛 TypeError — 锁定现状, 未来若改成静默 [] 会断此测。
    contents: list[object] = ["text", 42, p]
    with pytest.raises(TypeError):
        fake_cn_clip_embedder.embed_batch(contents)


def test_embed_dispatches_text_and_path() -> None:
    """``embed(content)`` 把 str → text、Path → image,镜像 ``ImageEmbedder.embed``。"""
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder

    emb = CnClipImageEmbedder.__new__(CnClipImageEmbedder)
    emb.model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    emb._model = _FakeModel(dim=4)
    emb._processor = _FakeProcessor()
    emb._dim = 4

    out_str = emb.embed("海边的日落")
    assert len(out_str) == 4
    assert emb._model.text_feature_calls == 1

    import tempfile

    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"
        Image.new("RGB", (4, 4), color=(1, 2, 3)).save(p)
        out_path = emb.embed(p)
        assert len(out_path) == 4
        assert emb._model.image_feature_calls == 1


def test_load_calls_eval_on_model(monkeypatch) -> None:
    """``_load()`` 触发 ``model.eval()`` — 推理模式。"""
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder

    monkeypatch.setattr(CnClipImageEmbedder, "is_available", staticmethod(lambda: True))

    emb = CnClipImageEmbedder.__new__(CnClipImageEmbedder)
    emb.model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    emb._model = None
    emb._processor = None
    emb._dim = None

    fake_model = _FakeModel(dim=4)
    fake_processor = _FakeProcessor()

    with (
        patch(
            "transformers.ChineseCLIPModel.from_pretrained",
            return_value=fake_model,
        ),
        patch(
            "transformers.ChineseCLIPProcessor.from_pretrained",
            return_value=fake_processor,
        ),
    ):
        m, p = emb._load()

    assert m is fake_model
    assert p is fake_processor
    assert fake_model.eval_called is True


# ── Dispatch (factory) ─────────────────────────────────────────────────────


def test_build_default_image_embedder_returns_cn_clip_for_cn_clip_provider(monkeypatch) -> None:
    """``image_provider == "cn_clip"`` 时工厂返回 ``CnClipImageEmbedder``。"""
    from mm_asset_rag.embedders import build_default_image_embedder
    from mm_asset_rag.embedders.cn_clip_embedder import CnClipImageEmbedder
    from mm_asset_rag.settings import get_settings

    monkeypatch.setattr(CnClipImageEmbedder, "is_available", staticmethod(lambda: True))
    monkeypatch.setattr(get_settings(), "image_provider", "cn_clip")

    emb = build_default_image_embedder()
    assert isinstance(emb, CnClipImageEmbedder)


def test_build_default_image_embedder_returns_sentence_transformers_by_default(monkeypatch) -> None:
    """``image_provider == "lite"``(默认)工厂仍返回 ``ImageEmbedder``。"""
    from mm_asset_rag.embedders import build_default_image_embedder
    from mm_asset_rag.embedders.image_embedder import ImageEmbedder
    from mm_asset_rag.settings import get_settings

    # 隔离 ImageEmbedder 构造时的 [clip] 检查 — 我们只测 dispatch,不动 model
    monkeypatch.setattr(ImageEmbedder, "_check_available", staticmethod(lambda: None))
    monkeypatch.setattr(get_settings(), "image_provider", "lite")

    emb = build_default_image_embedder()
    assert isinstance(emb, ImageEmbedder)
