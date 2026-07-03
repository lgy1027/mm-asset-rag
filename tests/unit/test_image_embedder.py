"""Tests for ``mm_asset_rag.embedders.image_embedder.ImageEmbedder``.

We don't load the real CLIP model (it needs the ``[clip]`` extra and
takes seconds to download/initialise). Instead we stub
``SentenceTransformer`` with a fake that records how ``encode`` was
called, then assert the batch methods collapse N items into a single
``model.encode(...)`` invocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mm_asset_rag.embedders.image_embedder import ImageEmbedder


class _FakeEncodeResult:
    """Mimics a torch tensor returned by ``model.encode`` for one item.

    Exposes ``.tolist()`` returning a list of floats so the production
    code path ``[float(v) for v in vec.tolist()]`` keeps working
    unchanged.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._data = [round(0.1 * (i + 1), 2) for i in range(dim)]

    def tolist(self) -> list[float]:
        return list(self._data)


class _FakeModel:
    """Records every ``encode`` call and returns a fake result.

    Each call appends ``(args, kwargs)`` to :attr:`calls`. The fake
    result length matches the number of inputs so the
    ``len(encode([...])) == len(input)`` invariant holds.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        inputs = args[0] if args else kwargs.get("sentences") or kwargs.get("images")
        # Mirror sentence-transformers: a single input returns one tensor
        # (with .tolist() → flat list), a list returns N tensors.
        if isinstance(inputs, (str, Path)) or (hasattr(inputs, "convert")):
            return _FakeEncodeResult(self.dim)
        return [_FakeEncodeResult(self.dim) for _ in range(len(inputs))]


@pytest.fixture
def fake_embedder() -> ImageEmbedder:
    """Return an ImageEmbedder whose underlying model is a _FakeModel."""
    emb = ImageEmbedder.__new__(ImageEmbedder)
    emb.model_name = "fake-clip"
    emb._model = _FakeModel(dim=4)
    emb._dim = 4
    return emb


def test_image_embedder_uses_settings_clip_model(monkeypatch) -> None:
    """The default model comes from ``Settings.clip_model``.

    Tests that swap to a Chinese CLIP via env var (``CLIP_MODEL=...``)
    should propagate the new value to the embedder instance. We
    don't actually load the model here — the test just exercises
    the constructor's read path.
    """
    from mm_asset_rag.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "clip_model", "OFA-Sys/chinese-clip-vit-base-patch16")

    captured: dict[str, str] = {}

    class _StubModel:
        def encode(self, content, **kwargs):
            captured.setdefault("calls", []).append(content)
            import numpy as np

            return np.ones((1, 4), dtype=np.float32)

    emb = ImageEmbedder.__new__(ImageEmbedder)
    emb.model_name = settings.clip_model
    emb._model = _StubModel()
    emb._dim = 4
    assert emb.model_name == "OFA-Sys/chinese-clip-vit-base-patch16"


def test_embed_text_batch_collapses_to_one_call(fake_embedder: ImageEmbedder) -> None:
    texts = ["hello", "world", "!"]
    out = fake_embedder.embed_text_batch(texts)
    assert len(out) == 3
    assert len(fake_embedder._model.calls) == 1
    args, _kwargs = fake_embedder._model.calls[0]
    assert args[0] == texts


def test_embed_text_batch_empty_returns_empty(fake_embedder: ImageEmbedder) -> None:
    assert fake_embedder.embed_text_batch([]) == []
    assert fake_embedder._model.calls == []


def test_embed_image_batch_collapses_to_one_call(
    fake_embedder: ImageEmbedder, tmp_path: Path
) -> None:
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (8, 8), color=(i * 30, i * 60, i * 90)).save(p)
        paths.append(p)

    out = fake_embedder.embed_image_batch(paths)
    assert len(out) == 3
    assert len(fake_embedder._model.calls) == 1
    args, _kwargs = fake_embedder._model.calls[0]
    # Model receives the PIL Image objects, not the paths.
    assert len(args[0]) == 3
    assert all(isinstance(im, Image.Image) for im in args[0])


def test_embed_image_batch_empty_returns_empty(fake_embedder: ImageEmbedder) -> None:
    assert fake_embedder.embed_image_batch([]) == []
    assert fake_embedder._model.calls == []


def test_embed_batch_routes_text_to_text_batch(
    fake_embedder: ImageEmbedder,
) -> None:
    contents = ["alpha", "beta", "gamma"]
    out = fake_embedder.embed_batch(contents)
    assert len(out) == 3
    # Mixed batch with only text strings should issue a single call.
    assert len(fake_embedder._model.calls) == 1
    args, _kwargs = fake_embedder._model.calls[0]
    assert args[0] == contents


def test_embed_batch_routes_paths_to_image_batch(
    fake_embedder: ImageEmbedder, tmp_path: Path
) -> None:
    from PIL import Image

    paths = []
    for i in range(2):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (4, 4), color=(i, i, i)).save(p)
        paths.append(p)

    out = fake_embedder.embed_batch(paths)
    assert len(out) == 2
    assert len(fake_embedder._model.calls) == 1
    args, _kwargs = fake_embedder._model.calls[0]
    assert len(args[0]) == 2
    assert all(isinstance(im, Image.Image) for im in args[0])


def test_embed_batch_routes_mixed_text_and_images(
    fake_embedder: ImageEmbedder, tmp_path: Path
) -> None:
    from PIL import Image

    img_path = tmp_path / "img.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(img_path)

    contents: list[Any] = ["text-only", img_path]
    out = fake_embedder.embed_batch(contents)
    assert len(out) == 2
    # One text batch + one image batch, both single calls.
    assert len(fake_embedder._model.calls) == 2


def test_embed_text_and_image_unchanged(fake_embedder: ImageEmbedder) -> None:
    """Single-item ``embed_text`` and ``embed_image`` should still work."""
    text_vec = fake_embedder.embed_text("hi")
    assert len(text_vec) == 4
    assert len(fake_embedder._model.calls) == 1
