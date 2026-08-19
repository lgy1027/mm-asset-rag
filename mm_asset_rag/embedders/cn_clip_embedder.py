"""Chinese-CLIP 图像 / 文本编码器(OFA-Sys/Chinese-CLIP 系列)。

作为 :class:`ImageEmbedder` 的姊妹 backend,在 ``image_provider == "cn_clip"``
时被 ``build_default_image_embedder`` 选中。底层走 ``transformers`` 的
``ChineseCLIPModel`` + ``ChineseCLIPProcessor``。``transformers`` 由 [docling]
或 [clip] extra 传递依赖带入(也可单独 ``pip install transformers``)。
无 cn_clip 专属 extra,依赖缺失时会抛 ``CnClipImageUnavailable``。

为什么不用 sentence-transformers 直加载?
``OFA-Sys/chinese-clip-*`` 不是 sentence-transformers 训练产物,直接
``SentenceTransformer(model_name)`` 会抛 ``ValueError: Invalid model
name``。中文图文检索必须经 ``ChineseCLIPProcessor``(RoBERTa-wwm 分词 +
中文视觉预处理);模型仍需 ``from_pretrained`` 加载,只是走 ``transformers``
而非 ``sentence-transformers``,这是官方支持的路径。

接口形态镜像 ``ImageEmbedder``:满足 ``Embedder`` 与 ``ImageEmbedderProtocol``
两个 Protocol,所以 ``qdrant_backend`` 的 ``isinstance`` 自动识别。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CnClipImageUnavailable(RuntimeError):
    """在没有 ``transformers`` 包或 ``ChineseCLIPModel`` 不可用时抛出。"""


class CnClipImageEmbedder:
    """Chinese-CLIP 图像 + 文本编码器(``transformers`` 后端)。

    模型名默认走 ``Settings.clip_model``,常规用法是 ``OFA-Sys/chinese-clip-vit-base-patch16``
    (768d, ~1 GB,中文 zero-shot Flickr30K-CN R@1 ~71%)。向量维度由 ``dim()`` 在
    首次探测时确定,Qdrant collection 会据此自动加 ``_768d`` 后缀 — 切换模型
    必须 ``mmrag reindex`` 重建 image collection。
    """

    modality = "image"

    def __init__(self, model_name: str | None = None) -> None:
        from ..settings import get_settings

        s = get_settings()
        self.model_name = model_name or s.clip_model
        self._model = None
        self._processor = None
        self._dim: int | None = None
        # 构造时就 fail-fast,调用方拿到清晰错误而非后续 KeyError。
        self._check_available()

    @property
    def name(self) -> str:
        return self.model_name

    @staticmethod
    def is_available() -> bool:
        try:
            from transformers import (  # noqa: F401
                ChineseCLIPModel,
                ChineseCLIPProcessor,
            )
        except ImportError:
            return False
        return True

    @staticmethod
    def _check_available() -> None:
        if not CnClipImageEmbedder.is_available():
            raise CnClipImageUnavailable(
                "Chinese-CLIP image embedding 需要 `transformers` 包 "
                "(ChineseCLIPModel / ChineseCLIPProcessor)。安装: "
                "`pip install transformers` —— 通常随 `[docling]` extra "
                "已间接装入。"
            )

    def dim(self) -> int:
        """返回向量维度。

        优先取 :attr:`Settings.image_embedding_dim`(已配置则跳过 probe,
        省一次模型调用),否则首次调用通过 ``embed_text("probe")`` 探测。
        """
        from ..settings import get_settings

        configured = get_settings().image_embedding_dim
        if configured is not None:
            return configured
        if self._dim is not None:
            return self._dim
        self._dim = len(self.embed_text("probe"))
        return self._dim

    def embed(self, content: Any) -> list[float]:
        if isinstance(content, str):
            return self.embed_text(content)
        if isinstance(content, Path):
            return self.embed_image(content)
        # 兜底:把内容转 Path 走图像路径(非 str/Path 类型,例如 int/dict 会让
        # ``Path()`` 抛 TypeError — 与 ``ImageEmbedder.embed`` 行为一致)。
        return self.embed_image(Path(content))

    def embed_text(self, text: str) -> list[float]:
        model, processor = self._load()
        import torch

        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return [float(v) for v in features[0].tolist()]

    def embed_image(self, image_path: Path) -> list[float] | None:
        """Encode one image. ``None`` if the file cannot be opened / encoded.

        Honors :class:`ImageEmbedderProtocol` graceful-degrade contract.
        """
        from PIL import Image, UnidentifiedImageError

        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None
        try:
            model, processor = self._load()
        except CnClipImageUnavailable:
            return None
        try:
            import torch

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return [float(v) for v in features[0].tolist()]
        except Exception:
            return None

    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本:一次 ``__call__`` 摊销模型调用开销。"""
        if not texts:
            return []
        model, processor = self._load()
        import torch

        inputs = processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return [[float(v) for v in row.tolist()] for row in features]

    def embed_image_batch(self, image_paths: list[Path]) -> list[list[float] | None]:
        """批量编码图像:跳过无法识别的文件,在其槽位放 ``None``。

        与 :class:`ImageEmbedderProtocol` 一致,``len(out) == len(image_paths)``
        契约保留。
        """
        if not image_paths:
            return []
        from PIL import Image, UnidentifiedImageError

        out: list[list[float] | None] = [None] * len(image_paths)
        valid_images: list = []
        valid_indices: list[int] = []
        for i, p in enumerate(image_paths):
            try:
                valid_images.append(Image.open(p).convert("RGB"))
                valid_indices.append(i)
            except (UnidentifiedImageError, OSError):
                continue
        if not valid_images:
            return out
        try:
            model, processor = self._load()
            import torch

            inputs = processor(images=valid_images, return_tensors="pt")
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            rows = features.tolist()
        except Exception:
            return out
        for idx, row in zip(valid_indices, rows):
            out[idx] = [float(v) for v in row]
        return out

    def embed_batch(self, contents: list[Any]) -> list[list[float]]:
        """混合 batch:按类型分流到 text/image batch,行为镜像 ``ImageEmbedder``。"""
        if not contents:
            return []
        texts: list[str] = []
        text_indices: list[int] = []
        images: list[Path] = []
        image_indices: list[int] = []
        for i, c in enumerate(contents):
            if isinstance(c, str):
                texts.append(c)
                text_indices.append(i)
            elif isinstance(c, Path):
                images.append(c)
                image_indices.append(i)
            else:
                # 未知类型回落到单条 encode,保证 ``len(out) == len(contents)`` 契约。
                return [self.embed(c) for c in contents]
        text_vecs = self.embed_text_batch(texts) if texts else []
        image_vecs = self.embed_image_batch(images) if images else []
        out: list[list[float] | None] = [None] * len(contents)
        for idx, vec in zip(text_indices, text_vecs):
            out[idx] = vec
        for idx, vec in zip(image_indices, image_vecs):
            out[idx] = vec
        return [v if v is not None else [] for v in out]

    def _load(self) -> tuple[Any, Any]:
        """Lazy 加载 model + processor。

        ``_check_available`` 仅兜底 ``transformers`` 包级 ImportError;
        ``from_pretrained`` 失败(网络、磁盘、模型名错误等)由本方法捕获并翻译为
        ``CnClipImageUnavailable``,调用方拿到统一错误而不是 raw stack trace。
        """
        if self._model is None or self._processor is None:
            self._check_available()
            try:
                from transformers import ChineseCLIPModel, ChineseCLIPProcessor

                self._processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
                self._model = ChineseCLIPModel.from_pretrained(self.model_name)
                self._model.eval()
            except Exception as exc:
                raise CnClipImageUnavailable(
                    f"加载 Chinese-CLIP 模型 {self.model_name!r} 失败: {exc}"
                ) from exc
        return self._model, self._processor


# 向后兼容别名(代码里同时存在 ``ImageEmbeddingProvider`` 等旧别名,这里不引新别名)。
