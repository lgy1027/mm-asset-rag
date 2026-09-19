"""Encoder fingerprint for the image vector index.

The image index (a dim-suffixed collection such as ``multimodal_image_512d``)
stores vectors produced by whichever image embedder was configured at ingest
time; the search routes embed queries with the *currently* configured
embedder. Same-dimension CLIP-family vectors land in the same dim-suffixed
collection name (e.g. ``multimodal_image_512d``), so a same-dimension
provider switch (``clip`` vs ``cn_clip``) is silent: every query scores near
zero and nothing errors.

The fingerprint records *which* encoder produced the index — concrete
provider class name, best-effort model name, vector dimension, and the
provider's embedding of a deterministic synthetic canary image — as a
JSON sidecar next to the index. The canary catches
same-name-different-weights (a model file updated in place). Ingest
refuses to mix vector spaces; search routes degrade loudly on mismatch.

The helper is image-specific today but deliberately provider-agnostic so
a text-index fingerprint can adopt the same shape later.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ..core.paths import get_indexes_dir
from ..core.protocols import ImageEmbedderProtocol

_CANARY_SIZE = 64
_FINGERPRINT_FILENAME = "image_embedder_fingerprint.json"
# Match iff cosine(canary, re-embedded canary) >= 1 - _COSINE_TOLERANCE.
_COSINE_TOLERANCE = 1e-3


class ImageEncoderMismatchError(ValueError):
    """The image index was built with a different encoder than the current one."""


@dataclass(frozen=True)
class ImageEmbedderFingerprint:
    """Provenance record for the vectors stored in the image index.

    ``provider`` is the concrete embedder class name (``CnClipImageEmbedder``
    / ``ImageEmbedder``); ``model`` is a best-effort model-name probe
    (``None`` when the provider does not expose one); ``dim`` is the vector
    dimension; ``canary`` is the provider's embedding of the deterministic
    synthetic canary image.
    """

    provider: str
    model: str | None
    dim: int
    canary: tuple[float, ...]


def _build_canary_image():
    """Deterministic 64x64 RGB pattern, generated in code (no external files)."""
    from PIL import Image

    size = _CANARY_SIZE
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                (x * 7 + y * 13) % 256,
                (x * 13 + y * 7) % 256,
                ((x ^ y) * 31) % 256,
            )
    return image


def _embed_canary(provider: ImageEmbedderProtocol) -> tuple[float, ...]:
    """Embed the synthetic canary image through ``provider``."""
    image = _build_canary_image()
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        temp_path = Path(handle.name)
        image.save(temp_path, format="PNG")
        vector = provider.embed_image(temp_path)
    if not vector:
        raise ValueError(f"{type(provider).__name__} failed to embed the fingerprint canary image")
    return tuple(float(v) for v in vector)


def _probe_model_name(provider: object) -> str | None:
    """Best-effort model name: ``_model_name`` / ``model_name`` attr probe."""
    for attr in ("_model_name", "model_name"):
        value = getattr(provider, attr, None)
        if value:
            return str(value)
    return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_image_fingerprint(provider: ImageEmbedderProtocol) -> ImageEmbedderFingerprint:
    """Fingerprint ``provider`` by embedding the deterministic canary image."""
    canary = _embed_canary(provider)
    return ImageEmbedderFingerprint(
        provider=type(provider).__name__,
        model=_probe_model_name(provider),
        dim=len(canary),
        canary=canary,
    )


def fingerprint_matches(
    recorded: ImageEmbedderFingerprint, provider: ImageEmbedderProtocol
) -> bool:
    """True iff ``provider`` reproduces the recorded fingerprint.

    Match requires the same provider class name, vector dimension, and
    model name (both ``None`` tolerated), plus cosine similarity of the
    re-embedded canary within ``1 - 1e-3`` of the recorded one.
    """
    current = compute_image_fingerprint(provider)
    if recorded.provider != current.provider:
        return False
    if recorded.dim != current.dim:
        return False
    if recorded.model != current.model:
        return False
    return _cosine(recorded.canary, current.canary) >= 1.0 - _COSINE_TOLERANCE


def save_image_fingerprint(path: Path, fingerprint: ImageEmbedderFingerprint) -> None:
    """Write ``fingerprint`` to ``path`` as a JSON sidecar, atomically.

    The payload is written to a temp file in the same directory and then
    ``os.replace``-ed onto the target, so a concurrent reader never sees a
    partially written sidecar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": fingerprint.provider,
        "model": fingerprint.model,
        "dim": fingerprint.dim,
        "canary": list(fingerprint.canary),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(temp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(temp_name)
        raise


def load_image_fingerprint(path: Path) -> ImageEmbedderFingerprint | None:
    """Read the fingerprint sidecar at ``path``; ``None`` when it does not exist.

    A sidecar that is valid JSON but not a fingerprint-shaped object raises
    ``ValueError`` — the loader's contract is "corrupt sidecar -> ValueError"
    so callers can degrade instead of crashing on ``TypeError``.
    """
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"corrupt image fingerprint sidecar at {path}: expected a JSON object, "
            f"got {type(payload).__name__}"
        )
    if not isinstance(payload.get("canary"), list):
        raise ValueError(
            f"corrupt image fingerprint sidecar at {path}: "
            f'"canary" must be a list, got {type(payload.get("canary")).__name__}'
        )
    return ImageEmbedderFingerprint(
        provider=str(payload["provider"]),
        model=payload.get("model"),
        dim=int(payload["dim"]),
        canary=tuple(float(v) for v in payload["canary"]),
    )


def image_fingerprint_path() -> Path:
    """Default sidecar location: ``<data_home>/indexes/image_embedder_fingerprint.json``."""
    return get_indexes_dir() / _FINGERPRINT_FILENAME
