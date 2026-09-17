"""ffmpeg / ffprobe helpers for the audio and video parsers.

The media parsers shell out to ffmpeg (normalisation, frame extraction,
subtitle probing) and ffprobe (stream inventory, duration). Both tools
come from the same ffmpeg install; when they're absent we raise a
friendly :class:`MediaProbeError` that the ingest workflow reports as a
per-asset failure — never a batch abort.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 120


class MediaProbeError(RuntimeError):
    """ffmpeg/ffprobe missing or a probe/extraction command failed."""


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise MediaProbeError(
            f"{tool} not found on PATH. Install ffmpeg "
            "(e.g. `brew install ffmpeg` or `apt install ffmpeg`)."
        )
    return path


def probe_media(path: Path, *, timeout_s: int = _DEFAULT_TIMEOUT_S) -> dict:
    """Return ffprobe's JSON inventory (format + streams) for ``path``."""
    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(f"ffprobe timed out on {path.name}") from exc
    if proc.returncode != 0:
        raise MediaProbeError(f"ffprobe failed on {path.name}: {proc.stderr.strip()[:200]}")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe returned non-JSON for {path.name}") from exc


def has_stream(probe: dict, kind: str) -> bool:
    """Whether the probe reports at least one stream of ``kind``
    (``"audio"`` / ``"video"`` / ``"subtitle"``)."""
    return any(s.get("codec_type") == kind for s in probe.get("streams", []))


def stream_duration_s(probe: dict) -> float | None:
    """Duration from the format section (seconds), or ``None``."""
    raw = (probe.get("format") or {}).get("duration")
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalize_to_wav(
    source: Path,
    out_path: Path,
    *,
    sample_rate: int = 16000,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> None:
    """Decode any ffmpeg-readable media to 16-bit mono WAV for ASR.

    ``-vn`` drops video (video files contribute only their audio track
    here; frames are a separate path). Failures raise
    :class:`MediaProbeError` with ffmpeg's stderr.
    """
    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(f"ffmpeg normalisation timed out on {source.name}") from exc
    if proc.returncode != 0 or not out_path.exists():
        raise MediaProbeError(f"ffmpeg could not decode {source.name}: {proc.stderr.strip()[:200]}")
