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


def ffmpeg_error_detail(stderr: str, *, limit: int = 300) -> str:
    """Human-readable tail of an ffmpeg stderr dump.

    ffmpeg prints its version banner and stream inventory first and the
    actual error last, so a head-truncation (``stderr[:200]``) shows only
    the banner. Keep the last non-empty line instead; fall back to a
    generic message when stderr carried nothing at all.
    """
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    if not lines:
        return "unknown ffmpeg error"
    return lines[-1][:limit]

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
        raise MediaProbeError(f"ffprobe failed on {path.name}: {ffmpeg_error_detail(proc.stderr)}")
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
        raise MediaProbeError(
            f"ffmpeg could not decode {source.name}: {ffmpeg_error_detail(proc.stderr)}"
        )


def subtitle_streams(probe: dict) -> list[dict]:
    """Subtitle streams from a probe, annotated with a language tag."""
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]
    for s in streams:
        tags = s.get("tags") or {}
        s["language"] = tags.get("language", "")
    return streams


def extract_subtitle_stream(
    source: Path,
    out_path: Path,
    *,
    index: int = 0,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> None:
    """Extract subtitle stream ``index`` as plain subrip text.

    ffmpeg converts whatever the container carries (mov_text / subrip /
    ass / webvtt) to srt text; failure raises :class:`MediaProbeError`
    and the caller falls back to ASR.
    """
    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-map",
        f"0:s:{index}",
        "-c:s",
        "srt",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(f"ffmpeg subtitle extraction timed out on {source.name}") from exc
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise MediaProbeError(
            f"ffmpeg could not extract subtitles from {source.name}: "
            f"{ffmpeg_error_detail(proc.stderr)}"
        )


def extract_frames(
    source: Path,
    out_dir: Path,
    *,
    interval_s: int = 10,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> list[Path]:
    """Sample one JPEG frame per ``interval_s`` into ``out_dir``.

    Returns frame paths in timeline order (``frame_0001.jpg`` …); frame N
    covers ``[(N-1)*interval, N*interval)``. Sampling (vs scene
    detection) keeps this CPU-cheap; scene-cut precision is a possible
    later refinement, not a blocker for retrieval.
    """
    ffmpeg = _require("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"fps=1/{max(int(interval_s), 1)}",
        "-q:v",
        "4",
        pattern,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(f"ffmpeg frame extraction timed out on {source.name}") from exc
    if proc.returncode != 0:
        raise MediaProbeError(
            f"ffmpeg could not extract frames from {source.name}: {ffmpeg_error_detail(proc.stderr)}"
        )
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise MediaProbeError(f"ffmpeg produced no frames for {source.name}")
    return frames
