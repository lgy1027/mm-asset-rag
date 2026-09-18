"""Video parsing: subtitles → ASR fallback → optional VLM frame captions.

Text-route design (see ``docs/design-audio-video.md``): every tier emits
timestamped ``ParsedChunk``s into the existing text index.

Three tiers, in priority order:

1. **Embedded subtitles** (cheapest, verbatim): ffprobe finds subtitle
   streams, ffmpeg converts the first parseable one to text. Skipped when
   the stream is image-based (``hdmv_pgs_subtitle`` / ``dvd_subtitle``
   yield no text and fall through to ASR).
2. **Audio-track ASR** (fallback): reuses the audio pipeline's
   :func:`~mm_asset_rag.parsers.audio_parser.transcript_for` — ffmpeg
   ``-vn`` normalisation already drops the video track, so a video file
   is transcribed exactly like an audio upload.
3. **Keyframe VLM captions** (opt-in via ``enable_vlm``, needs
   ``VLM_*`` creds): one frame per ``video_frame_interval_s`` gets a
   concise caption, covering what speech/subtitles never mention
   (slides, diagrams, on-screen text, silent footage).

Degradation contract (project-wide): missing ffmpeg, unprobed media, or
a missing VLM leaves that tier empty/failed — the asset either still
yields the other tiers' chunks or fails with a readable reason; the rest
of the ingest batch is untouched. A video with no audio stream
skips the ASR tier entirely and is covered by the VLM tier when enabled.
"""

from __future__ import annotations

import base64
import logging
import math
import tempfile
from pathlib import Path

from ..core.llm_transport import completion_message_content, post_chat_completion
from ..core.paths import get_parsed_dir
from ..core.schema import ParsedChunk
from ..core.settings import get_settings
from ..ingest.assets import IngestAsset
from .audio_parser import has_searchable_text, merge_sentences_into_windows, transcript_for
from .media_probe import (
    MediaProbeError,
    detect_scenes,
    extract_frame_at,
    extract_frames,
    extract_subtitle_stream,
    frames_too_similar,
    has_stream,
    probe_media,
    subtitle_streams,
)
from .subtitle_parser import parse_subtitle_text

log = logging.getLogger(__name__)

# Subtitle codecs that carry bitmap images, not text — extraction would
# produce garbage, so we skip straight to the ASR tier for them.
_BITMAP_SUBTITLE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "xsub", "dvb_subtitle"}

_FRAME_PROMPT = (
    "用中文简洁描述这张视频截图,50字以内。突出可检索的内容:画面中的对象、"
    "可见文字、图表/界面类型和正在发生的动作。只输出描述本身,不要分点,不要解释。"
)


def _subtitle_cues(asset: IngestAsset, probe: dict) -> list[dict]:
    """Parse the first usable embedded subtitle stream into cues."""
    for index, stream in enumerate(subtitle_streams(probe)):
        if stream.get("codec_name") in _BITMAP_SUBTITLE_CODECS:
            log.info(
                "video: skipping bitmap subtitle stream %d (%s)", index, stream.get("codec_name")
            )
            continue
        with tempfile.TemporaryDirectory(prefix="mmrag-sub-") as tmp:
            out = Path(tmp) / "stream.srt"
            try:
                extract_subtitle_stream(asset.file_path, out, index=index)
            except MediaProbeError as exc:
                log.warning("video: subtitle stream %d unusable: %s", index, exc)
                continue
            cues = parse_subtitle_text(out.read_text(encoding="utf-8", errors="replace"))
            if cues:
                return cues
    return []


def _chunks_from_cues(
    asset: IngestAsset,
    cues: list[dict],
    *,
    chunk_seconds: int,
    parser_name: str,
    kind: str,
) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []
    windows = merge_sentences_into_windows(cues, chunk_seconds=chunk_seconds)
    for window in windows:
        if not has_searchable_text(window["text"]):
            continue
        chunks.append(
            ParsedChunk(
                text=window["text"],
                metadata={
                    "asset_id": asset.asset_id,
                    "asset_title": asset.title,
                    "source_type": "video",
                    "source_path": asset.relative_path,
                    "source_url": asset.source_url,
                    "page": None,
                    "parser": parser_name,
                    "kind": kind,
                    "start": window["start"],
                    "end": window["end"],
                    "duration_s": round(window["end"] - window["start"], 3),
                    "tags": asset.tags,
                },
            )
        )
    return chunks


def _asr_chunks(asset: IngestAsset, *, chunk_seconds: int) -> list[ParsedChunk]:
    sentences = transcript_for(asset)
    return _chunks_from_cues(
        asset,
        sentences,
        chunk_seconds=chunk_seconds,
        parser_name="video-asr",
        kind="asr",
    )


def _caption_frame(frame_path: Path, *, timeout: float, temperature: float) -> str:
    """One concise VLM caption for a frame; ``""`` on any failure."""
    s = get_settings()
    base_url, api_key, model = s.vlm_creds
    if not base_url or not api_key or not model:
        return ""
    try:
        image_base64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        response = post_chat_completion(
            base_url,
            api_key,
            model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _FRAME_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            timeout=timeout,
            temperature=temperature,
        )
        return completion_message_content(response)[:500]
    except Exception as exc:
        log.warning("video: frame caption failed for %s: %s", frame_path.name, exc)
        return ""


def _effective_interval(interval_s: int, duration: float | None) -> int:
    """Clamp the frame-sampling interval so short videos still yield frames.

    ffmpeg's ``fps=1/N`` filter emits nothing when the footage is shorter
    than ``N`` seconds (the output grid lands beyond every input pts), so
    a 2 s clip with the default 10 s interval would produce zero frames
    and silently lose the whole VLM tier. Unknown duration keeps the
    configured interval.
    """
    interval = max(int(interval_s), 1)
    if duration:
        return max(1, min(interval, math.ceil(duration)))
    return interval


def _cap_scenes(scenes: list[tuple[float, float]], cap: int) -> list[tuple[float, float]]:
    """Evenly downsample ``scenes`` to at most ``cap`` windows, endpoints kept."""
    if cap <= 0 or len(scenes) <= cap:
        return scenes
    if cap == 1:
        return [scenes[0]]
    step = (len(scenes) - 1) / (cap - 1)
    return [scenes[round(i * step)] for i in range(cap)]


def _scene_samples(
    asset: IngestAsset,
    frames_dir: Path,
    *,
    interval: int,
    duration_f: float | None,
) -> list[tuple[Path, float, float]] | None:
    """One ``(frame, window_start, window_end)`` per detected scene.

    ``None`` when PySceneDetect is unavailable or the footage has no
    real cuts (< 2 scenes) — callers fall back to fixed-interval
    sampling, which covers single-scene clips more evenly. The scene
    count is capped so VLM cost stays in the same ballpark as interval
    sampling (≈ one frame per ``interval`` seconds).
    """
    settings = get_settings()
    scenes = detect_scenes(asset.file_path, threshold=settings.video_scene_threshold)
    if scenes is None or len(scenes) < 2:
        return None
    if duration_f:
        scenes = _cap_scenes(scenes, max(1, math.ceil(duration_f / interval)))
    samples: list[tuple[Path, float, float]] = []
    for index, (start, end) in enumerate(scenes):
        try:
            frame = extract_frame_at(
                asset.file_path, frames_dir, start + (end - start) / 2, index=index
            )
        except MediaProbeError as exc:
            log.warning("video: scene %d frame skipped: %s", index, exc)
            continue
        samples.append((frame, round(start, 3), round(end, 3)))
    return samples


def _frame_chunks(asset: IngestAsset, probe: dict) -> list[ParsedChunk]:
    """VLM captions for sampled keyframes, stored under ``parsed/<id>/frames/``.

    Scene-cut sampling when PySceneDetect detects real cuts (frames land
    on shot boundaries, timestamps are scene windows); fixed-interval
    sampling otherwise. Near-identical consecutive frames are deduped
    before captioning so VLM calls and index slots aren't spent twice
    on the same shot.
    """
    settings = get_settings()
    frames_dir = get_parsed_dir() / asset.asset_id / "frames"
    duration = probe.get("format", {}).get("duration")
    try:
        duration_f = float(duration) if duration else None
    except (TypeError, ValueError):
        duration_f = None
    interval = _effective_interval(int(settings.video_frame_interval_s), duration_f)

    samples = _scene_samples(asset, frames_dir, interval=interval, duration_f=duration_f)
    if samples is None:
        try:
            frames = extract_frames(asset.file_path, frames_dir, interval_s=interval)
        except MediaProbeError as exc:
            log.warning("video: frame extraction skipped: %s", exc)
            return []
        total = duration_f if duration_f else len(frames) * interval
        samples = [
            (frame, round(i * interval, 3), round(min((i + 1) * interval, total), 3))
            for i, frame in enumerate(frames)
        ]

    chunks: list[ParsedChunk] = []
    previous: Path | None = None
    for frame, start, end in samples:
        if previous is not None and frames_too_similar(previous, frame):
            continue
        previous = frame
        caption = _caption_frame(
            frame,
            timeout=settings.vlm_timeout,
            temperature=settings.vlm_temperature,
        )
        if not caption:
            continue
        chunks.append(
            ParsedChunk(
                text=caption,
                metadata={
                    "asset_id": asset.asset_id,
                    "asset_title": asset.title,
                    "source_type": "video",
                    "source_path": asset.relative_path,
                    "source_url": asset.source_url,
                    "page": None,
                    "parser": "video-frame-vlm",
                    "kind": "frame",
                    "start": start,
                    "end": end,
                    "duration_s": round(end - start, 3),
                    "frame_path": f"frames/{frame.name}",
                    "tags": asset.tags,
                },
            )
        )
    return chunks


def parse_video(
    asset: IngestAsset,
    *,
    enable_vlm: bool = False,
    chunk_seconds: int | None = None,
) -> list[ParsedChunk]:
    """Parse one video asset into timestamped ``ParsedChunk``s.

    Tiers 1 (subtitles) and 2 (ASR fallback) always run; tier 3 (frame
    captions) only when ``enable_vlm`` is set and VLM creds exist.
    """
    window_s = chunk_seconds if chunk_seconds is not None else get_settings().audio_chunk_seconds
    probe = probe_media(asset.file_path)
    if not has_stream(probe, "video"):
        raise MediaProbeError(f"{asset.relative_path}: no video stream found")

    chunks: list[ParsedChunk] = []
    vlm_ran = False
    cues = _subtitle_cues(asset, probe)
    if cues:
        chunks.extend(
            _chunks_from_cues(
                asset,
                cues,
                chunk_seconds=window_s,
                parser_name="video-subtitle",
                kind="subtitle",
            )
        )
    else:
        # No usable embedded subtitles → transcribe the audio track, when
        # there is one. ``has_stream`` gates the ASR call so silent footage
        # never shells out to ffmpeg only to fail; a decode failure is
        # recorded instead of raised, giving the VLM tier below a chance
        # to cover the asset from frames.
        asr_error: MediaProbeError | None = None
        if has_stream(probe, "audio"):
            try:
                chunks.extend(_asr_chunks(asset, chunk_seconds=window_s))
            except MediaProbeError as exc:
                asr_error = exc
        if not chunks and enable_vlm:
            chunks.extend(_frame_chunks(asset, probe))
            vlm_ran = True
        if not chunks:
            raise MediaProbeError(
                _explain_no_chunks(asset, probe, asr_error, enable_vlm=enable_vlm)
            )
    if enable_vlm and not vlm_ran:
        # Speech/subtitles succeeded; frames still add coverage for
        # on-screen content that speech never mentions.
        chunks.extend(_frame_chunks(asset, probe))
    return chunks


def _explain_no_chunks(
    asset: IngestAsset,
    probe: dict,
    asr_error: MediaProbeError | None,
    *,
    enable_vlm: bool,
) -> str:
    """Readable reason every text tier came up empty."""
    parts = ["no usable subtitles"]
    if not has_stream(probe, "audio"):
        parts.append("no audio stream")
    elif asr_error is not None:
        parts.append(f"ASR failed: {asr_error}")
    if enable_vlm:
        parts.append("VLM produced no frame captions")
    else:
        parts.append("VLM disabled (set ENABLE_VLM=true to index frames from silent footage)")
    return f"{asset.relative_path}: " + "; ".join(parts)


__all__ = ["parse_video"]
