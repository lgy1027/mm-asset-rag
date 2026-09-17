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
of the ingest batch is untouched.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

from ..core.llm_transport import post_chat_completion
from ..core.paths import get_parsed_dir
from ..core.schema import ParsedChunk
from ..core.settings import get_settings
from ..ingest.assets import IngestAsset
from .audio_parser import merge_sentences_into_windows, transcript_for
from .media_probe import (
    MediaProbeError,
    extract_frames,
    extract_subtitle_stream,
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
        if not window["text"]:
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
        choices = (response or {}).get("choices") or []
        message = (choices[0] or {}).get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            # Reasoning models may answer only in the thinking field
            # (same convention as contextual._THINK_RE / image_caption).
            reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
            content = reasoning.splitlines()[-1].strip() if reasoning else ""
        return content[:500]
    except Exception as exc:
        log.warning("video: frame caption failed for %s: %s", frame_path.name, exc)
        return ""


def _frame_chunks(asset: IngestAsset, probe: dict) -> list[ParsedChunk]:
    """VLM captions for sampled keyframes, stored under ``parsed/<id>/frames/``."""
    settings = get_settings()
    interval = max(int(settings.video_frame_interval_s), 1)
    frames_dir = get_parsed_dir() / asset.asset_id / "frames"
    duration = probe.get("format", {}).get("duration")
    try:
        frames = extract_frames(asset.file_path, frames_dir, interval_s=interval)
    except MediaProbeError as exc:
        log.warning("video: frame extraction skipped: %s", exc)
        return []
    total = float(duration) if duration else len(frames) * interval
    chunks: list[ParsedChunk] = []
    for index, frame in enumerate(frames):
        caption = _caption_frame(
            frame,
            timeout=settings.vlm_timeout,
            temperature=settings.vlm_temperature,
        )
        if not caption:
            continue
        start = round(index * interval, 3)
        end = round(min((index + 1) * interval, total), 3)
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
        # No usable embedded subtitles → transcribe the audio track. This
        # reuses the audio pipeline wholesale (including its transcript
        # cache); ffmpeg -vn drops the video track during normalisation.
        try:
            chunks.extend(_asr_chunks(asset, chunk_seconds=window_s))
        except MediaProbeError as exc:
            raise MediaProbeError(
                f"{asset.relative_path}: no usable subtitles and ASR failed: {exc}"
            ) from exc

    if enable_vlm:
        chunks.extend(_frame_chunks(asset, probe))
    return chunks


__all__ = ["parse_video"]
