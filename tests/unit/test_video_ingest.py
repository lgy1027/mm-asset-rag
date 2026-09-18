"""Tests for subtitle parsing and the video parser tier cascade."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.parsers import subtitle_parser, video_parser
from mm_asset_rag.parsers.media_probe import MediaProbeError

# ── subtitle text parsing ───────────────────────────────────────────────────

SRT = """1
00:00:01,000 --> 00:00:04,000
大家好

2
00:00:04,500 --> 00:00:09,000
今天讲视频检索
"""

VTT = """WEBVTT

00:01.000 --> 00:03.000
Hello

00:04.000 --> 00:06.500
World
"""

ASS = """[Script Info]
Title: demo

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,{\\an8}第一行
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,第二行\\N换行
"""


def test_parse_srt():
    cues = subtitle_parser.parse_subtitle_text(SRT)
    assert [(c["start"], c["end"], c["text"]) for c in cues] == [
        (1.0, 4.0, "大家好"),
        (4.5, 9.0, "今天讲视频检索"),
    ]


def test_parse_vtt_auto_detected():
    cues = subtitle_parser.parse_subtitle_text(VTT)
    assert [(c["start"], c["end"]) for c in cues] == [(1.0, 3.0), (4.0, 6.5)]


def test_parse_ass_strips_overrides_and_newlines():
    cues = subtitle_parser.parse_subtitle_text(ASS)
    assert [(c["start"], c["end"], c["text"]) for c in cues] == [
        (1.0, 3.5, "第一行"),
        (4.0, 6.0, "第二行 换行"),
    ]


def test_parse_subtitle_garbage_returns_empty():
    assert subtitle_parser.parse_subtitle_text("not a subtitle at all") == []


# ── video parser cascade ────────────────────────────────────────────────────


def _asset(tmp_path: Path) -> object:
    from mm_asset_rag.ingest.assets import IngestAsset

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"\x00" * 32)
    return IngestAsset(
        asset_id="clip",
        title="Clip",
        source_type="video",
        relative_path="clip.mp4",
        asset_dir=tmp_path,
    )


def _probe(*, with_subtitle: bool) -> dict:
    streams = [
        {"codec_type": "video", "codec_name": "h264"},
        {"codec_type": "audio", "codec_name": "aac"},
    ]
    if with_subtitle:
        streams.append(
            {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "chi"}}
        )
    return {"streams": streams, "format": {"duration": "45.0"}}


def test_video_subtitle_tier_skips_asr(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: _probe(with_subtitle=True))
    monkeypatch.setattr(
        video_parser,
        "_subtitle_cues",
        lambda _a, _p: [{"start": 0.0, "end": 5.0, "text": "字幕内容"}],
    )
    monkeypatch.setattr(
        video_parser,
        "transcript_for",
        lambda _a: calls.append("asr") or [],
    )
    chunks = video_parser.parse_video(asset, chunk_seconds=30)
    assert calls == []
    assert [c.metadata["kind"] for c in chunks] == ["subtitle"]
    assert chunks[0].metadata["parser"] == "video-subtitle"
    assert chunks[0].metadata["start"] == 0.0


def test_video_falls_back_to_asr_without_subtitles(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: _probe(with_subtitle=False))
    monkeypatch.setattr(video_parser, "_subtitle_cues", lambda _a, _p: [])
    monkeypatch.setattr(
        video_parser,
        "transcript_for",
        lambda _a: [{"start": 0.0, "end": 7.0, "text": "语音内容"}],
    )
    chunks = video_parser.parse_video(asset, chunk_seconds=30)
    assert [c.metadata["kind"] for c in chunks] == ["asr"]
    assert chunks[0].metadata["parser"] == "video-asr"


def test_video_bitmap_subtitles_fall_through_to_asr(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
            {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        ],
        "format": {"duration": "10.0"},
    }
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: probe)
    calls: list[str] = []
    monkeypatch.setattr(video_parser, "_subtitle_cues", lambda _a, _p: calls.append("sub") or [])
    monkeypatch.setattr(
        video_parser,
        "transcript_for",
        lambda _a: [{"start": 0.0, "end": 3.0, "text": "语音"}],
    )
    # Bitmap codecs are filtered inside _subtitle_cues; here we verify the
    # empty-cue → ASR contract holds regardless of why subtitles failed.
    chunks = video_parser.parse_video(asset, chunk_seconds=30)
    assert calls == ["sub"]
    assert [c.metadata["kind"] for c in chunks] == ["asr"]


def test_video_rejects_media_without_video_stream(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: {"streams": [], "format": {}})
    with pytest.raises(MediaProbeError, match="no video stream"):
        video_parser.parse_video(asset)


def test_video_vlm_frame_tier_appends_frame_chunks(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: _probe(with_subtitle=True))
    monkeypatch.setattr(
        video_parser,
        "_subtitle_cues",
        lambda _a, _p: [{"start": 0.0, "end": 5.0, "text": "字幕"}],
    )
    monkeypatch.setattr(video_parser, "extract_frames", lambda *_a, **_k: [])
    monkeypatch.setattr(
        video_parser,
        "_frame_chunks",
        lambda _a, _p: [
            video_parser.ParsedChunk(
                text="幻灯片显示季度图表",
                metadata={
                    "asset_id": "clip",
                    "source_type": "video",
                    "parser": "video-frame-vlm",
                    "kind": "frame",
                    "start": 0.0,
                    "end": 10.0,
                    "frame_path": "frames/frame_0001.jpg",
                },
            )
        ],
    )
    chunks = video_parser.parse_video(asset, enable_vlm=True, chunk_seconds=30)
    kinds = [c.metadata["kind"] for c in chunks]
    assert kinds == ["subtitle", "frame"]
    assert chunks[1].metadata["frame_path"] == "frames/frame_0001.jpg"


def test_video_vlm_tier_off_by_default(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: _probe(with_subtitle=True))
    monkeypatch.setattr(
        video_parser,
        "_subtitle_cues",
        lambda _a, _p: [{"start": 0.0, "end": 5.0, "text": "字幕"}],
    )
    monkeypatch.setattr(
        video_parser,
        "_frame_chunks",
        lambda _a, _p: pytest.fail("VLM tier must not run without enable_vlm"),
    )
    chunks = video_parser.parse_video(asset)
    assert [c.metadata["kind"] for c in chunks] == ["subtitle"]


# ── silent-video degradation (option B: auto-fallback to VLM) ───────────────


def _frame_chunk() -> video_parser.ParsedChunk:
    return video_parser.ParsedChunk(
        text="画面显示海底鱼群游动",
        metadata={
            "asset_id": "clip",
            "source_type": "video",
            "parser": "video-frame-vlm",
            "kind": "frame",
            "start": 0.0,
            "end": 2.0,
            "frame_path": "frames/frame_0001.jpg",
        },
    )


def _silent_probe() -> dict:
    return {
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
        "format": {"duration": "2.0"},
    }


def test_silent_video_skips_asr_and_falls_back_to_vlm(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: _silent_probe())
    monkeypatch.setattr(video_parser, "_subtitle_cues", lambda _a, _p: [])
    calls: list[str] = []
    monkeypatch.setattr(
        video_parser,
        "transcript_for",
        lambda _a: calls.append("asr") or [],
    )
    monkeypatch.setattr(video_parser, "_frame_chunks", lambda _a, _p: [_frame_chunk()])
    chunks = video_parser.parse_video(asset, enable_vlm=True, chunk_seconds=30)
    assert calls == []  # no audio stream → must not shell out to ffmpeg/ASR at all
    assert [c.metadata["kind"] for c in chunks] == ["frame"]


def test_silent_video_without_vlm_raises_readable_error(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: _silent_probe())
    monkeypatch.setattr(video_parser, "_subtitle_cues", lambda _a, _p: [])
    monkeypatch.setattr(
        video_parser,
        "transcript_for",
        lambda _a: pytest.fail("ASR must not run on a file with no audio stream"),
    )
    with pytest.raises(MediaProbeError, match="no audio stream"):
        video_parser.parse_video(asset, enable_vlm=False)
    # The hint must point at the VLM switch so users know the escape hatch.
    with pytest.raises(MediaProbeError, match="ENABLE_VLM"):
        video_parser.parse_video(asset, enable_vlm=False)


def test_silent_video_with_vlm_but_no_captions_raises_readable_error(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: _silent_probe())
    monkeypatch.setattr(video_parser, "_subtitle_cues", lambda _a, _p: [])
    monkeypatch.setattr(video_parser, "_frame_chunks", lambda _a, _p: [])
    with pytest.raises(MediaProbeError, match="no frame captions"):
        video_parser.parse_video(asset, enable_vlm=True)


def test_asr_failure_falls_back_to_vlm_frames(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "10.0"},
    }
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: probe)
    monkeypatch.setattr(video_parser, "_subtitle_cues", lambda _a, _p: [])
    monkeypatch.setattr(
        video_parser,
        "transcript_for",
        lambda _a: (_ for _ in ()).throw(MediaProbeError("ffmpeg could not decode")),
    )
    monkeypatch.setattr(video_parser, "_frame_chunks", lambda _a, _p: [_frame_chunk()])
    chunks = video_parser.parse_video(asset, enable_vlm=True, chunk_seconds=30)
    assert [c.metadata["kind"] for c in chunks] == ["frame"]


def test_asr_failure_without_vlm_keeps_asr_reason(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "10.0"},
    }
    monkeypatch.setattr(video_parser, "probe_media", lambda _p: probe)
    monkeypatch.setattr(video_parser, "_subtitle_cues", lambda _a, _p: [])
    monkeypatch.setattr(
        video_parser,
        "transcript_for",
        lambda _a: (_ for _ in ()).throw(MediaProbeError("ffmpeg could not decode clip: boom")),
    )
    with pytest.raises(MediaProbeError, match="ASR failed: .*boom"):
        video_parser.parse_video(asset, enable_vlm=False)


# ── ffmpeg error message hygiene ────────────────────────────────────────────


def test_ffmpeg_error_uses_last_line_not_banner():
    from mm_asset_rag.parsers import media_probe

    stderr = (
        "ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers\n"
        "  built with Apple clang version 21.0.0\n"
        "Output #0, wav, to '/tmp/x.wav':\n"
        "Error opening output file /tmp/x.wav.\n"
        "Error opening output files: Invalid argument\n"
    )
    assert media_probe.ffmpeg_error_detail(stderr) == "Error opening output files: Invalid argument"


def test_ffmpeg_error_detail_handles_empty():
    from mm_asset_rag.parsers import media_probe

    assert media_probe.ffmpeg_error_detail("") == "unknown ffmpeg error"
    assert media_probe.ffmpeg_error_detail("   \n  ") == "unknown ffmpeg error"


# ── VLM caption response handling ───────────────────────────────────────────


def _fake_settings():
    from types import SimpleNamespace

    return SimpleNamespace(
        vlm_creds=("http://vlm.local/v1", "key", "vlm-model"),
        vlm_timeout=10.0,
        vlm_temperature=0.1,
    )


def _fake_response(body: dict) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(json=lambda: body)


def test_caption_frame_reads_chat_completion_json_body(tmp_path, monkeypatch):
    """Regression: post_chat_completion returns requests.Response, not a dict."""
    frame = tmp_path / "frame_0001.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xe0")
    body = {"choices": [{"message": {"content": " 画面显示海底鱼群游动 "}}]}
    monkeypatch.setattr(video_parser, "get_settings", _fake_settings)
    monkeypatch.setattr(
        video_parser, "post_chat_completion", lambda *a, **k: _fake_response(body)
    )
    assert video_parser._caption_frame(frame, timeout=10, temperature=0.1) == "画面显示海底鱼群游动"


def test_caption_frame_falls_back_to_reasoning_field(tmp_path, monkeypatch):
    frame = tmp_path / "frame_0001.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xe0")
    body = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning": "思考过程: 1. 观察画面主体。 2. 判断为水下场景。",
                }
            }
        ]
    }
    monkeypatch.setattr(video_parser, "get_settings", _fake_settings)
    monkeypatch.setattr(
        video_parser, "post_chat_completion", lambda *a, **k: _fake_response(body)
    )
    caption = video_parser._caption_frame(frame, timeout=10, temperature=0.1)
    assert "水下场景" in caption


# ── frame sampling on videos shorter than the interval ──────────────────────


def test_effective_interval_clamped_to_short_video():
    # 2 s footage with a 10 s interval must still yield at least one frame:
    # ffmpeg's fps filter emits nothing when 1/interval > 1/duration.
    assert video_parser._effective_interval(10, 2.0) == 2
    assert video_parser._effective_interval(10, 21.75) == 10
    assert video_parser._effective_interval(10, 0.4) == 1
    assert video_parser._effective_interval(10, 0.0) == 10  # unknown duration: keep setting
    assert video_parser._effective_interval(10, None) == 10


def test_video_cue_chunks_drop_punctuation_only_windows(tmp_path, monkeypatch):
    asset = _asset(tmp_path)
    cues = [
        {"start": 0.0, "end": 5.0, "text": "有用内容。"},
        {"start": 5.0, "end": 9.0, "text": "，，。"},
    ]
    chunks = video_parser._chunks_from_cues(
        asset, cues, chunk_seconds=30, parser_name="video-asr", kind="asr"
    )
    assert [c.text for c in chunks] == ["有用内容。"]
