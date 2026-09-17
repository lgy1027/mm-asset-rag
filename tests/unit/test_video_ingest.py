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
    streams = [{"codec_type": "video", "codec_name": "h264"}]
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
