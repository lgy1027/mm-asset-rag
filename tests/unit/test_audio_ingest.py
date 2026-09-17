"""Tests for audio sniffing, ffmpeg probing, FunASR parsing, and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.ingest.sniff import sniff
from mm_asset_rag.parsers.audio_parser import merge_sentences_into_windows, parse_audio


def _write(tmp_path: Path, name: str, magic: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(magic + b"\x00" * 64)
    return path


# ── sniff: audio / video magic bytes ────────────────────────────────────────


def test_sniff_mp3_by_id3(tmp_path):
    got = sniff(_write(tmp_path, "talk.mp3", b"ID3\x04\x00\x00\x00\x00\x00\x00"))
    assert got.source_type == "audio"


def test_sniff_mp3_by_frame_sync(tmp_path):
    got = sniff(_write(tmp_path, "talk.mp3", b"\xff\xfb\x90\x44" + b"\x00" * 60))
    assert got.source_type == "audio"


def test_sniff_wav_is_audio_not_image(tmp_path):
    got = sniff(_write(tmp_path, "rec.wav", b"RIFF\x24\x08\x00\x00WAVEfmt "))
    assert got.source_type == "audio"


def test_sniff_webp_stays_image(tmp_path):
    got = sniff(_write(tmp_path, "pic.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 "))
    assert got.source_type == "image"


def test_sniff_avi_is_video(tmp_path):
    got = sniff(_write(tmp_path, "clip.avi", b"RIFF\x00\x00\x00\x00AVI LIST"))
    assert got.source_type == "video"


def test_sniff_flac(tmp_path):
    got = sniff(_write(tmp_path, "song.flac", b"fLaC\x00\x00\x00\x22"))
    assert got.source_type == "audio"


def test_sniff_ogg(tmp_path):
    got = sniff(_write(tmp_path, "ep.ogg", b"OggS\x00\x02\x00\x00\x00\x00"))
    assert got.source_type == "audio"


def test_sniff_m4a_ftyp_major_brand(tmp_path):
    got = sniff(_write(tmp_path, "voice.m4a", b"\x00\x00\x00\x18ftypM4A \x00\x00"))
    assert got.source_type == "audio"


def test_sniff_mp4_ftyp(tmp_path):
    got = sniff(_write(tmp_path, "meeting.mp4", b"\x00\x00\x00\x18ftypisom\x00\x00"))
    assert got.source_type == "video"


def test_sniff_mov_qt_brand(tmp_path):
    got = sniff(_write(tmp_path, "take.mov", b"\x00\x00\x00\x14ftypqt  \x00\x00"))
    assert got.source_type == "video"


def test_sniff_mkv_ebml(tmp_path):
    got = sniff(_write(tmp_path, "film.mkv", b"\x1a\x45\xdf\xa3\x93\x42\x82\x88"))
    assert got.source_type == "video"


def test_sniff_unknown_still_unknown(tmp_path):
    got = sniff(_write(tmp_path, "blob.bin", b"\x7fELF\x02\x01\x01\x00"))
    assert got.source_type == "unknown"


# ── window merging ──────────────────────────────────────────────────────────


def test_merge_windows_groups_sentences():
    sentences = [
        {"start": 0.0, "end": 5.0, "text": "大家好。"},
        {"start": 5.0, "end": 12.0, "text": "今天我们讲检索。"},
        {"start": 12.0, "end": 20.0, "text": "先讲 BM25。"},
        {"start": 20.0, "end": 26.0, "text": "再讲向量。"},
    ]
    windows = merge_sentences_into_windows(sentences, chunk_seconds=15)
    assert [(w["start"], w["end"]) for w in windows] == [(0.0, 12.0), (12.0, 26.0)]
    assert windows[0]["text"] == "大家好。今天我们讲检索。"


def test_merge_windows_long_sentence_kept_whole():
    sentences = [
        {"start": 0.0, "end": 40.0, "text": "很长的 uninterrupted 独白……"},
        {"start": 40.0, "end": 45.0, "text": "结尾。"},
    ]
    windows = merge_sentences_into_windows(sentences, chunk_seconds=15)
    assert [(w["start"], w["end"]) for w in windows] == [(0.0, 40.0), (40.0, 45.0)]


def test_merge_windows_empty():
    assert merge_sentences_into_windows([], chunk_seconds=15) == []


# ── parse_audio degradation + cache ─────────────────────────────────────────


def _fake_wav(tmp_path: Path) -> Path:
    """A structurally valid WAV header (0 data bytes) — good enough to
    reach the ASR stage, where the missing extra must surface."""
    import wave

    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"")
    return path


def test_parse_audio_friendly_error_without_asr_extra(tmp_path, monkeypatch):
    from mm_asset_rag.ingest.assets import IngestAsset

    wav = _fake_wav(tmp_path)
    asset = IngestAsset(
        asset_id="tone",
        title="Tone",
        source_type="audio",
        relative_path=wav.name,
        asset_dir=tmp_path,
    )
    monkeypatch.setitem(__import__("sys").modules, "funasr", None)
    with pytest.raises(RuntimeError, match=r"\[asr\]"):
        parse_audio(asset)


def test_parse_audio_chunks_carry_timestamps(tmp_path, monkeypatch):
    from mm_asset_rag.ingest.assets import IngestAsset
    from mm_asset_rag.parsers import audio_parser

    wav = _fake_wav(tmp_path)
    asset = IngestAsset(
        asset_id="talk",
        title="Talk",
        source_type="audio",
        relative_path=wav.name,
        asset_dir=tmp_path,
    )
    sentences = [
        {"start": 0.0, "end": 5.0, "text": "第一句。"},
        {"start": 5.0, "end": 12.0, "text": "第二句。"},
    ]
    calls: list[str] = []

    def fake_transcribe(_path):
        calls.append("transcribed")
        return sentences

    monkeypatch.setattr(audio_parser, "_transcribe", fake_transcribe)
    monkeypatch.setattr(audio_parser, "_normalize_to_wav", lambda _src, _dst: None)

    chunks = parse_audio(asset, chunk_seconds=30)
    assert calls == ["transcribed"]
    assert len(chunks) == 1
    assert chunks[0].metadata["start"] == 0.0
    assert chunks[0].metadata["end"] == 12.0
    assert chunks[0].metadata["parser"] == "audio-funasr"
    assert "第一句" in chunks[0].text

    # Second parse hits the transcript cache: no re-transcription.
    chunks_again = parse_audio(asset, chunk_seconds=30)
    assert calls == ["transcribed"]
    assert [c.metadata["end"] for c in chunks_again] == [12.0]


def test_media_probe_friendly_error_without_ffmpeg(monkeypatch):
    import shutil

    from mm_asset_rag.parsers import media_probe

    monkeypatch.setattr(shutil, "which", lambda _tool: None)
    with pytest.raises(media_probe.MediaProbeError, match="ffmpeg"):
        media_probe.probe_media(Path("/tmp/whatever.mp4"))


# ── ASR backend abstraction ─────────────────────────────────────────────────


def test_asr_backend_default_is_local():
    from mm_asset_rag.parsers.asr_backend import get_asr_backend

    assert get_asr_backend().name == "funasr-local"


def test_asr_backend_switch_to_http(monkeypatch, tmp_path):
    from mm_asset_rag.core.settings import get_settings
    from mm_asset_rag.parsers import asr_backend

    get_settings.cache_clear()
    monkeypatch.setenv("ASR_BACKEND", "http")
    monkeypatch.setenv("ASR_HTTP_URL", "http://127.0.0.1:9/v1/audio/transcriptions")
    asr_backend._BACKEND = None
    asr_backend._BACKEND_KIND = ""
    try:
        backend = asr_backend.get_asr_backend()
        assert backend.name == "http"
        # 16 kHz mono wav bytes — just needs to be a readable file.
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"RIFFfake")
        with pytest.raises(RuntimeError, match="HTTP"):
            backend.transcribe(wav)
    finally:
        asr_backend._BACKEND = None
        asr_backend._BACKEND_KIND = ""
        get_settings.cache_clear()


def test_http_asr_parses_verbose_json_segments(monkeypatch, tmp_path):
    from mm_asset_rag.parsers.asr_backend import HttpAsrBackend

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "segments": [
                    {"start": 0.0, "end": 3.5, "text": " 你好 "},
                    {"start": 4.0, "end": 7.0, "text": "世界"},
                ]
            }

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResponse())
    backend = HttpAsrBackend(url="http://x/v1/audio/transcriptions")
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFFfake")
    sentences = backend.transcribe(wav)
    assert sentences == [
        {"start": 0.0, "end": 3.5, "text": "你好"},
        {"start": 4.0, "end": 7.0, "text": "世界"},
    ]


def test_http_asr_plain_text_fallback(monkeypatch, tmp_path):
    from mm_asset_rag.parsers.asr_backend import HttpAsrBackend

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"text": "整段文本"}

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResponse())
    backend = HttpAsrBackend(url="http://x/v1/audio/transcriptions")
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFFfake")
    assert backend.transcribe(wav) == [{"start": 0.0, "end": 0.0, "text": "整段文本"}]


def test_http_asr_requires_url():
    from mm_asset_rag.parsers.asr_backend import HttpAsrBackend

    with pytest.raises(RuntimeError, match="ASR_HTTP_URL"):
        HttpAsrBackend(url="").transcribe(Path("/tmp/x.wav"))
