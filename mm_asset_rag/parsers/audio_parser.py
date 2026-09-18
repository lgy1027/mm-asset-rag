"""Audio parsing: ffmpeg normalisation + local FunASR transcription.

Text-route design (see ``docs/design-audio-video.md``): transcript
sentences become timestamped ``ParsedChunk``s that enter the *existing*
text index — embedding, BM25, rewrite, and rerank all work unchanged.

Degradation contract (project-wide): a missing ``[asr]`` extra or ffmpeg
surfaces as a friendly :class:`RuntimeError` which the ingest workflow
records as that asset's failure reason; the rest of the batch is
untouched. Transcripts are cached under ``$MM_ASSET_RAG_HOME/asr/``
keyed by asset id + source mtime/size, so reindex / force re-parse reuse
them without re-running ASR.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from ..core.paths import get_asr_dir
from ..core.schema import ParsedChunk
from ..core.settings import get_settings
from ..ingest.assets import IngestAsset
from .asr_backend import get_asr_backend
from .media_probe import MediaProbeError, normalize_to_wav

log = logging.getLogger(__name__)

_PARSER_NAME = "audio-funasr"


def merge_sentences_into_windows(
    sentences: list[dict],
    *,
    chunk_seconds: int,
) -> list[dict]:
    """Greedily merge consecutive ASR sentences into ``chunk_seconds`` windows.

    A sentence longer than the window becomes its own chunk (never split
    mid-sentence — ASR boundaries are the only trustworthy cut points).
    Each window: ``{"start", "end", "text"}`` with times in seconds.
    """
    windows: list[dict] = []
    current: list[dict] = []
    window_start: float | None = None
    for sentence in sentences:
        if window_start is None:
            window_start = float(sentence["start"])
            current = [sentence]
            continue
        if float(sentence["end"]) - window_start >= chunk_seconds:
            windows.append(_window(current, window_start))
            window_start = float(sentence["start"])
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        assert window_start is not None
        windows.append(_window(current, window_start))
    return windows


def has_searchable_text(text: str) -> bool:
    """Whether a transcript window carries any retrievable token.

    ASR on noisy or silent stretches can yield fragments of pure
    punctuation ("，，。"); they match nothing, pollute rankings, and add
    zero recall, so parsers drop them. CJK ideographs count as alnum in
    Unicode, so Chinese transcripts pass with a single character.
    """
    return any(ch.isalnum() for ch in text)


def _window(sentences: list[dict], start: float) -> dict:
    # Skip punctuation-only sentence fragments (ASR noise on silent
    # stretches) — they add no recall value while polluting the window.
    # The window keeps the full time span regardless.
    texts = [str(s.get("text", "")) for s in sentences if has_searchable_text(str(s.get("text", "")))]
    return {
        "start": start,
        "end": float(sentences[-1]["end"]),
        "text": "".join(texts).strip(),
    }


def _transcribe(wav_path: Path) -> list[dict]:
    """Transcribe a normalised WAV via the configured ASR backend."""
    return get_asr_backend().transcribe(wav_path)


def _cache_path(asset_id: str) -> Path:
    return get_asr_dir() / f"{asset_id}.json"


def transcript_for(asset: IngestAsset) -> list[dict]:
    """Cached sentence-level transcript for the asset's media file."""
    source = asset.file_path
    try:
        stat = source.stat()
        signature = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    except OSError:
        signature = {}
    cache = _cache_path(asset.asset_id)
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("signature") == signature and isinstance(payload.get("sentences"), list):
                return payload["sentences"]
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache → re-transcribe below
    sentences = _transcribe_normalized(source)
    try:
        cache.write_text(
            json.dumps({"signature": signature, "sentences": sentences}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("audio: cannot write transcript cache %s: %s", cache, exc)
    return sentences


def _normalize_to_wav(source: Path, out_path: Path) -> None:
    normalize_to_wav(source, out_path)


def _transcribe_normalized(source: Path) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="mmrag-asr-") as tmp:
        wav = Path(tmp) / "normalized.wav"
        _normalize_to_wav(source, wav)
        return _transcribe(wav)


def parse_audio(
    asset: IngestAsset,
    *,
    chunk_seconds: int | None = None,
    engine: str = "funasr",
) -> list[ParsedChunk]:
    """Transcribe one audio asset into timestamped ``ParsedChunk``s.

    ``engine`` exists for registry symmetry; ``funasr`` is the only
    built-in. ``chunk_seconds`` defaults to ``Settings.audio_chunk_seconds``.
    """
    _ = engine  # single built-in engine; keeps the signature stable
    window_s = chunk_seconds if chunk_seconds is not None else get_settings().audio_chunk_seconds
    sentences = transcript_for(asset)
    chunks: list[ParsedChunk] = []
    for window in merge_sentences_into_windows(sentences, chunk_seconds=window_s):
        if not has_searchable_text(window["text"]):
            continue
        chunks.append(
            ParsedChunk(
                text=window["text"],
                metadata={
                    "asset_id": asset.asset_id,
                    "asset_title": asset.title,
                    "kind": "asr",
                    "source_type": "audio",
                    "source_path": asset.relative_path,
                    "source_url": asset.source_url,
                    "page": None,
                    "parser": _PARSER_NAME,
                    "start": window["start"],
                    "end": window["end"],
                    "duration_s": round(window["end"] - window["start"], 3),
                    "tags": asset.tags,
                },
            )
        )
    return chunks


__all__ = [
    "MediaProbeError",
    "merge_sentences_into_windows",
    "parse_audio",
    "transcript_for",
]
