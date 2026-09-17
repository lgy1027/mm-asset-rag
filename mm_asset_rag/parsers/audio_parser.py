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


def _window(sentences: list[dict], start: float) -> dict:
    return {
        "start": start,
        "end": float(sentences[-1]["end"]),
        "text": "".join(str(s.get("text", "")) for s in sentences).strip(),
    }


# Process-wide FunASR handle, lazy like the PP-OCRv6 one: importing this
# module never requires torch; the extra is only needed at first use.
_ASR_MODEL: object | None = None


def _load_asr_model() -> object:
    global _ASR_MODEL
    if _ASR_MODEL is None:
        try:
            from funasr import AutoModel
        except ImportError as exc:  # pragma: no cover - exercised via friendly error
            raise RuntimeError(
                "Local audio transcription requires the [asr] extra: "
                'pip install -e ".[asr]"  (or pip install funasr).'
            ) from exc
        # paraformer-zh: Chinese-primary ASR; fsmn-vad splits speech into
        # sentences (giving us per-sentence timestamps); ct-punc restores
        # punctuation so window text stays readable. Models download from
        # ModelScope on first run.
        _ASR_MODEL = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
        )
    return _ASR_MODEL


def _transcribe(wav_path: Path) -> list[dict]:
    """Run FunASR on a normalised WAV; return sentence-level timestamps.

    Falls back to a single whole-file sentence when the model output
    carries no ``sentence_info`` (e.g. a vad-less build) — chunking then
    degrades to one chunk instead of failing.
    """
    model = _load_asr_model()
    result = model.generate(input=str(wav_path))  # type: ignore[attr-defined]
    payload = result[0] if isinstance(result, list) and result else {}
    sentences = payload.get("sentence_info")
    if isinstance(sentences, list) and sentences:
        return [
            {
                "start": round(float(s["start"]) / 1000.0, 3),
                "end": round(float(s["end"]) / 1000.0, 3),
                "text": str(s.get("text", "")),
            }
            for s in sentences
            if isinstance(s, dict) and "start" in s and "end" in s
        ]
    text = str(payload.get("text", "")).strip()
    return [{"start": 0.0, "end": 0.0, "text": text}] if text else []


def _cache_path(asset_id: str) -> Path:
    return get_asr_dir() / f"{asset_id}.json"


def _transcript_for(asset: IngestAsset) -> list[dict]:
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
    sentences = _transcript_for(asset)
    chunks: list[ParsedChunk] = []
    for window in merge_sentences_into_windows(sentences, chunk_seconds=window_s):
        if not window["text"]:
            continue
        chunks.append(
            ParsedChunk(
                text=window["text"],
                metadata={
                    "asset_id": asset.asset_id,
                    "asset_title": asset.title,
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
]
