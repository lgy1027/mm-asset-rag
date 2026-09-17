"""ASR backend abstraction: local FunASR vs remote transcription HTTP API.

Mirrors the ``OCR_BACKEND=local|http`` split: one provider-neutral
contract (:class:`AsrBackend`), two implementations, and a settings
switch (:attr:`Settings.asr_backend`) that picks which one transcribes.
Application code (:mod:`mm_asset_rag.parsers.audio_parser`) only ever
calls :func:`get_asr_backend`; it has no knowledge of torch, ModelScope,
or HTTP.

Both implementations return the same shape — sentence-level
``[{"start", "end", "text"}]`` with float seconds — and both degrade
with a friendly :class:`RuntimeError` when their dependency (the
``[asr]`` extra / a configured ``ASR_HTTP_URL``) is missing, which the
ingest workflow records as a per-asset failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..core.settings import get_settings

log = logging.getLogger(__name__)


@runtime_checkable
class AsrBackend(Protocol):
    """Transcribe a 16 kHz mono WAV into sentence-level timestamps."""

    name: str

    def transcribe(self, wav_path: Path) -> list[dict]: ...


class FunAsrLocalBackend:
    """In-process FunASR (paraformer-zh + fsmn-vad + ct-punc).

    Heavy deps (torch + modelscope) import lazily at first use, so this
    module stays importable on a bare install; the extra surfaces as a
    friendly error at transcribe time.
    """

    name = "funasr-local"

    def __init__(self) -> None:
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:  # pragma: no cover - exercised via friendly error
                raise RuntimeError(
                    "Local ASR requires the [asr] extra: "
                    'pip install -e ".[asr]"  (or set ASR_BACKEND=http).'
                ) from exc
            self._model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
            )
        return self._model

    def transcribe(self, wav_path: Path) -> list[dict]:
        model = self._load()
        result = model.generate(input=str(wav_path), sentence_timestamp=True)  # type: ignore[attr-defined]
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
        # No sentence_info: degrade to one whole-file chunk with the
        # word-level timestamp list's span when available.
        text = str(payload.get("text", "")).strip()
        if not text:
            return []
        timestamps = payload.get("timestamp") or []
        end = float(timestamps[-1][-1]) / 1000.0 if timestamps else 0.0
        return [{"start": 0.0, "end": round(end, 3), "text": text}]


class HttpAsrBackend:
    """Remote OpenAI-compatible ``/v1/audio/transcriptions``.

    ``response_format=verbose_json`` makes the server return ``segments``
    with per-segment ``start`` / ``end`` (seconds), which all mainstream
    compatible servers (OpenAI whisper, faster-whisper-server, Groq, …)
    honour. Missing URL surfaces as a friendly error.
    """

    name = "http"

    def __init__(
        self,
        *,
        url: str,
        token: str = "",
        model: str = "whisper-1",
        timeout: float = 300.0,
    ) -> None:
        self._url = url
        self._token = token
        self._model = model
        self._timeout = timeout

    def transcribe(self, wav_path: Path) -> list[dict]:
        if not self._url:
            raise RuntimeError(
                "ASR_BACKEND=http requires ASR_HTTP_URL "
                "(e.g. http://host:9000/v1/audio/transcriptions)."
            )
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is a core dep
            raise RuntimeError("HTTP ASR requires the requests package") from exc
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            with wav_path.open("rb") as fh:
                response = requests.post(
                    self._url,
                    headers=headers,
                    files={"file": (wav_path.name, fh, "audio/wav")},
                    data={"model": self._model, "response_format": "verbose_json"},
                    timeout=self._timeout,
                )
        except requests.RequestException as exc:
            raise RuntimeError(f"ASR endpoint unreachable ({self._url}): {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"ASR endpoint returned HTTP {response.status_code}: {response.text[:200]}"
            )
        payload = response.json()
        segments = payload.get("segments")
        if isinstance(segments, list) and segments:
            return [
                {
                    "start": round(float(s.get("start", 0.0)), 3),
                    "end": round(float(s.get("end", 0.0)), 3),
                    "text": str(s.get("text", "")).strip(),
                }
                for s in segments
                if isinstance(s, dict)
            ]
        # Servers without verbose segments support: fall back to the
        # plain text field as a single chunk.
        text = str(payload.get("text", "")).strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []


_BACKEND: AsrBackend | None = None
_BACKEND_KIND: str = ""


def get_asr_backend() -> AsrBackend:
    """Process-wide backend singleton, selected by ``Settings.asr_backend``."""
    global _BACKEND, _BACKEND_KIND
    settings = get_settings()
    if _BACKEND is not None and settings.asr_backend == _BACKEND_KIND:
        return _BACKEND
    if settings.asr_backend == "http":
        _BACKEND = HttpAsrBackend(
            url=settings.asr_http_url,
            token=settings.asr_http_token,
            model=settings.asr_http_model,
            timeout=settings.asr_http_timeout,
        )
    else:
        _BACKEND = FunAsrLocalBackend()
    _BACKEND_KIND = settings.asr_backend
    return _BACKEND
