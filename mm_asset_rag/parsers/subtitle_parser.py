"""Subtitle text parsing: srt / vtt / ass → timestamped cues.

Pure-text module — no ffmpeg, no I/O beyond the caller passing the
extracted subtitle text. :func:`parse_subtitle_text` auto-detects the
format from content (``WEBVTT`` header, ``-->`` cue markers, or ASS
``Dialogue:`` lines) so the video parser doesn't need to know which
container codec produced the text.

Times are normalised to float seconds. Malformed lines are skipped,
never fatal: a partially-corrupt subtitle stream still yields the cues
that parse.
"""

from __future__ import annotations

import re


def _parse_srt_time(raw: str) -> float | None:
    """``00:01:02,500`` (srt) or ``01:02.500`` / ``00:01:02.500`` (vtt) → seconds."""
    raw = raw.strip()
    match = re.match(r"(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})", raw)
    if not match:
        return None
    hours = match.group(1) or "0"
    minutes, seconds, frac = match.group(2), match.group(3), match.group(4)
    millis = int(frac.ljust(3, "0")[:3])
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000.0


def _parse_ass_time(raw: str) -> float | None:
    """``0:01:02.50`` (ASS centisecond) → seconds."""
    raw = raw.strip()
    match = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{2})", raw)
    if not match:
        return None
    hours, minutes, seconds, centis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centis) / 100.0


def _strip_ass_override(text: str) -> str:
    """Remove ASS inline override blocks (``{\\an8}`` etc.) from cue text."""
    return re.sub(r"\{[^}]*\}", "", text).strip()


def _is_srt_cue_line(line: str) -> bool:
    return "-->" in line


def _parse_srt_like(text: str) -> list[dict]:
    """Parse srt/vtt content: numbered or bare blocks with ``a --> b``."""
    cues: list[dict] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Skip vtt cue identifiers / srt ordinal numbers before the timing line.
        timing_idx = next((i for i, ln in enumerate(lines) if _is_srt_cue_line(ln)), None)
        if timing_idx is None:
            continue
        left, _, right = lines[timing_idx].partition("-->")
        start = _parse_srt_time(left)
        end = _parse_srt_time(right.split()[0]) if right.split() else None
        body = " ".join(lines[timing_idx + 1 :]).strip()
        if start is None or end is None or not body:
            continue
        cues.append({"start": round(start, 3), "end": round(end, 3), "text": body})
    return cues


def _parse_ass(text: str) -> list[dict]:
    """Parse ASS/SSA ``Dialogue:`` lines (Format order resolved per file)."""
    fmt: list[str] | None = None
    cues: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("format:"):
            fmt = [f.strip().lower() for f in stripped.split(":", 1)[1].split(",")]
            continue
        if not stripped.lower().startswith("dialogue:"):
            continue
        payload = stripped.split(":", 1)[1].lstrip()
        if fmt is None:
            # ASS default field order per the spec.
            fmt = [
                "layer",
                "start",
                "end",
                "style",
                "name",
                "marginl",
                "marginr",
                "marginv",
                "effect",
                "text",
            ]
        parts = payload.split(",", len(fmt) - 1)
        if len(parts) < len(fmt):
            continue
        fields = dict(zip(fmt, parts))
        start = _parse_ass_time(fields.get("start", ""))
        end = _parse_ass_time(fields.get("end", ""))
        body = _strip_ass_override(fields.get("text", "")).replace("\\N", " ")
        if start is None or end is None or not body:
            continue
        cues.append({"start": round(start, 3), "end": round(end, 3), "text": body})
    return cues


def parse_subtitle_text(text: str, *, format_hint: str = "") -> list[dict]:
    """Parse subtitle text into ``[{"start", "end", "text"}]`` (seconds).

    ``format_hint`` may be a file extension (``.srt`` / ``.vtt`` /
    ``.ass`` / ``.ssa``); without it the format is auto-detected.
    Returns an empty list when nothing parses.
    """
    hint = format_hint.lower().lstrip(".")
    if hint in {"ass", "ssa"}:
        return _parse_ass(text)
    if hint == "vtt":
        return _parse_srt_like(text.removeprefix("\ufeff").removeprefix("WEBVTT"))
    if hint == "srt":
        return _parse_srt_like(text)
    stripped = text.lstrip("\ufeff").lstrip()
    if stripped.upper().startswith("WEBVTT"):
        return _parse_srt_like(stripped)
    if any(ln.strip().lower().startswith("dialogue:") for ln in text.splitlines()):
        return _parse_ass(text)
    if "-->" in text:
        return _parse_srt_like(text)
    return []
