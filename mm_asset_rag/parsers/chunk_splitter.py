"""Markdown / text heading splitter for PDF chunks.

The v3 PDF parser produced one chunk per page. Long Chinese PDFs
(联宝 媒眼 / 联宝 ESG / Codex 全景指南) pack several distinct
sections into one page, so a single chunk dilutes the BM25 signal
when the user queries a specific sub-topic. ``split_by_heading``
walks the chunk text and emits a new ``ParsedDocument`` per
detected section, with the heading as ``metadata.section``.

Detection rules (intentionally conservative — false positives are
worse than misses here):

1. Markdown-style ATX headings: lines starting with 1-3 ``#`` chars
   (no leading whitespace). Most PaddleOCR-VL outputs preserve this
   convention; PyMuPDF text output often doesn't.
2. A font-size heuristic on PyMuPDF's ``page.get_text("dict")``
   output: spans whose ``size`` is at least 1.4x the page median
   are treated as headings. (Size threshold configurable via
   ``HEADING_SIZE_RATIO``; default 1.4x).
3. Standalone Chinese / English lines (< 80 chars) followed by a
   blank line are also candidates — the fallback when the markdown
   markers are absent. Conservative length cap keeps body sentences
   out of the section bucket.

The function is pure: it accepts text + optional font-size metadata
and returns a list of ``(section_name, body_text)`` tuples. The
caller wraps each into a ``ParsedDocument``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ATX_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
_CJK_RE = re.compile(r"[一-鿿]")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")
HEADING_SIZE_RATIO = 1.4
MAX_HEADING_LEN = 80


@dataclass
class Section:
    """A heading + its body text. Empty heading means 'no heading —
    body continues from the previous section or from the start'."""

    heading: str
    body: str


def split_by_heading(text: str, *, font_sizes: list[float] | None = None) -> list[Section]:
    """Walk ``text`` and emit a :class:`Section` per heading.

    ``font_sizes``, when provided, is a parallel list of font sizes
    for each line in the text (one entry per line, after splitting
    on ``\n``). Lines with a font size at least
    ``HEADING_SIZE_RATIO`` x the median are treated as headings.
    """
    if not text or not text.strip():
        return [Section(heading="", body=text or "")]
    sections: list[Section] = []
    body_lines: list[str] = []
    current_heading = ""

    lines = text.splitlines()
    median_size = _median(font_sizes) if font_sizes else None

    for i, line in enumerate(lines):
        if _is_heading(line, lines, i, font_sizes, median_size):
            if body_lines or current_heading:
                sections.append(
                    Section(heading=current_heading, body="\n".join(body_lines).strip())
                )
            current_heading = _clean_heading(line)
            body_lines = []
        else:
            body_lines.append(line)
    sections.append(Section(heading=current_heading, body="\n".join(body_lines).strip()))
    return [s for s in sections if s.body or s.heading]


def _is_heading(
    line: str,
    all_lines: list[str],
    idx: int,
    font_sizes: list[float] | None,
    median_size: float | None,
) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # 1. ATX markdown style
    if _ATX_HEADING_RE.match(stripped):
        return True
    # 2. Font size heuristic (PyMuPDF dict output)
    if font_sizes and median_size and idx < len(font_sizes):
        size = font_sizes[idx]
        if size and size >= median_size * HEADING_SIZE_RATIO and len(stripped) <= MAX_HEADING_LEN:
            return True
    # 3. Standalone short line followed by a blank line. Conservative
    #    pattern: both boundaries must be blank for the line to look
    #    like a heading. The single-line case (a PDF paragraph
    #    rendered as one long line) is NOT a heading — falling into
    #    this branch for arbitrary text is the main source of
    #    false-positive splits, so we require both blank above and
    #    below to take it.
    if len(stripped) > MAX_HEADING_LEN:
        return False
    if stripped[-1:] in (".", "。", "?", "!", ",", ";"):
        return False
    next_blank = idx + 1 < len(all_lines) and not all_lines[idx + 1].strip()
    prev_blank = idx == 0 or not all_lines[idx - 1].strip()
    # Both blank = clear heading pattern.
    if next_blank and prev_blank:
        return True
    # Single boundary blank only — accept if a body line follows
    # (next not blank) and we already saw another heading. This catches
    # consecutive heading lines ("# A\n# B\nbody") where the second
    # heading is followed by body without a blank in between.
    return bool(next_blank and idx > 0 and _ATX_HEADING_RE.match(all_lines[idx - 1].strip() or ""))


def _clean_heading(line: str) -> str:
    """Strip ``#`` markers and surrounding whitespace from a heading line."""
    m = _ATX_HEADING_RE.match(line.strip())
    if m:
        return m.group(2).strip()
    return line.strip()


def _median(values: list[float]) -> float | None:
    cleaned = [v for v in values if v]
    if not cleaned:
        return None
    s = sorted(cleaned)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]
