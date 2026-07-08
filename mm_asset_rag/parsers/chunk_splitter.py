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

import itertools
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
    body continues from the previous section or from the start'.

    ``bbox`` is the page-space bounding box of the section's body text,
    as ``(x0, y0, x1, y1)`` in PDF points. ``None`` when the caller did
    not supply per-line bboxes (e.g. plain-text / markdown input with no
    geometry). Used by ``pdf_images.associate_images`` for the spatial
    figure-attachment fallback — chunks that sit next to a figure without
    naming it.
    """

    heading: str
    body: str
    bbox: tuple[float, float, float, float] | None = None


def _union_bbox(a, b):
    """Combine two ``(x0,y0,x1,y1)`` bboxes, ignoring ``None`` sides."""
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def split_by_heading(
    text: str,
    *,
    font_sizes: list[float] | None = None,
    line_bboxes: list | None = None,
) -> list[Section]:
    """Walk ``text`` and emit a :class:`Section` per heading.

    ``font_sizes``, when provided, is a parallel list of font sizes
    for each line in the text (one entry per line, after splitting
    on ``\\n``). Lines with a font size at least
    ``HEADING_SIZE_RATIO`` x the median are treated as headings.

    ``line_bboxes`` is an optional parallel list of per-line bboxes
    (``(x0, y0, x1, y1)`` tuples or ``None``), one per line in
    ``text.splitlines()``. When supplied, each emitted ``Section``
    carries the union bbox of its body lines in ``Section.bbox`` — used
    by the PyMuPDF path to spatially attach nearby figures. When omitted
    (the markdown / plain-text callers), ``bbox`` stays ``None`` and
    behavior is unchanged.
    """
    if not text or not text.strip():
        return [Section(heading="", body=text or "", bbox=None)]
    sections: list[Section] = []
    body_lines: list[str] = []
    body_bbox = None
    current_heading = ""

    lines = text.splitlines()
    median_size = _median(font_sizes) if font_sizes else None

    for i, line in enumerate(lines):
        if _is_heading(line, lines, i, font_sizes, median_size):
            if body_lines or current_heading:
                sections.append(
                    Section(
                        heading=current_heading,
                        body="\n".join(body_lines).strip(),
                        bbox=body_bbox,
                    )
                )
            current_heading = _clean_heading(line)
            body_lines = []
            body_bbox = None
        else:
            body_lines.append(line)
            if line_bboxes and i < len(line_bboxes):
                body_bbox = _union_bbox(body_bbox, line_bboxes[i])
    sections.append(
        Section(
            heading=current_heading,
            body="\n".join(body_lines).strip(),
            bbox=body_bbox,
        )
    )
    return [s for s in sections if s.body or s.heading]


# ─── Recursive token-budget splitting ────────────────────────────────────
#
# ``split_by_heading`` produces one chunk per detected heading. Long
# sections (a 10-page methods chapter under a single ``# Methods``) still
# become a single oversized chunk — the dense embedder truncates it, BM25
# signal dilutes, and the cross-encoder reranker is misled by long-body
# token frequency. ``recursive_split`` caps each chunk to a token budget
# with overlap, walking a hierarchy of separators so cuts land on natural
# boundaries (paragraph → line → sentence → char).
#
# Benchmark guidance (Vecta 7-strategy + arXiv 8-method surveys): a
# ~500-token recursive chunk wins on retrieval accuracy; >800 starts to
# dilute. Corpus- and model-agnostic: token counts default to a character
# approximation (token ≈ chars/3.5, mixed zh/en) with no tokenizer
# dependency, with an optional HF tokenizer upgrade path.

# Separator hierarchy, coarsest first. Paragraph break is preferred over
# line break over sentence end over space over char. CJK sentence-enders
# (fullwidth period/exclamation/question/semicolon) sit alongside their
# ASCII counterparts so Chinese prose splits on its own punctuation.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " ", "")

# Char-to-token ratio for the approximation. Pure CJK ≈ 1 char/token,
# pure ASCII ≈ 4 chars/token; 3.5 is the mixed-corpus compromise.
_CHARS_PER_TOKEN = 3.5


def _char_count_tokens(text: str) -> int:
    """Approximate token count from character length (model-agnostic)."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _make_token_counter(tokenizer_id: str | None) -> tuple[object | None, callable]:
    """Build a token-counting callable.

    Returns ``(counter, tokenizer_or_none)``. When ``tokenizer_id`` is
    set and importable, ``counter`` calls the real tokenizer; otherwise it
    falls back to the char approximation. The tokenizer object is returned
    so the caller can keep it alive (some tokenizers hold resources).
    """
    if not tokenizer_id:
        return None, _char_count_tokens
    try:
        from transformers import AutoTokenizer  # type: ignore[import]

        tok = AutoTokenizer.from_pretrained(tokenizer_id)

        def _count(text: str) -> int:
            return len(tok(text, add_special_tokens=False)["input_ids"])

        return tok, _count
    except Exception:
        # Missing transformers, bad id, network — degrade to char approx
        # rather than failing the whole parse.
        return None, _char_count_tokens


def _split_on_separator(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep``, keeping the separator attached to the
    preceding piece (so ``"a。b。"`` → ``["a。", "b。"]``, not ``["a", "b"]``).
    A trailing empty piece is dropped. The empty-string separator returns
    the whole text as one piece (char-level fallback uses a different path).
    """
    if sep == "":
        return [text] if text else []
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(text):
        # Greedy: find the next separator occurrence from i.
        nxt = text.find(sep, i)
        if nxt == -1:
            buf.append(text[i:])
            break
        end = nxt + len(sep)
        buf.append(text[i:end])
        parts.append("".join(buf))
        buf = []
        i = end
    if buf:
        parts.append("".join(buf))
    # Drop a trailing piece that is only the separator / whitespace.
    return [p for p in parts if p]


def recursive_split(
    body: str,
    *,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 60,
    count_tokens: callable = _char_count_tokens,
) -> list[str]:
    """Split ``body`` into chunks of ≈ ``target_tokens`` (≤ ``max_tokens``),
    with ``overlap_tokens`` of shared context between adjacent chunks.

    Walks :data:`_SEPARATORS` from coarsest to finest: a piece under
    ``target_tokens`` is emitted as-is; one over ``max_tokens`` is split
    with the next-finer separator. When every separator has been tried
    and a piece is still over ``max_tokens``, it is hard-cut at
    ``max_tokens`` characters-of-approx so no chunk is ever unbounded.

    Overlap is applied post-split: each chunk (after the first) is
    prefixed with the tail of the previous chunk, sized to
    ``overlap_tokens``. The overlap is pulled back to a separator
    boundary when possible so it does not start mid-word.

    Pure function; returns a non-empty list (``[body]`` when no split is
    needed or ``body`` is tiny).
    """
    if not body or not body.strip():
        return [body] if body is not None else []
    if count_tokens(body) <= target_tokens:
        return [body]

    def _split_recursive(text: str, sep_idx: int) -> list[str]:
        if count_tokens(text) <= target_tokens or sep_idx >= len(_SEPARATORS):
            return [text]
        sep = _SEPARATORS[sep_idx]
        pieces = _split_on_separator(text, sep)
        if len(pieces) <= 1:
            # This separator found nothing to cut on; try the next finer one.
            return _split_recursive(text, sep_idx + 1)
        out: list[str] = []
        for piece in pieces:
            if count_tokens(piece) <= target_tokens:
                out.append(piece)
            else:
                out.extend(_split_recursive(piece, sep_idx + 1))
        return out

    chunks = _split_recursive(body, 0)
    # Merge tiny tail pieces back into the previous chunk when together
    # they stay under target — avoids degenerate 1-token chunks from a
    # long run of short sentences.
    merged: list[str] = []
    for chunk in chunks:
        if merged and count_tokens(merged[-1] + chunk) <= target_tokens:
            merged[-1] = merged[-1] + chunk
        else:
            merged.append(chunk)
    # Hard cap: anything still over max_tokens after all separators is
    # force-cut so the dense embedder never sees an unbounded payload.
    capped: list[str] = []
    for chunk in merged:
        if count_tokens(chunk) <= max_tokens:
            capped.append(chunk)
        else:
            # Char-level hard cut at the approx max boundary, then keep
            # splitting the remainder.
            limit = int(max_tokens * _CHARS_PER_TOKEN)
            i = 0
            while i < len(chunk):
                piece = chunk[i : i + limit]
                capped.append(piece)
                i += limit
    # Apply overlap between adjacent chunks.
    if overlap_tokens <= 0 or len(capped) <= 1:
        return capped
    out = [capped[0]]
    for prev, cur in itertools.pairwise(capped):
        tail = _tail_for_overlap(prev, overlap_tokens, count_tokens)
        out.append(tail + cur if tail else cur)
    return out


def _tail_for_overlap(text: str, overlap_tokens: int, count_tokens: callable) -> str:
    """Return the trailing slice of ``text`` sized to ≈ ``overlap_tokens``,
    pulled back to a separator boundary so the overlap does not start
    mid-word. Returns "" when the text is already shorter than the budget.
    """
    if not text:
        return ""
    limit = int(overlap_tokens * _CHARS_PER_TOKEN)
    if len(text) <= limit:
        return text
    start = len(text) - limit
    # Pull start back to the nearest separator so we don't cut mid-word.
    for sep in _SEPARATORS:
        if sep == "":
            continue
        idx = text.rfind(sep, 0, start)
        if idx != -1:
            start = idx + len(sep)
            break
    return text[start:]


def split_with_recursion(
    text: str,
    *,
    font_sizes: list[float] | None = None,
    line_bboxes: list | None = None,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 60,
    count_tokens: callable = _char_count_tokens,
) -> list[Section]:
    """Heading-aware split + recursive token-budget sizing.

    Two layers: :func:`split_by_heading` finds the structural boundaries
    (preserving heading text + bbox for figure attachment), then each
    section's body is :func:`recursive_split` to the token budget. Every
    sub-chunk inherits its parent section's heading and bbox — figure
    attachment (``associate_images``) keeps working unchanged, and the
    answer layer's per-hit image cap de-dupes the repeated attachments.

    Sections whose body is empty after stripping are dropped (same rule
    as :func:`split_by_heading`), so a bare heading with no following body
    does not pollute the index.
    """
    sections = split_by_heading(text, font_sizes=font_sizes, line_bboxes=line_bboxes)
    out: list[Section] = []
    for section in sections:
        body = section.body.strip() if section.body else ""
        if not body:
            continue
        if count_tokens(body) <= target_tokens:
            out.append(Section(heading=section.heading, body=body, bbox=section.bbox))
            continue
        for piece in recursive_split(
            body,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            count_tokens=count_tokens,
        ):
            piece = piece.strip()
            if not piece:
                continue
            out.append(Section(heading=section.heading, body=piece, bbox=section.bbox))
    return out


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
