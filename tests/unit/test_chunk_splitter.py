"""Tests for ``mm_asset_rag.parsers.chunk_splitter.split_by_heading``."""

from __future__ import annotations

from mm_asset_rag.parsers.chunk_splitter import Section, split_by_heading


def test_split_empty_text() -> None:
    assert split_by_heading("") == [Section(heading="", body="")]
    assert split_by_heading("   \n  ") == [Section(heading="", body="   \n  ")]


def test_split_no_headings() -> None:
    text = "This is a single paragraph.\nNo headings here."
    sections = split_by_heading(text)
    # Single section with empty heading + the full body.
    assert len(sections) == 1
    assert sections[0].heading == ""
    assert sections[0].body == text


def test_split_atx_markdown() -> None:
    text = "# Introduction\nFirst paragraph.\n\n# Methods\nMethods paragraph."
    sections = split_by_heading(text)
    headings = [s.heading for s in sections]
    assert "Introduction" in headings
    assert "Methods" in headings
    # First section's body is the first paragraph only.
    intro = next(s for s in sections if s.heading == "Introduction")
    assert "First paragraph" in intro.body
    assert "Methods paragraph" not in intro.body


def test_split_font_size_heuristic() -> None:
    """Large font spans are treated as headings when font_sizes is provided."""
    # Body lines are intentionally long (> 80 chars) so they don't
    # qualify as headings themselves via the "standalone short line"
    # fallback rule.
    body1 = "First body line with extra padding to make it exceed the eighty char heading cap."
    body2 = "Second body line with extra padding to make it exceed the eighty char heading cap."
    text = f"Big Heading\n\n{body1}\n{body2}"
    font_sizes = [16.0, 0.0, 10.0, 10.0]  # 0.0 = blank line, ignored in median
    sections = split_by_heading(text, font_sizes=font_sizes)
    assert sections[0].heading == "Big Heading"
    # The body of the first section should include the body text.
    first_body = next(s for s in sections if s.heading == "Big Heading").body
    assert "First body line" in first_body
    assert "Second body line" in first_body


def test_split_short_line_heuristic() -> None:
    """Short standalone lines (≤80 chars) followed by blank on both
    sides → heading. The body must be a long line so it doesn't
    itself qualify as a heading.
    """
    text = "联宝 ESG\n\n联宝 2026 财年正式启幕,本次新财年将围绕 ESG 与智能制造两大主题深入推进各项业务工作,相关负责人在开幕会议上明确表达了新一年的战略方向与重点投入领域。\n\n联宝 媒眼\n\n媒眼 2026 年第二期"
    sections = split_by_heading(text)
    # 双边界 blank:首行 + 末尾行都被识别为 heading
    assert "联宝 ESG" in [s.heading for s in sections]
    assert "联宝 媒眼" in [s.heading for s in sections]
    assert any("联宝 2026" in s.body for s in sections)


def test_split_ignores_body_sentences() -> None:
    """A short line with terminal punctuation is body, not heading."""
    text = "First sentence ends here.\n\nSecond sentence."
    sections = split_by_heading(text)
    # No heading detected — body keeps both sentences.
    assert all(s.heading == "" for s in sections)
    full_body = "\n".join(s.body for s in sections)
    assert "First sentence" in full_body
    assert "Second sentence" in full_body


def test_split_multiple_levels() -> None:
    text = "# Top\nIntro.\n\n## Sub\nSub body.\n\n## Sub 2\nSub 2 body."
    sections = split_by_heading(text)
    assert "Top" in [s.heading for s in sections]
    # Subheadings detected as separate sections.
    assert "Sub" in [s.heading for s in sections]
    assert "Sub 2" in [s.heading for s in sections]


def test_split_preserves_blank_lines_in_body() -> None:
    text = "# Title\n\nPara 1.\n\nPara 2."
    sections = split_by_heading(text)
    body = next(s for s in sections if s.heading == "Title").body
    assert "Para 1" in body
    assert "Para 2" in body


def test_split_bbox_none_when_not_supplied() -> None:
    """No ``line_bboxes`` → every Section.bbox stays ``None``."""
    sections = split_by_heading("# H\nbody line one\nbody line two")
    assert all(s.bbox is None for s in sections)


def test_split_bbox_unions_body_lines() -> None:
    """``line_bboxes`` parallel to lines → each section's bbox is the
    union of its body lines' bboxes (heading bbox excluded)."""
    # Lines: ["# H", "aa", "bb", "# I", "cc"]
    text = "# H\naa\nbb\n# I\ncc"
    line_bboxes = [
        (0, 0, 50, 10),  # heading "# H"
        (0, 12, 30, 22),  # "aa"
        (0, 24, 30, 34),  # "bb"
        (0, 40, 50, 50),  # heading "# I"
        (0, 52, 30, 62),  # "cc"
    ]
    sections = split_by_heading(text, line_bboxes=line_bboxes)
    h = next(s for s in sections if s.heading == "H")
    i = next(s for s in sections if s.heading == "I")
    # H's body = "aa" + "bb" → union of their bboxes
    assert h.bbox == (0, 12, 30, 34)
    # I's body = "cc"
    assert i.bbox == (0, 52, 30, 62)


def test_split_bbox_tolerates_none_entries() -> None:
    """A ``None`` bbox entry (blank line / unknown) is skipped, not unioned."""
    body1 = "First body line long enough to exceed the eighty char heading cap for safety."
    body2 = "Second body line long enough to exceed the eighty char heading cap for safety."
    text = f"# H\n{body1}\n\n{body2}"
    line_bboxes = [
        (0, 0, 50, 10),
        (0, 12, 30, 22),
        None,  # blank line
        (0, 30, 30, 40),
    ]
    sections = split_by_heading(text, line_bboxes=line_bboxes)
    h = next(s for s in sections if s.heading == "H")
    assert h.bbox == (0, 12, 30, 40)


# ─── recursive_split + split_with_recursion ──────────────────────────────

from mm_asset_rag.parsers.chunk_splitter import (  # noqa: E402
    _char_count_tokens,
    recursive_split,
    split_with_recursion,
)


def test_recursive_split_short_body_returns_single() -> None:
    """A body under the target budget is returned as one piece."""
    body = "Short paragraph. No need to split."
    assert recursive_split(body, target_tokens=500, max_tokens=800) == [body]


def test_recursive_split_caps_long_body() -> None:
    """A body over max_tokens is cut into pieces, none exceeding max."""
    # ~3.5 chars/token → 1000 tokens ≈ 3500 chars; max=200 forces ~17 chunks.
    body = "句子。" * 1000  # 4000 chars, ~1143 tokens
    pieces = recursive_split(body, target_tokens=100, max_tokens=200, overlap_tokens=10)
    assert len(pieces) > 1
    for p in pieces:
        assert _char_count_tokens(p) <= 200 + 1  # cap + slack from hard-cut


def test_recursive_split_prefers_separator_boundaries() -> None:
    """Cuts land on paragraph/sentence boundaries, not mid-word."""
    body = "First sentence here.\n\nSecond sentence here.\n\nThird sentence here."
    # target just under one sentence so each sentence is its own chunk.
    pieces = recursive_split(body, target_tokens=8, max_tokens=20, overlap_tokens=0)
    # No piece should start mid-word (no leading lowercase fragment from
    # a cut inside "sentence").
    for p in pieces:
        assert not p.startswith("entence")  # would mean cut inside "sentence"


def test_recursive_split_overlap_present() -> None:
    """Adjacent chunks share an overlap tail from the previous chunk."""
    body = "句子一。" * 400 + "句子二。" * 400  # long enough to split
    pieces = recursive_split(body, target_tokens=50, max_tokens=100, overlap_tokens=15)
    assert len(pieces) >= 2
    # piece[1] should begin with a suffix of piece[0] (the overlap tail).
    first, second = pieces[0], pieces[1]
    overlap_found = any(second.startswith(first[-i:]) for i in range(1, min(len(first), 60) + 1))
    assert overlap_found, f"no overlap between chunks: {first[-30:]!r} → {second[:30]!r}"


def test_recursive_split_empty_body() -> None:
    assert recursive_split("", target_tokens=500, max_tokens=800) == [""]


def test_split_with_recursion_inherits_heading_and_bbox() -> None:
    """Sub-chunks keep the parent section's heading + bbox."""
    # One heading + a body long enough to exceed the target.
    body = "正文内容一。" * 300
    text = f"# 标题\n{body}"
    line_bboxes = [(0, 0, 50, 10)] + [(0, 12, 30, 22)] * 300
    sections = split_with_recursion(
        text,
        line_bboxes=line_bboxes,
        target_tokens=50,
        max_tokens=100,
        overlap_tokens=10,
    )
    assert len(sections) > 1
    # All sub-chunks share the parent heading + bbox.
    assert all(s.heading == "标题" for s in sections)
    assert all(s.bbox == (0, 12, 30, 22) for s in sections)


def test_split_with_recursion_short_section_preserved() -> None:
    """A section under the target stays one chunk (no needless split)."""
    text = "# H\nShort body."
    sections = split_with_recursion(text, target_tokens=500, max_tokens=800)
    assert len(sections) == 1
    assert sections[0].heading == "H"
    assert sections[0].body == "Short body."


def test_split_with_recursion_drops_empty_body() -> None:
    """A heading with no body is skipped (no placeholder chunk)."""
    text = "# H1\n\n# H2\nReal body."
    sections = split_with_recursion(text, target_tokens=500, max_tokens=800)
    # H1 has no body → dropped; only H2's body remains.
    assert all(s.body.strip() for s in sections)
    assert any(s.heading == "H2" for s in sections)


def test_split_with_recursion_multiple_headings_split_independently() -> None:
    """Each heading's body is sized independently."""
    long_body = "段落。" * 200
    text = f"# A\n{long_body}\n\n# B\n{long_body}"
    sections = split_with_recursion(text, target_tokens=30, max_tokens=80, overlap_tokens=5)
    # Both headings appear across the sub-chunks.
    headings = {s.heading for s in sections}
    assert "A" in headings and "B" in headings
    assert len(sections) > 2  # each long body split into multiple
