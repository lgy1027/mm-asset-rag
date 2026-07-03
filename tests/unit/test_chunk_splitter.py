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
