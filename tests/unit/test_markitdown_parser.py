"""Tests for the MarkItDown format adapter (``mm_asset_rag.parsers.markitdown_parser``).

MarkItDown is a core dependency, but these tests stub the ``markitdown``
import so the adapter is exercised without the real (and slow) Office
converter installs in CI. The stub injects a fake ``MarkItDown`` whose
``convert`` returns an object carrying ``.text_content`` (the markdown
string), mirroring the real API. This mirrors the ``test_docling_parser``
style of mocking the converter.

The adapter's defining behaviour — decoding docx/pptx base64 data-URL
image refs to disk and rewriting them to ``images/...`` — is exercised
with synthetic data URLs so no real Office files are needed.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mm_asset_rag.assets import Asset
from mm_asset_rag.parsers.markitdown_parser import build_ir_markitdown

# A 1x1 transparent PNG used as the data-URL payload in several tests so
# the decoded bytes are a real, readable image file.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _data_url(png: bytes = _PNG_BYTES) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _asset(tmp_home: Path, *, asset_id: str = "markitdown_test") -> Asset:
    return Asset(
        asset_id=asset_id,
        title="MarkItDown Test",
        source_type="document",
        relative_path="doc.docx",
        tags=["t"],
        asset_dir=tmp_home,
    )


def _install_markitdown_stub(monkeypatch, text_content: str) -> None:
    """Inject a fake ``markitdown`` package whose ``convert`` returns an
    object with ``.text_content`` == ``text_content``. ``build_ir_markitdown``
    calls ``MarkItDown().convert(asset.file_path, keep_data_uris=True)``;
    the path need not exist because the converter is stubbed. The real
    adapter passes ``keep_data_uris=True`` so MarkItDown keeps the base64
    payload (it strips data URLs to ``...`` by default); the stub ignores
    the kwarg and returns the canned ``text_content`` verbatim."""
    conv_res = SimpleNamespace(text_content=text_content)
    converter = MagicMock()
    converter.convert.return_value = conv_res

    fake_module = SimpleNamespace(MarkItDown=lambda: converter)
    # Drop any cached real markitdown first (a dev with it installed would
    # otherwise shadow the stub).
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)


# ─── build_ir_markitdown: text / headings / tables ──────────────────────────


def test_build_ir_markitdown_turns_paragraphs_into_blocks(tmp_home, monkeypatch) -> None:
    """Plain paragraphs split on blank lines become body Blocks."""
    md = "First paragraph.\n\nSecond paragraph.\n\nThird."
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert ir.parser == "markitdown"
    assert [b.text for b in ir.blocks] == ["First paragraph.", "Second paragraph.", "Third."]
    assert all(b.heading == "" for b in ir.blocks)


def test_build_ir_markitdown_promotes_atx_headings(tmp_home, monkeypatch) -> None:
    """ATX heading lines become heading Blocks (heading + level set)."""
    md = "# Title\n\n## Subsection\n\nbody under the subsection"
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert ir.blocks[0].heading == "Title"
    assert ir.blocks[0].level == 1
    assert ir.blocks[1].heading == "Subsection"
    assert ir.blocks[1].level == 2
    assert ir.blocks[2].heading == ""
    assert ir.blocks[2].text == "body under the subsection"


def test_build_ir_markitdown_keeps_markdown_tables_as_blocks(tmp_home, monkeypatch) -> None:
    """A markdown table survives as a body block (no special handling)."""
    md = "| a | b |\n|---|---|\n| 1 | 2 |"
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert len(ir.blocks) == 1
    assert ir.blocks[0].text.startswith("| a | b |")


def test_build_ir_markitdown_empty_document_yields_no_blocks(tmp_home, monkeypatch) -> None:
    """An empty text_content produces no blocks and no export file."""
    _install_markitdown_stub(monkeypatch, "")

    ir = build_ir_markitdown(_asset(tmp_home))
    assert ir.blocks == []
    assert ir.markdown_paths == []


def test_build_ir_markitdown_writes_export_markdown(tmp_home, monkeypatch) -> None:
    """A non-empty parse writes ``markitdown_export.md`` for the answer layer."""
    _install_markitdown_stub(monkeypatch, "# Hi\n\nbody")
    ir = build_ir_markitdown(_asset(tmp_home))
    assert len(ir.markdown_paths) == 1
    assert Path(ir.markdown_paths[0]).name == "markitdown_export.md"
    assert Path(ir.markdown_paths[0]).exists()


# ─── the defining behaviour: data-URL image decode + ref rewrite ────────────


def test_build_ir_markitdown_decodes_data_url_image_to_disk(tmp_home, monkeypatch) -> None:
    """A base64 data-URL image ref is decoded to images/ and the ref rewritten.

    This is the core MarkItDown-specific behaviour: docx/pptx embed images
    as ``![](data:image/png;base64,...)`` but the image-association regex
    (``_MD_IMAGE_RE``) only matches ``images/...`` paths, so without this
    rewrite the image would be invisible to retrieval / answer-time
    attachment.
    """
    md = f"Intro paragraph.\n\n![]({_data_url()})"
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert len(ir.images) == 1
    rel = ir.images[0].path
    assert rel.startswith("images/markitdown_")
    assert rel.endswith(".png")
    # The decoded file is on disk under parsed/<id>/images/.
    img_abs = Path(ir.markdown_paths[0]).parent / rel
    assert img_abs.exists()
    assert img_abs.read_bytes() == _PNG_BYTES
    # The ref in the block text was rewritten to the relative path — the
    # original data URL must be gone.
    assert "data:image" not in ir.blocks[-1].text
    assert f"![]({rel})" in ir.blocks[-1].text
    # images_dir is populated so the answer layer can locate the dir.
    assert ir.images_dir.endswith("images")


def test_build_ir_markitdown_decodes_multiple_data_url_images(tmp_home, monkeypatch) -> None:
    """Multiple data-URL images in one block each decode to their own file."""
    md = f"![]({_data_url()})\n\n![]({_data_url()})"
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert len(ir.images) == 2
    # Same payload → same md5 hash → same filename; both refs point at it.
    paths = {img.path for img in ir.images}
    assert len(paths) == 1  # deduped on disk by content hash


def test_build_ir_markitdown_skips_invalid_base64_payload(tmp_home, monkeypatch) -> None:
    """A malformed base64 payload drops just that image, not the whole parse."""
    md = "body text here\n\n![](data:image/png;base64,@@@not-valid-b64@@@)"
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert ir.images == []
    # The body block survives; the bad ref was removed (replaced with "").
    assert ir.blocks[0].text == "body text here"


def test_build_ir_markitdown_keeps_relative_path_image_refs_as_is(tmp_home, monkeypatch) -> None:
    """html-style relative-path refs (``![](relative.png)``) are passed through.

    v1 does not resolve them to images/ — they don't match the
    association regex, so no ImageRef is emitted, but the ref text is
    preserved in the block (not stripped). This matches the plan's
    explicit "html relative-path images: v1 skip" boundary.
    """
    md = "body\n\n![](assets/figure1.png)"
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert ir.images == []
    assert "![](assets/figure1.png)" in ir.blocks[-1].text


def test_build_ir_markitdown_data_url_mime_to_suffix(tmp_home, monkeypatch) -> None:
    """A jpeg data URL is saved with a .jpg suffix."""
    # 1x1 JPEG is fiddly to hand-build; reuse the PNG bytes — the adapter
    # only keys off the mime subtype for the suffix, not the actual bytes.
    jpeg_url = "data:image/jpeg;base64," + base64.b64encode(_PNG_BYTES).decode()
    md = f"![]({jpeg_url})"
    _install_markitdown_stub(monkeypatch, md)

    ir = build_ir_markitdown(_asset(tmp_home))
    assert len(ir.images) == 1
    assert ir.images[0].path.endswith(".jpg")


# ─── end-to-end: image reaches chunk meta via ir_to_documents ───────────────


def test_build_ir_markitdown_data_url_image_reaches_chunk_meta(tmp_home, monkeypatch) -> None:
    """End-to-end: a data-URL image survives ir_to_documents into chunk meta.

    The regression this guards: if the ref rewrite were missing,
    ``extract_markdown_image_refs`` would find no ``images/...`` ref in the
    chunk body and the chunk would carry no images — the picture decoded
    to disk but invisible to retrieval / answer-time attachment.
    """
    md = f"a paragraph of body text long enough to chunk\n\n![]({_data_url()})"
    _install_markitdown_stub(monkeypatch, md)

    from mm_asset_rag.parsers.document_ir import ir_to_documents

    ir = build_ir_markitdown(_asset(tmp_home))
    docs = ir_to_documents(ir)
    assert docs, "expected at least one chunk"
    chunk_image_paths = [img["path"] for d in docs for img in (d.metadata.get("images") or [])]
    assert any(p.startswith("images/markitdown_") for p in chunk_image_paths), (
        f"data-URL image did not reach any chunk meta.images: {chunk_image_paths}"
    )


def test_build_ir_markitdown_passes_keep_data_uris_to_convert(tmp_home, monkeypatch) -> None:
    """The adapter must call ``convert(..., keep_data_uris=True)`` — without
    it MarkItDown's markdownify layer strips data URLs to ``data:...;base64...``,
    discarding the base64 payload this adapter decodes. This pins the kwarg
    so a future refactor doesn't silently drop embedded images."""
    conv_res = SimpleNamespace(text_content="body")
    converter = MagicMock()
    converter.convert.return_value = conv_res
    fake_module = SimpleNamespace(MarkItDown=lambda: converter)
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    build_ir_markitdown(_asset(tmp_home))
    assert converter.convert.called
    _, kwargs = converter.convert.call_args
    assert kwargs.get("keep_data_uris") is True


# ─── missing-install error path ─────────────────────────────────────────────


def test_parse_with_markitdown_raises_friendly_when_package_missing(tmp_home, monkeypatch) -> None:
    """Without markitdown importable, parse_with_markitdown raises a
    RuntimeError pointing at the install command — not a raw ImportError."""
    monkeypatch.setitem(sys.modules, "markitdown", None)

    from mm_asset_rag.parsers.pdf_parser import parse_with_markitdown

    with pytest.raises(RuntimeError, match="markitdown"):
        parse_with_markitdown(_asset(tmp_home))
