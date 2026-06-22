"""Tests for the parser / embedder / backend registries."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_asset_rag.protocols import Embedder, Parser, VectorBackend, runtime_checkable
from mm_asset_rag.registry import (
    Registry,
    embedders,
    parsers,
    backends,
    register_embedder,
    register_parser,
)


# ─── Stub implementations satisfying the Protocols ──────────────────────


class StubPdfParser:
    name = "stub_pdf"
    source_type = "pdf"

    def parse(self, asset, **options):
        return []


class StubAudioParser:
    name = "stub_audio"
    source_type = "audio"

    def parse(self, asset, **options):
        return []


class StubTextEmbedder:
    name = "stub_text"
    modality = "text"

    def dim(self) -> int:
        return 4

    def embed(self, content) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, contents) -> list[list[float]]:
        return [self.embed(c) for c in contents]


# ─── Registry basics ─────────────────────────────────────────────────────


def test_registry_register_and_get():
    reg: Registry[str] = Registry("test")
    reg.register("k1", "v1")
    assert reg.get("k1") == "v1"
    assert "k1" in reg
    assert len(reg) == 1


def test_registry_rejects_duplicate_without_replace():
    reg: Registry[str] = Registry("test")
    reg.register("k1", "v1")
    with pytest.raises(ValueError, match="already registered"):
        reg.register("k1", "v2")
    # replace=True overrides
    reg.register("k1", "v2", replace=True)
    assert reg.get("k1") == "v2"


def test_registry_missing_key_lists_available():
    reg: Registry[str] = Registry("test")
    reg.register("alpha", "x")
    reg.register("beta", "y")
    with pytest.raises(KeyError) as excinfo:
        reg.get("gamma")
    assert "alpha" in str(excinfo.value)
    assert "beta" in str(excinfo.value)


def test_registry_compound_key():
    """Parsers and embedders use a (modality, name) tuple key."""
    reg: Registry[str] = Registry("test")
    reg.register(("a", "1"), "x")
    reg.register(("a", "2"), "y")
    reg.register(("b", "1"), "z")
    assert len(reg) == 3
    assert reg.get(("a", "1")) == "x"


# ─── Protocol structural typing ─────────────────────────────────────────


def test_parser_protocol_accepts_structural_conformers():
    assert isinstance(StubPdfParser(), Parser)
    assert isinstance(StubAudioParser(), Parser)


def test_embedder_protocol_accepts_structural_conformers():
    assert isinstance(StubTextEmbedder(), Embedder)


def test_non_conformer_is_not_a_parser():
    class NotAParser:
        name = "x"
        # Missing source_type and parse()

    assert not isinstance(NotAParser(), Parser)


# ─── Module-level registries ────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Snapshot the module-level registries around each test."""
    saved_parsers = parsers.all()
    saved_embedders = embedders.all()
    saved_backends = backends.all()
    # Clear so each test starts fresh.
    parsers._items.clear()
    embedders._items.clear()
    backends._items.clear()
    yield
    parsers._items.clear()
    embedders._items.clear()
    backends._items.clear()
    parsers._items.update(saved_parsers)
    embedders._items.update(saved_embedders)
    backends._items.update(saved_backends)


def test_register_parser_indexes_by_source_type_and_name():
    register_parser(StubPdfParser())
    register_parser(StubAudioParser())
    # Same source_type, different name — should be two separate entries.
    assert parsers.get(("pdf", "stub_pdf")) is not None
    assert parsers.get(("audio", "stub_audio")) is not None
    # Missing pair is a clear KeyError.
    with pytest.raises(KeyError, match="pymupdf"):
        parsers.get(("pdf", "pymupdf"))


def test_register_embedder_indexes_by_modality_and_name():
    register_embedder(StubTextEmbedder())
    embedder = embedders.get(("text", "stub_text"))
    assert embedder.dim() == 4