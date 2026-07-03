"""Tests for ``mm_asset_rag.cli.command_reindex`` interactive prompt.

The reindex command drops and rebuilds the Qdrant collection — the
prompt is the only thing standing between a typo and a destroyed
index. ``--yes`` exists for CI / scripted reindex. We pin both
behaviours here.
"""

from __future__ import annotations

import argparse

import pytest

from mm_asset_rag.cli import command_reindex


def _args(**overrides) -> argparse.Namespace:
    base = dict(text_only=False, image_only=False, yes=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_reindex_aborts_on_empty_input(monkeypatch, capsys) -> None:
    """Empty stdin (e.g. CI) is treated as 'no'."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    called = {"reindex": 0}
    monkeypatch.setattr(
        "mm_asset_rag.service.get_service",
        lambda: type(
            "S",
            (),
            {
                "reindex": lambda self, **kw: (
                    called.__setitem__("reindex", called["reindex"] + 1) or ("ok",)
                )
            },
        )(),
    )
    with pytest.raises(SystemExit) as exc:
        command_reindex(_args())
    # ``SystemExit("aborted")`` carries the message in ``.value``.
    assert str(exc.value) == "aborted"
    assert called["reindex"] == 0


def test_reindex_aborts_on_n(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "n")
    called = {"reindex": 0}
    monkeypatch.setattr(
        "mm_asset_rag.service.get_service",
        lambda: type(
            "S",
            (),
            {
                "reindex": lambda self, **kw: (
                    called.__setitem__("reindex", called["reindex"] + 1) or ("ok",)
                )
            },
        )(),
    )
    with pytest.raises(SystemExit):
        command_reindex(_args())
    assert called["reindex"] == 0


def test_reindex_yes_flag_skips_prompt(monkeypatch) -> None:
    """``--yes`` runs straight through without consulting stdin."""

    def _fail_input(_):
        raise AssertionError("input() should not be called with --yes")

    monkeypatch.setattr("builtins.input", _fail_input)
    called = {"reindex": 0, "text": 0, "image": 0}
    fake_service = type(
        "S",
        (),
        {
            "reindex": lambda self, text_only, image_only: (
                called.__setitem__("text", called["text"] + (0 if text_only else 1)),
                called.__setitem__("image", called["image"] + (0 if image_only else 1)),
                called.__setitem__("reindex", called["reindex"] + 1),
                ("text:ok", "image:ok"),
            )[-1]
        },
    )()
    monkeypatch.setattr("mm_asset_rag.service.get_service", lambda: fake_service)
    command_reindex(_args(yes=True))
    assert called["reindex"] == 1
    assert called["text"] == 1
    assert called["image"] == 1


def test_reindex_image_only_passes_through(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "y")
    called = {"text_only": None, "image_only": None}
    fake_service = type(
        "S",
        (),
        {
            "reindex": lambda self, text_only, image_only: (
                called.__setitem__("text_only", text_only),
                called.__setitem__("image_only", image_only),
                ("image:ok",),
            )[-1]
        },
    )()
    monkeypatch.setattr("mm_asset_rag.service.get_service", lambda: fake_service)
    command_reindex(_args(image_only=True))
    assert called["text_only"] is False
    assert called["image_only"] is True


def test_reindex_lock_held_error(monkeypatch) -> None:
    """``QdrantLockHeldError`` should surface as a clear SystemExit."""
    from mm_asset_rag.backends.qdrant_backend import QdrantLockHeldError

    monkeypatch.setattr("builtins.input", lambda _: "y")

    class _LockS:
        def reindex(self, **kw):
            raise QdrantLockHeldError("locked by another process")

    monkeypatch.setattr("mm_asset_rag.service.get_service", lambda: _LockS())
    with pytest.raises(SystemExit) as exc:
        command_reindex(_args())
    assert "locked by another process" in str(exc.value)
