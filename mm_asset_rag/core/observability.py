"""Small process-local counters for retrieval and evidence decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from .settings import get_settings


class RuntimeMetrics:
    """Aggregate operational events without retaining requests or document content."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._retrieval: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "count": 0,
                "elapsed_ms": 0,
                "max_elapsed_ms": 0,
                "candidates": 0,
                "returned": 0,
                "reasons": Counter(),
            }
        )
        self._refusals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"count": 0, "candidates": 0}
        )

    def record_retrieval(
        self, *, route: str, elapsed_ms: int, candidates: int, returned: int, reason: str
    ) -> None:
        with self._lock:
            row = self._retrieval[route]
            row["count"] = int(row["count"]) + 1
            row["elapsed_ms"] = int(row["elapsed_ms"]) + max(elapsed_ms, 0)
            row["max_elapsed_ms"] = max(int(row["max_elapsed_ms"]), max(elapsed_ms, 0))
            row["candidates"] = int(row["candidates"]) + max(candidates, 0)
            row["returned"] = int(row["returned"]) + max(returned, 0)
            row["reasons"][reason] += 1  # type: ignore[index]

    def record_refusal(self, *, reason: str, candidates: int) -> None:
        with self._lock:
            row = self._refusals[reason]
            row["count"] += 1
            row["candidates"] += max(candidates, 0)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "retrieval": {
                    route: {
                        "count": int(row["count"]),
                        "avg_elapsed_ms": round(int(row["elapsed_ms"]) / max(int(row["count"]), 1)),
                        "max_elapsed_ms": int(row["max_elapsed_ms"]),
                        "candidates": int(row["candidates"]),
                        "returned": int(row["returned"]),
                        "reasons": dict(row["reasons"]),
                    }
                    for route, row in self._retrieval.items()
                },
                "refusals": {reason: dict(row) for reason, row in self._refusals.items()},
            }


runtime_metrics = RuntimeMetrics()


# ─── Pluggable tracing ─────────────────────────────────────────────────────
#
# ``RuntimeMetrics`` above is process-local and always on. Tracing here is
# optional and provider-pluggable: the default ``NoOpTracer`` costs nothing,
# and a real provider (Langfuse via the ``[langfuse]`` extra) plugs in
# through ``Settings.tracing_provider`` without any call site importing the
# SDK. Code instruments against the ``Tracer`` / ``Span`` protocols only.


@runtime_checkable
class Span(Protocol):
    """One active observation (span or generation) inside a trace."""

    def set_attribute(self, key: str, value: object) -> None:
        """Attach or overwrite one metadata attribute on this observation."""
        ...

    def update(
        self,
        *,
        output: object | None = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, object] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Merge terminal details (output / token usage / error level)."""
        ...

    def score(
        self,
        *,
        name: str,
        value: float,
        comment: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Attach a quality score to the trace this observation belongs to.

        Scores are the primary Langfuse filtering/aggregation dimension
        (recall trends, answer quality), so evaluators report through this
        hook instead of logging.
        """
        ...

    def set_tags(self, tags: list[str]) -> None:
        """Set trace-level tags (replaces), e.g. ``collection:x mode:hybrid``."""
        ...


@runtime_checkable
class SpanContext(Protocol):
    """Context manager returned by ``Tracer.start_*``: enters to a ``Span``."""

    def __enter__(self) -> Span: ...

    def __exit__(self, *args: Any) -> bool: ...


@runtime_checkable
class Tracer(Protocol):
    """Factory for root observations. Implementations must be thread-safe."""

    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, object] | None = None,
        input: object | None = None,
    ) -> SpanContext:
        """Return a context manager yielding a ``Span``.

        ``input`` lands on the observation's Input field, which trace list
        views render as their Input column; ``attributes`` only appear in
        the detail metadata panel. Pass the user-facing payload (query,
        question) as ``input`` so lists are scannable.
        """
        ...

    def start_generation(
        self,
        name: str,
        *,
        model: str | None = None,
        input: object | None = None,
        model_parameters: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SpanContext:
        """Return a context manager yielding an LLM ``Span``."""
        ...

    def flush(self) -> None:
        """Block until all buffered observations are delivered."""
        ...


class _NoOpSpan:
    def set_attribute(self, key: str, value: object) -> None:
        return None

    def update(
        self,
        *,
        output: object | None = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, object] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        return None

    def score(
        self,
        *,
        name: str,
        value: float,
        comment: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        return None

    def set_tags(self, tags: list[str]) -> None:
        return None


_NO_OP_SPAN = _NoOpSpan()


class NoOpTracer:
    """Default tracer: context managers yield a shared do-nothing span."""

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, object] | None = None,
        input: object | None = None,
    ):
        yield _NO_OP_SPAN

    @contextmanager
    def start_generation(
        self,
        name: str,
        *,
        model: str | None = None,
        input: object | None = None,
        model_parameters: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ):
        yield _NO_OP_SPAN

    def flush(self) -> None:
        return None


_tracer: Tracer | None = None
_tracer_lock = Lock()


def _build_tracer(settings: Any) -> Tracer:
    provider = str(getattr(settings, "tracing_provider", "none") or "none").lower()
    if provider in {"", "none", "off", "disabled"}:
        return NoOpTracer()
    if provider == "langfuse":
        from .langfuse_tracer import build_langfuse_tracer

        tracer = build_langfuse_tracer(settings)
        if tracer is not None:
            return tracer
    return NoOpTracer()


def get_tracer() -> Tracer:
    """Return the process-wide tracer, building it from settings on first use.

    The built tracer is pinned to the settings snapshot at first call (mirrors
    the cached ``get_settings``); a later env change does not reconfigure it.
    """
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = _build_tracer(get_settings())
    return _tracer


def set_tracer(tracer: Tracer | None) -> None:
    """Install a tracer (tests) or reset to rebuild-from-settings (``None``)."""
    global _tracer
    _tracer = tracer
