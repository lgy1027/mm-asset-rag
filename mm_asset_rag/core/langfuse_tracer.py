"""Langfuse-backed ``Tracer`` implementation.

The ``langfuse`` SDK is an optional dependency (the ``[langfuse]`` extra)
and is imported lazily inside :func:`build_langfuse_tracer` so importing
this module never requires the SDK. Every other package only depends on
the provider-neutral protocols in :mod:`mm_asset_rag.core.observability`.
"""

from __future__ import annotations

import atexit
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .settings import Settings

log = logging.getLogger(__name__)


class _LangfuseSpan:
    """Adapter exposing a Langfuse OTel span/generation as our ``Span``."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def set_attribute(self, key: str, value: object) -> None:
        self._raw.update(metadata={key: value})

    def update(
        self,
        *,
        output: object | None = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, object] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if usage:
            kwargs["usage_details"] = {k: int(v) for k, v in usage.items()}
        if metadata:
            kwargs["metadata"] = metadata
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = status_message
        if kwargs:
            self._raw.update(**kwargs)

    def score(
        self,
        *,
        name: str,
        value: float,
        comment: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"name": name, "value": value}
        if comment:
            kwargs["comment"] = comment
        if metadata:
            kwargs["metadata"] = metadata
        self._raw.score(**kwargs)

    def set_tags(self, tags: list[str]) -> None:
        self._raw.update_trace(tags=list(tags))


class _TimedContext:
    """Context manager wrapping one ``start_as_current_*`` call.

    Records elapsed-ms and surfaced exceptions onto the observation so
    providers see latency and error level without call sites caring.
    """

    def __init__(self, client: Any, kind: str, name: str, kwargs: dict[str, Any]) -> None:
        self._client = client
        self._kind = kind
        self._name = name
        self._kwargs = kwargs
        self.span: _LangfuseSpan | None = None
        self._cm: Any = None

    def __enter__(self) -> _LangfuseSpan:
        try:
            starter = (
                self._client.start_as_current_generation
                if self._kind == "generation"
                else self._client.start_as_current_span
            )
            self._started = time.perf_counter()
            self._cm = starter(name=self._name, **self._kwargs)
            raw = self._cm.__enter__()
            self.span = _LangfuseSpan(raw)
            return self.span
        except Exception as exc:
            # Tracing must never break the pipeline: a failing span start
            # degrades to a do-nothing span for the rest of the block.
            log.warning("langfuse span start failed: %s; span disabled", exc)
            self.span = _LangfuseSpan(_FailingRawSpan())
            return self.span

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._cm is None:
            return False  # enter() already degraded; nothing to close
        try:
            elapsed_ms = int((time.perf_counter() - self._started) * 1000)
        except AttributeError:
            elapsed_ms = 0
        try:
            if self.span is not None:
                self.span.set_attribute("elapsed_ms", elapsed_ms)
                if exc is not None:
                    self.span.update(
                        level="ERROR",
                        status_message=f"{type(exc).__name__}: {exc}",
                    )
        except Exception as log_exc:
            # Attribute stamping is telemetry — never mask the block's own
            # exception (or convert a clean return into a failure).
            log.warning("langfuse span attribute update failed: %s", log_exc)
        finally:
            cm, self._cm = self._cm, None
            try:
                cm.__exit__(exc_type, exc, tb)
            except Exception as exit_exc:
                log.warning("langfuse span close failed: %s", exit_exc)
        return False


class _FailingRawSpan:
    """Stand-in raw span used when span start failed: every call no-ops."""

    def update(self, **kwargs: Any) -> None:
        return None

    def score(self, **kwargs: Any) -> None:
        return None

    def update_trace(self, **kwargs: Any) -> None:
        return None


class LangfuseTracer:
    """``Tracer`` implementation delegating to a Langfuse v3 client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, object] | None = None,
        input: object | None = None,
    ) -> _TimedContext:
        kwargs: dict[str, Any] = {}
        if attributes:
            kwargs["metadata"] = dict(attributes)
        if input is not None:
            kwargs["input"] = input
        return _TimedContext(self._client, "span", name, kwargs)

    def start_generation(
        self,
        name: str,
        *,
        model: str | None = None,
        input: object | None = None,
        model_parameters: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> _TimedContext:
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if input is not None:
            kwargs["input"] = input
        if model_parameters:
            kwargs["model_parameters"] = dict(model_parameters)
        merged: dict[str, object] = {}
        if metadata:
            merged.update(metadata)
        if merged:
            kwargs["metadata"] = merged
        return _TimedContext(self._client, "generation", name, kwargs)

    def flush(self) -> None:
        self._client.flush()


def build_langfuse_tracer(settings: Settings) -> LangfuseTracer | None:
    """Create a tracer from settings, or ``None`` when the SDK is unusable.

    Missing SDK, missing credentials, and client-construction failures all
    degrade to ``None`` so the caller falls back to ``NoOpTracer`` — tracing
    must never break retrieval or answering.
    """
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning(
            "tracing_provider=langfuse but the langfuse SDK is not installed; "
            "install the [langfuse] extra to enable tracing"
        )
        return None

    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if not public_key or not secret_key:
        log.warning(
            "tracing_provider=langfuse but langfuse_public_key/langfuse_secret_key "
            "are not set; tracing disabled"
        )
        return None

    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("langfuse client init failed: %s; tracing disabled", exc)
        return None

    # The parallel multi-query rewrite searches inside ThreadPoolExecutor
    # workers; without context propagation each child span becomes an orphan
    # root trace. The threading instrumentor patches ``pool.submit`` to carry
    # the OTel context across threads.
    try:
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor

        ThreadingInstrumentor().instrument()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("threading instrumentor failed: %s; nested spans may be orphaned", exc)

    tracer = LangfuseTracer(client)
    atexit.register(client.flush)
    log.info("langfuse tracing enabled host=%s", settings.langfuse_host)
    return tracer
