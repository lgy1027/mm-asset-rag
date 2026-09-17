"""Tracer protocol, no-op default, Langfuse factory, and instrumentation seams."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from mm_asset_rag.core import (
    langfuse_tracer as mm_asset_rag_langfuse_tracer,
)
from mm_asset_rag.core import observability
from mm_asset_rag.core.observability import NoOpTracer, get_tracer, set_tracer
from mm_asset_rag.core.settings import Settings


class RecordingSpan:
    def __init__(self, log: list, name: str) -> None:
        self.log = log
        self.name = name
        self.attributes: dict[str, object] = {}
        self.terminal: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def update(self, *, output=None, usage=None, metadata=None, level=None, status_message=None):
        self.terminal = {
            "output": output,
            "usage": usage,
            "metadata": metadata,
            "level": level,
            "status_message": status_message,
        }

    def score(self, *, name, value, comment=None, metadata=None):
        self.terminal.setdefault("scores", []).append(
            {"name": name, "value": value, "comment": comment, "metadata": metadata}
        )

    def set_tags(self, tags):
        self.attributes["tags"] = list(tags)


class RecordingTracer:
    """Captures every span/generation opened through it."""

    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []
        self.generations: list[RecordingSpan] = []
        self.flushed = False

    @contextmanager
    def start_span(self, name, *, attributes=None, input=None):
        span = RecordingSpan(self.spans, name)
        span.attributes.update(attributes or {})
        if input is not None:
            span.attributes["input"] = input
        self.spans.append(span)
        yield span

    @contextmanager
    def start_generation(
        self, name, *, model=None, input=None, model_parameters=None, metadata=None
    ):
        span = RecordingSpan(self.generations, name)
        if model:
            span.attributes["model"] = model
        if model_parameters:
            span.attributes["model_parameters"] = model_parameters
        if metadata:
            span.attributes.update(metadata)
        self.generations.append(span)
        yield span

    def flush(self) -> None:
        self.flushed = True


# ── default behaviour ────────────────────────────────────────────────────────


def test_default_tracer_is_noop():
    assert isinstance(get_tracer(), NoOpTracer)


def test_noop_span_tolerates_any_call():
    tracer = NoOpTracer()
    with tracer.start_span("s", attributes={"a": 1}) as span:
        span.set_attribute("b", 2)
        span.update(output="x", usage={"input": 1}, level="ERROR")
    with tracer.start_generation("g", model="m", input=[{"role": "user"}]) as gen:
        gen.update(output="y")
    tracer.flush()  # must not raise


def test_tracer_protocol_runtime_checkable():
    assert isinstance(NoOpTracer(), observability.Tracer)
    assert isinstance(RecordingTracer(), observability.Tracer)


def test_build_tracer_unknown_provider_falls_back_to_noop():
    settings = Settings(tracing_provider="some-future-provider")
    assert isinstance(observability._build_tracer(settings), NoOpTracer)


def test_set_tracer_installs_and_reset_rebuilds(monkeypatch):
    fake = RecordingTracer()
    set_tracer(fake)
    assert get_tracer() is fake
    set_tracer(None)
    monkeypatch.setenv("TRACING_PROVIDER", "none")
    assert isinstance(get_tracer(), NoOpTracer)


# ── langfuse factory ─────────────────────────────────────────────────────────


def test_build_langfuse_tracer_without_credentials_returns_none():
    """Missing keys disable tracing whether or not the SDK is installed."""
    settings = Settings(tracing_provider="langfuse")
    assert mm_asset_rag_langfuse_tracer.build_langfuse_tracer(settings) is None


def test_build_langfuse_tracer_without_sdk_returns_none(monkeypatch):
    """No SDK in the environment → factory degrades to the no-op tracer."""
    settings = Settings(
        tracing_provider="langfuse",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    monkeypatch.setitem(
        sys.modules, "langfuse", None
    )  # forces ImportError on `from langfuse import Langfuse`
    # The SDK import is lazy inside the factory, so blocking ``langfuse``
    # in sys.modules is enough — no module reload needed.
    assert mm_asset_rag_langfuse_tracer.build_langfuse_tracer(settings) is None


def test_build_langfuse_tracer_with_fake_sdk(monkeypatch):
    """A minimal fake SDK module exercises the real construction path."""

    class FakeLangfuse:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def flush(self) -> None:
            pass

    fake_module = ModuleType("langfuse")
    fake_module.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    settings = Settings(
        tracing_provider="langfuse",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="http://localhost:3000",
    )
    tracer = mm_asset_rag_langfuse_tracer.build_langfuse_tracer(settings)
    assert tracer is not None
    assert tracer._client.kwargs["public_key"] == "pk-test"
    assert tracer._client.kwargs["host"] == "http://localhost:3000"


def test_langfuse_tracer_records_elapsed_and_error():
    """The timed context stamps elapsed_ms and ERROR level on exceptions."""
    import mm_asset_rag.core.langfuse_tracer as lt

    events: list[dict] = []
    raw_spans: list[FakeRawSpan] = []

    class FakeCM:
        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            return self.raw

        def __exit__(self, *args):
            events.append(("exit", args[0]))
            return False

    class FakeClient:
        def start_as_current_span(self, *, name, **kwargs):
            events.append(("span", name))
            raw = FakeRawSpan()
            raw_spans.append(raw)
            return FakeCM(raw)

    tracer = lt.LangfuseTracer(FakeClient())
    with tracer.start_span("ok", attributes={"a": 1}) as span:
        span.set_attribute("inner", True)
    assert ("span", "ok") in events
    # elapsed_ms + the manual attribute land as metadata updates on the raw span
    keys = [k for u in raw_spans[0].updates for k in u.get("metadata", {})]
    assert "elapsed_ms" in keys
    assert "inner" in keys

    with pytest.raises(ValueError, match="boom"), tracer.start_span("fails"):
        raise ValueError("boom")
    error_exits = [e for e in events if e[0] == "exit" and e[1] is ValueError]
    assert error_exits
    error_update = raw_spans[1].updates[-1]
    assert error_update["level"] == "ERROR"
    assert "boom" in error_update["status_message"]


class FakeRawSpan:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


def test_timed_context_swallows_raw_failures():
    """A raising raw span must never leak into the instrumented pipeline."""
    import mm_asset_rag.core.langfuse_tracer as lt

    class RaisingRaw:
        def update(self, **kwargs):
            raise ValueError("otel exploded")

    class RaisingCM:
        def __enter__(self):
            return RaisingRaw()

        def __exit__(self, *args):
            raise RuntimeError("close exploded")

    class RaisingClient:
        def start_as_current_span(self, *, name, **kwargs):
            return RaisingCM()

    tracer = lt.LangfuseTracer(RaisingClient())

    # Clean path: no exception escapes the with-block.
    with tracer.start_span("clean"):
        pass

    # Exception path: the block's own ValueError propagates unchanged, and
    # neither the failing attribute update nor the failing cm.__exit__
    # replaces it.
    with pytest.raises(ValueError, match="block-error"), tracer.start_span("fails"):
        raise ValueError("block-error")

    # enter() itself raising degrades to a no-op span.
    class EnterRaisesCM:
        def __enter__(self):
            raise RuntimeError("enter exploded")

        def __exit__(self, *args):
            return False

    class EnterRaisesClient:
        def start_as_current_span(self, *, name, **kwargs):
            return EnterRaisesCM()

    tracer2 = lt.LangfuseTracer(EnterRaisesClient())
    with tracer2.start_span("enter-fails") as span:
        span.set_attribute("k", "v")  # must not raise


# ── instrumentation seams ────────────────────────────────────────────────────


def test_llm_transport_emits_generation(monkeypatch):
    """``post_chat_completion`` reports one generation through the tracer."""
    from mm_asset_rag.core import llm_transport

    tracer = RecordingTracer()
    set_tracer(tracer)

    class FakeResp:
        status_code = 200

    monkeypatch.setattr(
        llm_transport.OpenAIChatAdapter, "complete", lambda self, messages, **kw: FakeResp()
    )
    out = llm_transport.post_chat_completion(
        "https://example.com/v1",
        "sk-test",
        "test-model",
        [{"role": "user", "content": "hi"}],
        timeout=5.0,
    )
    assert out.status_code == 200
    assert len(tracer.generations) == 1
    gen = tracer.generations[0]
    assert gen.name == "llm.chat_completion"
    assert gen.attributes["model"] == "test-model"
    assert gen.attributes["stream"] is False
    assert gen.attributes["status_code"] == 200
    assert gen.attributes["model_parameters"] == {
        "stream": False,
        "temperature": 0.1,
        "max_tokens": None,
    }


def test_llm_transport_extracts_usage(monkeypatch):
    """Non-streaming completions report token usage; stream/absent → None."""
    from mm_asset_rag.core.llm_transport import _extract_usage

    class Resp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    assert _extract_usage(
        Resp({"usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})
    ) == {
        "input": 11,
        "output": 7,
        "total": 18,
    }
    # OpenAI-style new key names
    assert _extract_usage(Resp({"usage": {"input_tokens": 3, "output_tokens": 4}})) == {
        "input": 3,
        "output": 4,
    }
    assert _extract_usage(Resp({"choices": []})) is None
    assert _extract_usage(Resp("not-a-dict")) is None

    class BrokenJson:
        def json(self):
            raise ValueError("no json")

    assert _extract_usage(BrokenJson()) is None


def test_dispatch_search_emits_span(monkeypatch):
    tracer = RecordingTracer()
    set_tracer(tracer)
    from mm_asset_rag.core.schema import SearchHit
    from mm_asset_rag.query.search_service import SearchCommand, SearchMode, SearchService

    hit = SearchHit(
        route="text-to-image",
        score=0.9,
        asset_id="a1",
        title="t",
        source_type="image",
        source_path="p",
        evidence="e",
        metadata={"document_id": "d1", "allowed_principals": ["alice"], "collection": "docs"},
    )

    class StubBackend:
        name = "stub"

        def search_text_to_image(self, *, query, top_k, search_filter=None):
            return [hit]

    service = SearchService(backend=StubBackend())
    hits = service.execute(
        SearchCommand(
            query="q", mode=SearchMode.TEXT_TO_IMAGE, top_k=3, collection="docs", principal="alice"
        )
    )
    assert [h.asset_id for h in hits] == ["d1"]
    assert [s.name for s in tracer.spans] == ["search.dispatch"]
    span = tracer.spans[0]
    assert span.attributes["input"] == "q"
    assert span.attributes["top_k"] == 3
    assert span.attributes["tags"] == ["route:text-to-image", "collection:docs"]
    assert span.terminal["output"]["returned"] == 1
    assert span.terminal["output"]["route"] == "text-to-image"


def test_eval_case_emits_span_with_retrieval_scores(tmp_path):
    from mm_asset_rag.core.schema import SearchHit
    from mm_asset_rag.eval.evaluation import run_eval

    tracer = RecordingTracer()
    set_tracer(tracer)
    cases = tmp_path / "cases.json"
    cases.write_text(
        '{"version": "v1", "groups": {'
        '"en": [{"query_id": "q1", "query": "handbook"}], '
        '"negative": [{"query_id": "n1", "query": "offtopic"}]}, '
        '"qrels": {"q1": {"doc-1": 1}, "n1": {}}}',
        encoding="utf-8",
    )

    def search(command):
        hit = SearchHit(
            route="text",
            score=0.9,
            asset_id="a1",
            title="t",
            source_type="pdf",
            source_path="p",
            evidence="e",
            metadata={"document_id": "doc-1"},
        )
        return [hit] if "handbook" in command.query else []

    results = run_eval(
        search_fn=search,
        collection="team",
        principal="alice",
        cases_path=cases,
    )
    assert [result.hit for result in results] == [True, False]
    assert [s.name for s in tracer.spans] == ["eval.case", "eval.case"]
    positive, negative = tracer.spans
    assert positive.attributes["query_id"] == "q1"
    assert positive.terminal["scores"] == [
        {"name": "retrieval_hit", "value": 1.0, "comment": "rank=1", "metadata": None}
    ]
    assert [s["name"] for s in negative.terminal["scores"]] == [
        "retrieval_hit",
        "negative_reject",
    ]
    assert negative.terminal["scores"][1]["value"] == 1.0


def test_answer_question_emits_span(monkeypatch):
    from mm_asset_rag.answer import answer as answer_module

    tracer = RecordingTracer()
    set_tracer(tracer)
    hit = SimpleNamespace(
        asset_id="a1",
        score=0.9,
        route="text",
        title="t",
        source_type="text",
        source_path="p",
        evidence="e",
        metadata={},
        images=[],
        cache_id=None,
    )
    monkeypatch.setattr(
        answer_module,
        "assess_answer_evidence",
        lambda question, hits, settings: SimpleNamespace(sufficient=True, reason="ok"),
    )
    monkeypatch.setattr(
        answer_module,
        "llm_answer",
        lambda question, hits: {"question": question, "answer": "答案", "sources": [1]},
    )
    out = answer_module.answer_question("问题", hits=[hit])
    assert out["answer"] == "答案"
    span_names = [s.name for s in tracer.spans]
    assert "answer.generate" in span_names
    span = tracer.spans[span_names.index("answer.generate")]
    assert span.terminal["output"]["answer_chars"] == 2
    assert span.terminal["output"]["sources"] == 1
    assert span.terminal["output"]["refusal"] is False


def test_rerank_emits_span():
    from mm_asset_rag.embedders.reranker import Reranker

    tracer = RecordingTracer()
    set_tracer(tracer)

    class Passthrough(Reranker):
        def _load(self):
            raise AssertionError("not used — no text hits")

    hit = SimpleNamespace(
        route="image-to-image",
        source_type="image",
        metadata={"raw_score": 0.5},
        score=0.1,
        asset_id="a1",
        title="t",
        source_path="p",
        evidence="e",
        images=[],
        cache_id=None,
    )
    out = Passthrough().rerank("q", [hit], top_k=1)
    assert len(out) == 1
    assert [s.name for s in tracer.spans] == ["rerank.score"]
