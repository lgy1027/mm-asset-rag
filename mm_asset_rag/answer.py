"""LLM-grounded answer generation."""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests

from .evidence_policy import assess_answer_evidence
from .llm_transport import LlmTransportError, post_chat_completion
from .observability import runtime_metrics
from .paths import safe_parsed_image_path
from .schema import SearchHit
from .search_service import SearchCommand, SearchMode, SearchService, get_search_service
from .settings import get_settings

# Hard cap on total images sent in one chat request, on top of the per-hit
# ``answer_image_max_per_hit`` cap. Bounds token cost: a 5-hit answer with
# 2 images each would otherwise push 10 base64'd figures through an 8B
# local model. 4 is enough to cover the top-2 hits' figures.
_MAX_TOTAL_IMAGES = 4
log = logging.getLogger(__name__)

_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "你是多模态资料检索助手。只能基于给定证据回答；"
        "如果证据不足，要明确说不足。每个事实句或段落后必须紧跟证据编号[N]，"
        "N 只能是给定证据块的编号；不得把引用只放在末尾来源列表。"
        "证据可能附带图片，可看图回答图中的数字、表格、流程等。"
    ),
}

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    reason: str | None = None


def validate_answer_citations(answer: str, evidence_count: int) -> CitationValidation:
    """Check that a substantive answer uses in-range citations inline."""
    markers = list(_CITATION_RE.finditer(answer))
    if not markers:
        return CitationValidation(False, "no_citation")
    if any(int(marker.group(1)) < 1 or int(marker.group(1)) > evidence_count for marker in markers):
        return CitationValidation(False, "out_of_range")
    for marker in markers:
        prefix = answer[max(0, marker.start() - 80) : marker.start()]
        if re.search(r"(?:来源|参考来源|sources?)\s*[:：]?\s*$", prefix, flags=re.IGNORECASE):
            return CitationValidation(False, "detached_citation")
    return CitationValidation(True)


def format_sources(hits: list[SearchHit]) -> list[dict[str, object]]:
    return [
        {
            "document_id": hit.metadata.get("document_id"),
            "chunk_id": hit.metadata.get("chunk_id"),
            "title": hit.title,
            "source_type": hit.source_type,
            "source_path": hit.source_path,
            "score": round(hit.score, 4),
            "routes": hit.metadata.get("routes", [hit.route]),
            "page": hit.metadata.get("page"),
            "parser": hit.metadata.get("parser") or hit.metadata.get("provider"),
            "images": _without_asset_id(hit.images or hit.metadata.get("images") or []),
        }
        for hit in hits
    ]


def _without_asset_id(value: object) -> object:
    """Remove compatibility-only physical IDs from public answer payloads."""
    if isinstance(value, dict):
        return {key: _without_asset_id(item) for key, item in value.items() if key != "asset_id"}
    if isinstance(value, list):
        return [_without_asset_id(item) for item in value]
    return value


def _image_hint(hit: SearchHit) -> str:
    """One-line summary of a hit's associated images for the LLM context.

    The LLM cannot see image pixels (tier 1) but can cite figure captions
    and tell the user which figure to look at — e.g. "见证据[1]关联的图3:
    双碳目标路线图". Returns "" when the hit has no images.
    """
    images = hit.images or hit.metadata.get("images") or []
    if not images:
        return ""
    parts = []
    for img in images:
        if not isinstance(img, dict):
            continue
        cap = str(img.get("caption") or "").strip()
        fig = img.get("figure_id")
        tag = f"图{fig}" if fig else "图"
        label = f"{tag}: {cap}" if cap else tag
        parts.append(
            f"{label} (/parsed-image/{hit.metadata.get('document_id', '')}/{img.get('path', '')})"
        )
    return f"关联图片: {'; '.join(parts)}" if parts else ""


def _build_evidence_context(hits: list[SearchHit]) -> str:
    """Assemble the numbered evidence block fed to the LLM.

    Each hit becomes ``[N] document_id=... chunk_id=...`` then
    the evidence text, then — when the hit carries associated figures — a
    ``关联图片:`` line so a text-only LLM can still cite which figure the
    user should look at ("见证据[1]的图3: 双碳目标路线图").
    """
    blocks = []
    for index, hit in enumerate(hits, start=1):
        header = (
            f"[{index}] document_id={hit.metadata.get('document_id')} "
            f"chunk_id={hit.metadata.get('chunk_id')} title={hit.title} "
            f"source={hit.source_path} page={hit.metadata.get('page')}"
        )
        body = hit.evidence[:1200]
        hint = _image_hint(hit)
        blocks.append(f"{header}\n{body}\n{hint}" if hint else f"{header}\n{body}")
    return "\n\n".join(blocks)


def _read_image_data_url(asset_id: str, image_path: str) -> str | None:
    """Read ``parsed/<asset_id>/images/<basename>`` and return a base64 data URL.

    ``image_path`` is the payload form ("images/p1_i0.jpeg"); only the base
    name is trusted. Resolution + traversal guard is delegated to
    :func:`paths.safe_parsed_image_path` (same validation the
    ``/parsed-image`` endpoint uses). Returns ``None`` when the file is
    missing/invalid — the caller skips it rather than failing the answer.
    """
    basename = Path(image_path).name
    candidate = safe_parsed_image_path(asset_id, basename)
    if candidate is None:
        return None
    suffix = candidate.suffix.lower().lstrip(".")
    # jpeg → jpg mime canonical, others use the suffix as-is.
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    try:
        data = candidate.read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _collect_image_parts(hits: list[SearchHit], settings) -> list[dict]:
    """Build ``image_url`` content parts for the hit's associated figures.

    Caps at ``settings.answer_image_max_per_hit`` per hit and
    :data:`_MAX_TOTAL_IMAGES` overall. Skips unreadable / missing images
    silently — a single corrupt figure must not abort the answer.
    """
    parts: list[dict] = []
    per_hit = max(0, settings.answer_image_max_per_hit)
    for hit in hits:
        if len(parts) >= _MAX_TOTAL_IMAGES:
            break
        images = hit.images or hit.metadata.get("images") or []
        for img in images[:per_hit]:
            if not isinstance(img, dict):
                continue
            url = _read_image_data_url(hit.cache_id, str(img.get("path") or ""))
            if url is None:
                continue
            parts.append({"type": "image_url", "image_url": {"url": url}})
            if len(parts) >= _MAX_TOTAL_IMAGES:
                break
    return parts


def _user_content(question: str, context: str, hits: list[SearchHit], settings) -> str | list:
    """Build the user message ``content`` — string, or content-parts list.

    When ``settings.answer_with_images`` is on and the hits carry images,
    returns OpenAI-compatible multimodal content: a text part holding the
    question + evidence, followed by ``image_url`` parts. Otherwise returns
    the plain text string (identical to pre-tier-3 behaviour).
    """
    text = f"问题：{question}\n\n证据：\n{context}"
    if not settings.answer_with_images:
        return text
    parts = _collect_image_parts(hits, settings)
    if not parts:
        return text
    return [{"type": "text", "text": text}, *parts]


def _post_chat(
    base_url: str, api_key: str, model: str, messages: list, *, stream: bool, timeout: float
) -> requests.Response:
    """One OpenAI-compatible chat completion POST (shared by answer + stream)."""
    return post_chat_completion(
        base_url, api_key, model, messages, timeout=timeout, stream=stream, temperature=0.1
    )


def _degrade_to_text(messages: list, question: str, context: str) -> None:
    """Replace the user message content with plain text (in place).

    Used when an image-bearing request fails — e.g. the configured model
    is not vision-capable and the server rejects the ``image_url`` parts.
    Falling back to text-only keeps ``/answer`` working when the user
    toggles ``ANSWER_WITH_IMAGES`` on without a multimodal model.
    """
    for m in messages:
        if m.get("role") == "user":
            m["content"] = f"问题：{question}\n\n证据：\n{context}"
            return


def fallback_answer(question: str, hits: list[SearchHit]) -> dict[str, object]:
    evidence = "\n\n".join(
        f"证据摘要 [{index}]：{hit.evidence[:300]}"
        for index, hit in enumerate(hits[:3], start=1)
        if hit.evidence
    )
    return {
        "question": question,
        "answer": (
            "当前未配置 LLM 或无法使用 LLM，因此返回检索证据摘要。"
            "请先检查 sources 中的原始资料、页码和解析器，再决定是否接入生成式回答。\n\n" + evidence
        ),
        "sources": format_sources(hits),
        # Marker consumed by ``answer_evaluation.run_answer_eval`` to split
        # coverage / citation stats between real LLM answers and fallback
        # evidence summaries. A user running ``mmrag eval --answer-quality``
        # on a machine without a configured LLM connection needs the
        # report to distinguish "the LLM is broken" from "we never called
        # one" — coverage / citation are necessarily near-zero on the
        # fallback path, but that doesn't mean the eval regressed.
        "_fallback": True,
    }


def _repair_citations(
    question: str,
    context: str,
    draft: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> str | None:
    """Ask once for citation-only repair; never manufacture markers locally."""
    messages = [
        {
            "role": "system",
            "content": "修复草稿的证据引用。只保留有证据支持的内容，每个事实后附有效[N]。",
        },
        {
            "role": "user",
            "content": f"问题：{question}\n\n证据：\n{context}\n\n草稿：\n{draft}",
        },
    ]
    try:
        response = _post_chat(base_url, api_key, model, messages, stream=False, timeout=timeout)
        response.raise_for_status()
        return re.sub(
            r"<think>.*?</think>",
            "",
            str(response.json()["choices"][0]["message"]["content"]),
            flags=re.DOTALL,
        ).strip()
    except Exception:
        return None


def llm_answer(question: str, hits: list[SearchHit]) -> dict[str, object]:
    settings = get_settings()
    base_url, api_key, model = settings.llm_creds
    if not base_url or not api_key or not model:
        return fallback_answer(question, hits)

    context = _build_evidence_context(hits)
    content = _user_content(question, context, hits, settings)
    messages = [_SYSTEM_MSG, {"role": "user", "content": content}]
    timeout = float(settings.llm_timeout)
    try:
        response = _post_chat(base_url, api_key, model, messages, stream=False, timeout=timeout)
        response.raise_for_status()
    except LlmTransportError:
        return fallback_answer(question, hits)
    except Exception:
        # Image mode can fail when the model isn't vision-capable — degrade
        # to text-only and retry so /answer stays usable.
        if not isinstance(content, list):
            raise
        _degrade_to_text(messages, question, context)
        try:
            response = _post_chat(base_url, api_key, model, messages, stream=False, timeout=timeout)
            response.raise_for_status()
        except LlmTransportError:
            return fallback_answer(question, hits)
    raw_answer = response.json()["choices"][0]["message"]["content"]
    # Strip reasoning-model <think>...</think> blocks.
    answer = re.sub(r"<think>.*?</think>", "", str(raw_answer), flags=re.DOTALL).strip()
    if not validate_answer_citations(answer, len(hits)).valid:
        repaired = _repair_citations(question, context, answer, base_url, api_key, model, timeout)
        if repaired is None or not validate_answer_citations(repaired, len(hits)).valid:
            return fallback_answer(question, hits)
        answer = repaired
    return {
        "question": question,
        "answer": answer,
        "sources": format_sources(hits),
    }


def answer_question(
    question: str,
    top_k: int = 5,
    hits: list[SearchHit] | None = None,
    *,
    search_service: SearchService | None = None,
    collection: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    principal: str | None = None,
    min_confidence: float = 0.5,
) -> dict[str, object]:
    if hits is None:
        if not collection or not principal:
            raise ValueError("collection and principal are required for answer retrieval")
        service = search_service or get_search_service()
        hits = service.execute(
            SearchCommand(
                query=question,
                mode=SearchMode.HYBRID,
                top_k=top_k,
                collection=collection,
                metadata_filter=metadata_filter,
                principal=principal,
            )
        )
    assessment = assess_answer_evidence(question, hits or [], get_settings())
    if not assessment.sufficient:
        log.info("answer_refusal reason=%s candidates=%d", assessment.reason, len(hits or []))
        runtime_metrics.record_refusal(reason=assessment.reason, candidates=len(hits or []))
        return {
            "question": question,
            "answer": "证据不足，无法基于当前知识库可靠回答。",
            "sources": [],
            "refusal_reason": assessment.reason,
        }
    return llm_answer(question, hits)


def stream_answer_chunks(
    question: str, hits: list[SearchHit], *, min_confidence: float = 0.5
) -> Iterator[str]:
    """Yield LLM answer chunks one at a time (OpenAI-compatible SSE format).

    Falls back to yielding the deterministic evidence summary as a single chunk
    when LLM credentials are not configured. Reasoning-model ``<think>`` blocks
    are stripped across chunk boundaries so the user only sees the final answer.
    """
    assessment = assess_answer_evidence(question, hits, get_settings())
    if not assessment.sufficient:
        log.info("answer_refusal reason=%s candidates=%d", assessment.reason, len(hits))
        runtime_metrics.record_refusal(reason=assessment.reason, candidates=len(hits))
        yield "证据不足，无法基于当前知识库可靠回答。"
        return

    base_url, api_key, model = get_settings().llm_creds
    if not base_url or not api_key or not model:
        fb = fallback_answer(question, hits)
        cleaned = re.sub(r"<think>.*?</think>", "", str(fb["answer"]), flags=re.DOTALL).strip()
        if cleaned:
            yield cleaned
        return

    context = _build_evidence_context(hits)
    settings = get_settings()
    content = _user_content(question, context, hits, settings)
    messages = [_SYSTEM_MSG, {"role": "user", "content": content}]
    timeout = float(settings.llm_timeout)
    try:
        response = _post_chat(base_url, api_key, model, messages, stream=True, timeout=timeout)
        response.raise_for_status()
    except LlmTransportError:
        yield str(fallback_answer(question, hits)["answer"])
        return
    except Exception:
        # Image mode can fail for non-vision models — degrade and retry.
        if not isinstance(content, list):
            yield str(fallback_answer(question, hits)["answer"])
            return
        _degrade_to_text(messages, question, context)
        try:
            response = _post_chat(base_url, api_key, model, messages, stream=True, timeout=timeout)
            response.raise_for_status()
        except Exception:
            yield str(fallback_answer(question, hits)["answer"])
            return

    # Buffer across chunks so we can strip <think>...</think> that spans boundaries.
    buffer = ""
    in_think = False
    # ollama and other OpenAI-compat servers may not send an explicit charset;
    # requests' decode_unicode=True then falls back to latin-1 and shreds CJK
    # bytes. Force utf-8 by reading raw bytes and decoding ourselves.
    response.encoding = "utf-8"
    pending = b""
    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            continue
        pending += chunk
        while b"\n" in pending:
            raw_line, pending = pending.split(b"\n", 1)
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                choices = obj.get("choices") if isinstance(obj, dict) else None
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    continue
                delta = choices[0].get("delta", {}).get("content")
            except (json.JSONDecodeError, TypeError):
                continue
            if not delta:
                continue
            buffer += delta
            # Strip <think>...</think> blocks. Reasoning models may emit
            # more than one block, and a single block can span chunks —
            # ``in_think`` remembers we are inside an unterminated block
            # whose tail may arrive in a later delta.
            while True:
                if in_think:
                    end = buffer.find("</think>")
                    if end == -1:
                        buffer = ""
                        break
                    buffer = buffer[end + len("</think>") :]
                    in_think = False
                    continue
                start = buffer.find("<think>")
                if start == -1:
                    break
                end = buffer.find("</think>", start)
                if end != -1:
                    buffer = buffer[:start] + buffer[end + len("</think>") :]
                    continue
                # Opening tag without a close: keep the text before it,
                # hold the rest until the block closes.
                buffer = buffer[:start]
                in_think = True
                break
            if buffer:
                yield buffer
                buffer = ""
    if buffer:
        yield buffer


def answer_json(
    question: str,
    top_k: int = 5,
    *,
    search_service: SearchService | None = None,
    collection: str | None = None,
    metadata_filter: dict[str, object] | None = None,
    principal: str | None = None,
    min_confidence: float = 0.5,
) -> str:
    return json.dumps(
        answer_question(
            question,
            top_k=top_k,
            search_service=search_service,
            collection=collection,
            metadata_filter=metadata_filter,
            principal=principal,
            min_confidence=min_confidence,
        ),
        ensure_ascii=False,
        indent=2,
    )
