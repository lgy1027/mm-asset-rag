"""LLM-grounded answer generation."""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

import requests

from .paths import safe_parsed_image_path
from .retrieval import hybrid_search
from .schema import SearchHit
from .settings import get_settings

# Hard cap on total images sent in one chat request, on top of the per-hit
# ``answer_image_max_per_hit`` cap. Bounds token cost: a 5-hit answer with
# 2 images each would otherwise push 10 base64'd figures through an 8B
# local model. 4 is enough to cover the top-2 hits' figures.
_MAX_TOTAL_IMAGES = 4

_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "你是多模态资料检索助手。只能基于给定证据回答；"
        "如果证据不足，要明确说不足。回答要列出关键来源。"
        "证据可能附带图片，可看图回答图中的数字、表格、流程等。"
    ),
}


def format_sources(hits: list[SearchHit]) -> list[dict[str, object]]:
    return [
        {
            "asset_id": hit.asset_id,
            "title": hit.title,
            "source_type": hit.source_type,
            "source_path": hit.source_path,
            "score": round(hit.score, 4),
            "routes": hit.metadata.get("routes", [hit.route]),
            "page": hit.metadata.get("page"),
            "parser": hit.metadata.get("parser") or hit.metadata.get("provider"),
            "images": hit.images or hit.metadata.get("images") or [],
        }
        for hit in hits
    ]


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
        parts.append(f"{label} (/parsed-image/{hit.asset_id}/{img.get('path', '')})")
    return f"关联图片: {'; '.join(parts)}" if parts else ""


def _build_evidence_context(hits: list[SearchHit]) -> str:
    """Assemble the numbered evidence block fed to the LLM.

    Each hit becomes ``[N] asset_id=... title=... source=... page=...`` then
    the evidence text, then — when the hit carries associated figures — a
    ``关联图片:`` line so a text-only LLM can still cite which figure the
    user should look at ("见证据[1]的图3: 双碳目标路线图").
    """
    blocks = []
    for index, hit in enumerate(hits, start=1):
        header = (
            f"[{index}] asset_id={hit.asset_id} title={hit.title} "
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
            url = _read_image_data_url(hit.asset_id, str(img.get("path") or ""))
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
    return requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0.1, "stream": stream, "messages": messages},
        timeout=timeout,
        stream=stream,
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
    evidence = "\n\n".join(hit.evidence[:300] for hit in hits[:3] if hit.evidence)
    return {
        "question": question,
        "answer": (
            "当前未配置 LLM，因此返回检索证据摘要。"
            "请先检查 sources 中的原始资料、页码和解析器，再决定是否接入生成式回答。\n\n" + evidence
        ),
        "sources": format_sources(hits),
    }


def llm_answer(question: str, hits: list[SearchHit]) -> dict[str, object]:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not base_url or not api_key or not model:
        return fallback_answer(question, hits)

    context = _build_evidence_context(hits)
    settings = get_settings()
    content = _user_content(question, context, hits, settings)
    messages = [_SYSTEM_MSG, {"role": "user", "content": content}]
    timeout = float(os.environ.get("LLM_TIMEOUT", "120"))
    try:
        response = _post_chat(base_url, api_key, model, messages, stream=False, timeout=timeout)
        response.raise_for_status()
    except Exception:
        # Image mode can fail when the model isn't vision-capable — degrade
        # to text-only and retry so /answer stays usable.
        if not isinstance(content, list):
            raise
        _degrade_to_text(messages, question, context)
        response = _post_chat(base_url, api_key, model, messages, stream=False, timeout=timeout)
        response.raise_for_status()
    raw_answer = response.json()["choices"][0]["message"]["content"]
    # Strip reasoning-model <think>...</think> blocks.
    answer = re.sub(r"<think>.*?</think>", "", str(raw_answer), flags=re.DOTALL).strip()
    return {
        "question": question,
        "answer": answer,
        "sources": format_sources(hits),
    }


def answer_question(
    question: str,
    top_k: int = 5,
    hits: list[SearchHit] | None = None,
) -> dict[str, object]:
    if hits is None:
        hits = hybrid_search(question, top_k=top_k)
    return llm_answer(question, hits)


def stream_answer_chunks(question: str, hits: list[SearchHit]) -> Iterator[str]:
    """Yield LLM answer chunks one at a time (OpenAI-compatible SSE format).

    Falls back to yielding the deterministic evidence summary as a single chunk
    when LLM credentials are not configured. Reasoning-model ``<think>`` blocks
    are stripped across chunk boundaries so the user only sees the final answer.
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
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
    timeout = float(os.environ.get("LLM_TIMEOUT", "120"))
    try:
        response = _post_chat(base_url, api_key, model, messages, stream=True, timeout=timeout)
        response.raise_for_status()
    except Exception:
        # Image mode can fail for non-vision models — degrade and retry.
        if not isinstance(content, list):
            raise
        _degrade_to_text(messages, question, context)
        response = _post_chat(base_url, api_key, model, messages, stream=True, timeout=timeout)
        response.raise_for_status()

    # Buffer across chunks so we can strip <think>...</think> that spans boundaries.
    buffer = ""
    think_done = False
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
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            except json.JSONDecodeError:
                continue
            if not delta:
                continue
            buffer += delta
            if not think_done:
                end = buffer.find("</think>")
                if end != -1:
                    buffer = buffer[end + len("</think>") :]
                    think_done = True
                elif "<think>" in buffer:
                    # Still inside a <think> block; keep buffering until </think>.
                    continue
                # else: no think tags at all — fall through and yield.
            if buffer:
                yield buffer
                buffer = ""
    if buffer:
        yield buffer


def answer_json(question: str, top_k: int = 5) -> str:
    return json.dumps(answer_question(question, top_k=top_k), ensure_ascii=False, indent=2)
