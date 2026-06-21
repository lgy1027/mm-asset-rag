"""LLM-grounded answer generation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator

import requests

from .retrieval import hybrid_search
from .schema import SearchHit


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
        }
        for hit in hits
    ]


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

    context = "\n\n".join(
        f"[{index}] asset_id={hit.asset_id} title={hit.title} source={hit.source_path} "
        f"page={hit.metadata.get('page')}\n{hit.evidence[:1200]}"
        for index, hit in enumerate(hits, start=1)
    )
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是多模态资料检索助手。只能基于给定证据回答；"
                    "如果证据不足，要明确说不足。回答要列出关键来源。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n证据：\n{context}",
            },
        ],
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
    )
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

    context = "\n\n".join(
        f"[{index}] asset_id={hit.asset_id} title={hit.title} source={hit.source_path} "
        f"page={hit.metadata.get('page')}\n{hit.evidence[:1200]}"
        for index, hit in enumerate(hits, start=1)
    )
    payload = {
        "model": model,
        "temperature": 0.1,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是多模态资料检索助手。只能基于给定证据回答；"
                    "如果证据不足，要明确说不足。回答要列出关键来源。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n证据：\n{context}",
            },
        ],
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        stream=True,
    )
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
            data = line[len("data: "):]
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
                    buffer = buffer[end + len("</think>"):]
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
