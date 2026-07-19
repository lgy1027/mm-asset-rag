"""LLM-generated context for Contextual Retrieval.

Implements the Anthropic Contextual Retrieval idea (2024, -49% retrieval
failure rate) in a lightweight form: instead of sending the full document
with every chunk (which on this corpus would be ~328M input tokens), we
send a short document summary + the chunk itself. The LLM returns 50-100
tokens situating the chunk within the document, which is prepended to the
chunk text before embedding + BM25 indexing.

Three retrieval channels benefit:
- **dense** (bge-m3): the context disambiguates generic terms ("diffusion"
  → "DDPM denoising diffusion probabilistic model" vs "Stable Diffusion
  application paper"), so semantically-adjacent-but-imprecise papers stop
  stealing the top rank.
- **BM25-en / BM25-zh**: the context injects exact-match tokens the body
  may dilute (e.g. "DDPM", "去噪扩散概率模型").

Functions are independent of the index path: ``service._do_parse`` calls
them at parse time and stores the result in ``metadata["context"]``;
``build_qdrant_text_index`` then prepends it to the embedding input while
keeping the payload ``text`` clean for evidence / answer generation.

Degradation: when ``OPENAI_*`` is unconfigured or a request fails, both
functions return ``""`` — the caller simply skips the context, preserving
the pre-contextual behavior. Nothing here raises.
"""

from __future__ import annotations

import re

import requests

from .settings import get_settings

# Reasoning models (e.g. MiniMax-M3 in thinking mode) may wrap output in
# <think>...</think>; the context we store should be the final answer only.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _llm_credentials() -> tuple[str | None, str | None, str | None]:
    """Return ``(base_url, api_key, model)`` for the contextual LLM.

    Reuses the shared ``Settings.llm_creds`` (``OPENAI_*`` preferred, with
    a ``VLM_*`` fallback) so a deployment that only configured a
    multimodal VLM still gets Contextual Retrieval — previously this read
    ``OPENAI_*`` directly and silently degraded to a no-op in a VLM-only
    deploy. ``CONTEXTUAL_MODEL`` overrides only the model name, matching
    the other LLM call sites (``answer.py``, ``image_caption.py``,
    ``auto_meta.py``) that all go through the same ``llm_creds`` /
    ``vlm_creds`` properties.
    """
    s = get_settings()
    base_url, api_key, model = s.llm_creds
    if s.contextual_model:
        model = s.contextual_model
    return base_url, api_key, model


def _chat(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """One OpenAI-compatible chat completion call. Returns ``""`` on any failure.

    Shared by :func:`generate_doc_summary` and :func:`generate_chunk_context`.
    Strips ``<think>`` blocks so reasoning-model output stays clean. A single
    failure (network, non-200, missing credentials) degrades to ``""`` — the
    caller treats empty context as "skip", never as an error.
    """
    base_url, api_key, model = _llm_credentials()
    if not base_url or not api_key or not model:
        return ""
    try:
        from .answer import _warn_insecure_base_url

        _warn_insecure_base_url(base_url)
    except Exception:  # pragma: no cover - never block contextual
        pass
    payload = {"model": model, "temperature": temperature, "messages": messages}
    try:
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=float(get_settings().contextual_timeout),
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""
    return _THINK_RE.sub("", str(raw)).strip()


def generate_doc_summary(full_text: str, *, asset_title: str = "") -> str:
    """Summarize a whole document into ~300 tokens.

    The summary is reused as cheap context for every chunk in the document,
    so each chunk call only pays for the chunk + this summary instead of the
    full text. Returns ``""`` on failure — caller skips doc-level context.
    """
    if not full_text.strip():
        return ""
    # Cap the input: median PDF is ~158k chars but a few exceed 1M. MiniMax-M3
    # has 1M context but we don't want to pay for the tail; the summary only
    # needs the document's shape, not every word.
    cap = 200_000
    truncated = full_text[:cap] if len(full_text) > cap else full_text
    title_hint = f"文档标题：{asset_title}\n\n" if asset_title else ""
    messages = [
        {
            "role": "system",
            "content": (
                "你用 300 token 以内概括文档的核心主题、方法、贡献,便于后续给文档片段"
                "定位上下文。只输出摘要本身,不要额外解释。"
            ),
        },
        {"role": "user", "content": f"{title_hint}文档全文:\n{truncated}"},
    ]
    summary = _chat(messages, temperature=0.0)
    return summary[:cap] if summary else ""


def generate_chunk_context(
    chunk_text: str,
    doc_summary: str,
    *,
    section: str = "",
    asset_title: str = "",
) -> str:
    """Generate a 50-100 token context situating ``chunk_text`` in its document.

    Anthropic's prompt, adapted: given the document summary and this chunk,
    "give a short succinct context to situate this chunk within the overall
    document". The context is prepended to the chunk before embedding + BM25,
    so dense + sparse channels both see it. Returns ``""`` on failure.
    """
    if not chunk_text.strip():
        return ""
    s = get_settings()
    # Cap chunk input — long chunks (max 54k chars in this corpus) would
    # dominate the prompt; the context only needs enough to identify the
    # chunk's topic.
    chunk_cap = s.contextual_chunk_max_chars
    truncated = chunk_text[:chunk_cap] if len(chunk_text) > chunk_cap else chunk_text
    parts: list[str] = []
    if asset_title:
        parts.append(f"文档: {asset_title}")
    if section:
        parts.append(f"章节: {section}")
    if doc_summary:
        parts.append(f"文档摘要: {doc_summary}")
    parts.append(f"片段内容:\n{truncated}")
    user_content = "\n\n".join(parts)
    messages = [
        {
            "role": "system",
            "content": (
                "你用 50-100 token 简要说明下面这个文档片段在整篇文档中的位置和主题,"
                "便于检索时区分语义相近的内容。只输出上下文说明本身,不要复述片段,"
                "不要额外解释。"
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _chat(messages, temperature=0.0)


def enrich_docs_with_context(
    docs: list,  # list[ParsedDocument] — typed loosely to avoid a circular import
    *,
    asset_title: str = "",
    cache_path=None,
) -> None:
    """Attach ``metadata["context"]`` to each doc in place.

    Pipeline (per asset, called from ``service._do_parse``):
    1. Build the full document text by concatenating chunk texts.
    2. Generate one document summary (1 LLM call per asset).
    3. Generate one chunk context per chunk (concurrent).
    4. Persist to ``cache_path`` (a ``parsed/<id>/context.jsonl``) so a later
       ``mmrag reindex`` reuses the context without re-calling the LLM.

    On any LLM failure the corresponding ``metadata["context"]`` is left unset
    (or set to ``""``), and the index path treats empty context as "no prefix"
    — identical to the pre-contextual behavior. Image single-chunk assets are
    skipped (chunk_cap + empty doc summary yield no benefit on a caption).

    The cache is keyed by ``chunk_index`` (falling back to position) so a
    partial re-parse picks up where it left off.
    """
    if not docs:
        return
    # Reuse cache when present (reindex / non-force re-parse).
    import json
    from concurrent.futures import ThreadPoolExecutor

    cache: dict[str, str] = {}
    if cache_path is not None:
        from pathlib import Path

        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                for line in cache_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        obj = json.loads(line)
                        cache[obj["key"]] = obj.get("context", "")
            except Exception:
                cache = {}

    # 1. Document summary (skip if cached summary exists under a reserved key).
    full_text = "\n\n".join(d.text for d in docs)
    summary = cache.get("__summary__")
    if not summary:
        summary = generate_doc_summary(full_text, asset_title=asset_title)
        if summary:
            cache["__summary__"] = summary

    # 2. Per-chunk context (concurrent). Key by chunk_index when present,
    #    else by position — matches how the cache is written below.
    def _key_for(idx: int, d) -> str:
        ci = d.metadata.get("chunk_index")
        return f"chunk:{ci}" if ci is not None else f"pos:{idx}"

    to_generate: list[tuple[int, str]] = []
    for idx, d in enumerate(docs):
        key = _key_for(idx, d)
        cached = cache.get(key, "")
        if cached:
            d.metadata["context"] = cached
        else:
            to_generate.append((idx, key))

    if to_generate:
        max_workers = max(1, get_settings().contextual_concurrency)

        def _one(args: tuple[int, str]) -> tuple[int, str, str]:
            idx, key = args
            d = docs[idx]
            ctx = generate_chunk_context(
                d.text,
                summary,
                section=str(d.metadata.get("section") or ""),
                asset_title=asset_title,
            )
            return idx, key, ctx

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, key, ctx in ex.map(_one, to_generate):
                if ctx:
                    docs[idx].metadata["context"] = ctx
                    cache[key] = ctx

    # 3. Persist cache for reuse — but only if we actually produced any
    #    context (summary or per-chunk). When the LLM is unconfigured or every
    #    call failed, there is nothing worth caching; skipping the write keeps
    #    the parse dir clean (matches pre-contextual behavior, e.g. for image
    #    single-chunk assets / no-credentials runs). Nothing here raises.
    has_context = bool(summary) or any(d.metadata.get("context") for d in docs)
    if cache_path is not None and has_context:
        from pathlib import Path

        cache_path = Path(cache_path)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps({"key": "__summary__", "context": summary}, ensure_ascii=False)
                    + "\n"
                )
                for idx, d in enumerate(docs):
                    key = _key_for(idx, d)
                    ctx = d.metadata.get("context", "")
                    f.write(json.dumps({"key": key, "context": ctx}, ensure_ascii=False) + "\n")
        except Exception as exc:  # disk full / permission — degrade, but log
            import logging

            logging.getLogger(__name__).warning(
                "contextual cache write failed for %s; a later reindex will "
                "re-invoke the LLM per chunk instead of reusing the cache: %s",
                cache_path,
                exc,
            )
