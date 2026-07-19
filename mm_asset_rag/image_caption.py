"""VLM captions for document-embedded figures.

A document's embedded figures (docx/pptx pictures via markitdown / docling,
PDF figures via PyMuPDF) are saved to ``parsed/<id>/images/`` and associated
with chunks via ``metadata["images"]``, but their *content* is otherwise
invisible to the text index: a slide whose only payload is a diagram, or a
figure referenced by "如图3所示", carries no searchable semantics. A user
querying "双碳路线图" cannot hit a slide that is *only* that diagram.

This module is the text-route fix: generate a short Chinese caption for each
embedded figure with an existing caption (figure_id / "图N: …" detected by
PyMuPDF is preserved — we only fill the empty ones), append it to the
chunk's text so the figure's semantics enter the dense + BM25 channels, and
record it in ``metadata["images"][*]["caption"]`` so the answer layer's
``_image_hint`` can cite it.

Constraints respected (project design):
- Figures stay on the **text route**. They are *not* embedded into the CLIP
  image index — that channel is reserved for standalone ``images/`` uploads
  (``source_type=image``). A caption is text; it goes into the text index.
- Works with any OpenAI-compatible VLM via ``VLM_*``. No model-specific code.

Pipeline position: ``service._do_parse`` calls this *before* the Contextual
Retrieval pass, so the contextual LLM sees chunks that already include figure
captions (richer context). Cached under ``captions/<asset_id>.jsonl`` keyed by
image path; ``mmrag reindex`` and force re-parse reuse it because figure bytes
are stable across re-parses (the cache lives outside ``parsed/<id>/``, which
force clears).

Degradation: when ``VLM_*`` is unconfigured or a request fails, the caption
is left empty and the chunk text is untouched — identical to the pre-caption
behavior. Nothing here raises.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .paths import get_parsed_dir
from .settings import get_settings


def _image_abs_path(asset_id: str, image_rel_path: str) -> Path | None:
    """Resolve an ``images/<fname>`` ref to its on-disk file.

    ``image_rel_path`` is the ``metadata["images"][*]["path"]`` shape
    (``"images/markitdown_abc.png"``), relative to ``parsed/<asset_id>/``.
    Returns ``None`` when the file is missing so the caller can skip it
    without raising — an unrenderable figure shouldn't abort captioning the
    rest.

    Defense-in-depth: refuse paths that escape ``parsed/<asset_id>/``
    (absolute paths, ``..`` traversal, or symlinks resolving outside).
    ``image_rel_path`` comes from parser-emitted metadata today, but
    bounding it stops a future parser path bug from reading arbitrary files.
    """
    base = get_parsed_dir() / asset_id
    candidate = base / image_rel_path
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        return None
    return resolved if resolved.exists() else None


def _load_cache(cache_path: Path) -> dict[str, str]:
    """Read ``captions/<asset_id>.jsonl`` → ``{image_path: caption}``.

    One JSON object per line: ``{"path": "images/x.png", "caption": "…"}``.
    A missing / unreadable cache yields ``{}`` — callers treat absence as
    "not yet captioned" and will (re)generate.
    """
    if not cache_path.exists():
        return {}
    cache: dict[str, str] = {}
    try:
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cap = str(obj.get("caption", "") or "").strip()
            if cap:
                cache[str(obj.get("path", ""))] = cap
    except Exception:
        return {}
    return cache


def _write_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Persist ``{image_path: caption}`` for reuse across reindex / re-parse.

    Only written when at least one caption was produced (or reused), so an
    unconfigured-VLM run leaves the captions dir clean — matching the
    contextual cache's no-op-on-empty behaviour.
    """
    if not cache:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            for path, cap in cache.items():
                f.write(json.dumps({"path": path, "caption": cap}, ensure_ascii=False) + "\n")
    except Exception as exc:  # disk full / permission — degrade, but log
        import logging

        logging.getLogger(__name__).warning(
            "image caption cache write failed for %s; a later reindex will "
            "re-invoke the VLM per figure instead of reusing the cache: %s",
            cache_path,
            exc,
        )


def _caption_one(asset_id: str, image_rel_path: str) -> str:
    """Generate a concise retrieval caption for one embedded figure.

    Returns ``""`` on any failure (missing file, missing VLM creds, request
    error) so the caller leaves the chunk untouched.

    Uses a *concise* prompt rather than reusing
    :func:`parsers.image_parser.call_vlm_caption`: that function's prompt
    asks for a five-dimension detailed description (title / objects / text /
    scene / answerable questions) which suits a standalone image the answer
    layer cites verbatim, but produces ~1600-char essays that would bloat a
    chunk's text and dilute its BM25 signal. An embedded figure's caption
    here is retrieval bait — short, term-rich ("AI模型性能测试表格,含TTFT/TPOT指标")
    — so the figure becomes searchable without distorting its chunk. The
    transport (OpenAI-compatible vision chat) is identical; only the prompt
    differs. Model-agnostic: any VLM returns a short description to a short
    prompt.

    Reasoning-model aware: some VLMs (gemma3/4 thinking, o1-style, deepseek-r1,
    …) emit their answer in a ``reasoning`` / ``thinking`` field and leave
    ``content`` empty — or only populate ``content`` after a long internal
    monologue. We reuse the configured ``VLM_MAX_TOKENS`` (not a tiny hard
    cap) so a reasoning model has budget to finish thinking and emit the
    caption, and fall back to the reasoning field when ``content`` is empty.
    This mirrors how the rest of the project handles reasoning models
    (``contextual._THINK_RE``) and is not specific to any one model.
    """
    abs_path = _image_abs_path(asset_id, image_rel_path)
    if abs_path is None:
        return ""
    import base64

    import requests

    s = get_settings()
    base_url, api_key, model = s.vlm_creds
    if not base_url or not api_key or not model:
        return ""
    try:
        from .answer import _warn_insecure_base_url

        _warn_insecure_base_url(base_url)
    except Exception:  # pragma: no cover - never block captioning
        pass
    try:
        image_base64 = base64.b64encode(abs_path.read_bytes()).decode("ascii")
        suffix = abs_path.suffix.lower().replace(".", "") or "png"
        mime = "jpeg" if suffix == "jpg" else suffix
        payload = {
            "model": model,
            "temperature": s.vlm_temperature,
            "max_tokens": s.vlm_max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "用中文简洁描述这张图片,50字以内。突出可检索的对象名称、"
                                "图中可见文字、场景和图表类型(如表格/流程图/示意图)。"
                                "只输出描述本身,不要分点,不要解释。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{image_base64}"},
                        },
                    ],
                }
            ],
        }
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=s.vlm_timeout,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        content = str(message.get("content") or "").strip()
        # Reasoning-model fallback: when content is empty the answer may live
        # in the reasoning/thinking field. Strip leading "Thinking Process:"
        # / "思考:" style prefaces and take the last sentence (reasoning
        # narratives often end with the concrete description).
        if not content:
            raw = str(message.get("reasoning") or message.get("thinking") or "").strip()
            if raw:
                content = _strip_reasoning_preface(raw)
        return content
    except Exception:
        return ""


def _strip_reasoning_preface(raw: str) -> str:
    """Best-effort recovery of a caption from a reasoning-model's thought dump.

    Reasoning models (gemma thinking, deepseek-r1, …) sometimes emit the
    whole answer inside ``reasoning`` with an empty ``content``. The raw
    narrative usually opens with a meta preface ("Thinking Process:", "思考过程:",
    "1. **Analyze...") and ends with the concrete description. We drop the
    preface lines and keep the final non-empty sentence(s), capped so a
    runaway monologue doesn't bloat the chunk. Model-agnostic heuristic; if
    it yields nothing usable the caller treats the caption as empty.
    """
    import re

    # Drop common preface openers and numbered analysis step lines.
    cleaned = re.sub(
        r"^(Thinking Process|思考过程|思考|分析|Analyze|Reasoning)\s*[:：].*",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    # Split into sentences on CJK / Latin terminators.
    sentences = re.split(r"[。.!！?？\n]+", cleaned)
    # strip leading bullets / numbering / punctuation from each sentence.
    _bullet_re = re.compile(r"^[\s*\-•0-9.、]+")
    sentences = [_bullet_re.sub("", s).strip() for s in sentences]
    sentences = [s for s in sentences if s]
    if not sentences:
        return ""
    # The concrete description tends to be the last 1-2 sentences; cap length.
    tail = "。".join(sentences[-2:])
    return tail[:200]


def enrich_docs_with_image_captions(
    docs: list,  # list[ParsedDocument] — typed loosely to avoid a circular import
    *,
    asset_id: str,
    cache_path: Path | None = None,
) -> None:
    """Append a VLM caption for each embedded figure to its chunk's text, in place.

    Called from ``service._do_parse`` before the Contextual Retrieval pass.
    For every ``metadata["images"]`` entry whose ``caption`` is empty, generate
    a Chinese description (cached / concurrent), then:

    * set ``metadata["images"][i]["caption"]`` so the answer layer can cite it;
    * append ``"图片描述: <caption>"`` to the chunk's ``text`` so the dense +
      BM25 channels index the figure's semantics.

    Figures that already carry a caption (PyMuPDF "图N: …" detection) are
    reused as-is — their caption is appended too, so a referenced figure's
    detected label also enters the text index. De-dupes by image path within a
    chunk (overlap may associate one figure with two chunks; each chunk gets
    the caption once). A no-op when ``image_caption_enabled`` is false, when
    ``VLM_*`` is unconfigured, or when the asset has no embedded figures.
    """
    if not docs:
        return
    s = get_settings()
    if not s.image_caption_enabled:
        return
    # VLM creds gate: skip entirely when no VLM is configured (no cost, no
    # cache). Mirrors contextual's no-op-on-unconfigured-LLM contract.
    base_url, api_key, model = s.vlm_creds
    if not base_url or not api_key or not model:
        return

    cache: dict[str, str] = {}
    if cache_path is not None:
        cache = _load_cache(Path(cache_path))

    # Collect every distinct image path referenced across chunks, with its
    # existing caption (if any). Only paths needing a caption (empty in the
    # chunk's image entry) are queued for VLM generation.
    existing_caption: dict[str, str] = {}
    need_caption: set[str] = set()
    for d in docs:
        for img in d.metadata.get("images") or []:
            if not isinstance(img, dict):
                continue
            path = str(img.get("path") or "")
            if not path:
                continue
            cap = str(img.get("caption") or "").strip()
            if cap:
                existing_caption[path] = cap
            elif path not in existing_caption:
                need_caption.add(path)

    # Fill from cache first (figure bytes are stable → cache hits are safe).
    still_need = set()
    for path in need_caption:
        cached = cache.get(path, "")
        if cached:
            existing_caption[path] = cached
        else:
            still_need.add(path)

    # Generate the rest concurrently. One VLM call per uncached figure.
    if still_need:
        max_workers = max(1, s.image_caption_concurrency)

        def _one(path: str) -> tuple[str, str]:
            return path, _caption_one(asset_id, path)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for path, cap in ex.map(_one, sorted(still_need)):
                if cap:
                    existing_caption[path] = cap
                    cache[path] = cap

    if not existing_caption:
        # Nothing was captioned (no figures, or every VLM call failed) — leave
        # docs untouched and skip the cache write (nothing worth persisting).
        return

    # Apply: stamp each chunk's image entries + append the caption to text.
    for d in docs:
        images = d.metadata.get("images") or []
        seen_in_chunk: set[str] = set()
        appended: list[str] = []
        for img in images:
            if not isinstance(img, dict):
                continue
            path = str(img.get("path") or "")
            cap = existing_caption.get(path, "")
            if cap:
                img["caption"] = cap
                if path not in seen_in_chunk:
                    seen_in_chunk.add(path)
                    appended.append(cap)
        if appended:
            d.text = d.text.rstrip() + "\n\n图片描述: " + " ".join(appended)

    if cache_path is not None:
        _write_cache(Path(cache_path), cache)
