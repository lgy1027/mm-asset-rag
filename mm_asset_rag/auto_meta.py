"""Auto-extract asset metadata via a vision-language model.

The upload pipeline calls ``auto_meta_image`` (and for PDFs
``auto_meta_pdf_first_page``) during the ``/upload/preview`` phase. The
model is asked to return a single JSON object with ``title``,
``description``, ``tags`` and ``dominant_objects`` — one round trip
instead of four.

Configuration
-------------

Reads from ``Settings`` (pydantic-settings singleton):

- ``auto_meta_enabled`` (default ``True``) — set to ``False`` to skip
  VLM entirely. Sniff metadata is still produced.
- ``auto_meta_image_prompt`` — override the default Chinese prompt.
- ``auto_meta_timeout`` (default 30s)
- ``auto_meta_max_tokens`` (default 800)

Network calls hit ``VLM_BASE_URL / VLM_API_KEY / VLM_MODEL`` (with the
same ``OPENAI_*`` fallbacks as the legacy caption path). When any of
those three is missing, the helpers return ``None`` and the caller
falls back to the sniff-derived title and an empty tag list.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .settings import get_settings

log = logging.getLogger(__name__)


# JSON object template the model is asked to fill. Kept here so callers
# can quote it in error messages and tests can compare prompt bodies.
DEFAULT_IMAGE_PROMPT = """请分析这张图片,只输出合法 JSON(不要 markdown 代码块、不要任何解释文字):
{{
  "title": "<一句话标题,不超过 30 字>",
  "description": "<1-3 句可检索描述,讲清楚对象/动作/场景>",
  "tags": ["<可检索的 noun tag>", "<...>"],
  "dominant_objects": ["<主要对象1>", "<...>"]
}}
tags 数量控制在 5-10 个,中文或英文 noun 均可。
"""

DEFAULT_PDF_PROMPT = """这是 PDF 第一页的截图。请分析并只输出合法 JSON(不要 markdown 代码块、不要任何解释文字):
{{
  "title": "<论文标题,不超过 50 字>",
  "description": "<1-2 句摘要,讲清楚主题/方法/结论>",
  "tags": ["<关键词1>", "<...>"],
  "page_summary": "<这一页讲了什么>"
}}
tags 数量 5-10 个,中文或英文 keyword 均可。
"""


@dataclass
class AutoMeta:
    """Structured metadata extracted by the VLM.

    All fields are user-editable in the preview UI; ``None`` here means
    the model didn't surface that field and the UI will fall back to the
    sniff-derived value.
    """

    title: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    dominant_objects: list[str] = field(default_factory=list)
    page_summary: str | None = None  # only populated for PDF
    raw_response: str | None = None  # kept for debugging the JSON parser


# ─── VLM transport ─────────────────────────────────────────────────────


def _vlm_creds() -> tuple[str, str, str] | None:
    """Return ``(base_url, api_key, model)`` or ``None`` when unconfigured.

    Delegates to :attr:`Settings.vlm_creds` (VLM_* preferred, OPENAI_*
    fallback) so the VLM channel and the LLM channel share one resolution
    rule — a deployment that configures either triple gets both paths.
    """
    base_url, api_key, model = get_settings().vlm_creds
    if not (base_url and api_key and model):
        return None
    return base_url, api_key, model


def _encode_image(path: Path) -> tuple[str, str]:
    """Return ``(data_url, mime)`` for an image file."""
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix == "jpg" else suffix
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}", mime


def _vlm_chat_json(
    text: str,
    image_data_url: str | None,
    prompt_override: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Issue a single chat completion with ``response_format=json_object``.

    Raises ``requests.HTTPError`` on a non-2xx response or
    ``KeyError``/``ValueError`` on a malformed payload. Callers catch
    and return ``None``.
    """
    creds = _vlm_creds()
    if creds is None:
        raise RuntimeError("VLM is not configured (missing base_url/api_key/model)")
    base_url, api_key, model = creds
    try:
        from .answer import _warn_insecure_base_url

        _warn_insecure_base_url(base_url)
    except Exception:  # pragma: no cover - never block ingestion
        pass

    prompt = prompt_override or DEFAULT_IMAGE_PROMPT
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt + "\n" + text}]
    if image_data_url is not None:
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": user_content}],
    }
    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=get_settings().auto_meta_timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError(f"VLM returned non-string content: {type(content).__name__}")
    return _parse_json_response(content)


# ─── JSON parsing with fallback ─────────────────────────────────────────


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse a JSON object from a model response.

    Tries strict ``json.loads`` first. If the model wrapped the JSON in
    markdown fences or preamble text, falls back to extracting the first
    ``{...}`` block via regex.
    """
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(content)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(f"VLM response is not valid JSON: {content[:120]!r}")


# ─── Field cleaning ────────────────────────────────────────────────────


def _clean_str_list(value: Any, *, limit: int = 12) -> list[str]:
    """Coerce a JSON field into a clean list of short strings."""
    if not isinstance(value, list):
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            item = str(item)
        s = item.strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


# ─── Public API ────────────────────────────────────────────────────────


def auto_meta_image(image_path: Path) -> AutoMeta | None:
    """Extract metadata from a single image.

    Returns ``None`` when the VLM is unconfigured, the request fails, or
    the response cannot be parsed as JSON. The caller is expected to
    fall back to the sniff-derived title and an empty tag list.
    """
    settings = get_settings()
    if not settings.auto_meta_enabled:
        return None
    if _vlm_creds() is None:
        return None
    try:
        data_url, _ = _encode_image(image_path)
        text = (
            "请只输出一个合法 JSON 对象,字段名为 title/description/tags/"
            "dominant_objects,不要任何其它文字。"
        )
        raw = _vlm_chat_json(
            text=text,
            image_data_url=data_url,
            prompt_override=settings.auto_meta_image_prompt,
            max_tokens=settings.auto_meta_max_tokens,
        )
    except Exception as exc:
        log.warning("auto_meta_image(%s) failed: %s", image_path.name, exc)
        return None

    return AutoMeta(
        title=_clean_optional_str(raw.get("title")),
        description=_clean_optional_str(raw.get("description")),
        tags=_clean_str_list(raw.get("tags")),
        dominant_objects=_clean_str_list(raw.get("dominant_objects")),
        raw_response=None,
    )


def auto_meta_pdf_first_page(pdf_path: Path) -> AutoMeta | None:
    """Render the PDF's first page as a PNG and feed it to the VLM.

    Returns ``None`` when PyMuPDF isn't installed, the page can't be
    rendered, or any of the image-mode failure modes fire.
    """
    settings = get_settings()
    if not settings.auto_meta_enabled:
        return None
    if _vlm_creds() is None:
        return None

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        log.warning("auto_meta_pdf_first_page: PyMuPDF not installed")
        return None

    tmp_png: Path | None = None
    try:
        with fitz.open(str(pdf_path)) as doc:
            if doc.page_count == 0:
                return None
            if doc.page_count > settings.auto_meta_pdf_max_pages:
                log.info(
                    "auto_meta_pdf_first_page(%s) skipped: %s pages > %s",
                    pdf_path.name,
                    doc.page_count,
                    settings.auto_meta_pdf_max_pages,
                )
                return None
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=settings.auto_meta_pdf_render_dpi)
            if pix.width * pix.height > settings.auto_meta_pdf_max_render_pixels:
                log.info(
                    "auto_meta_pdf_first_page(%s) skipped: render pixels %s > %s",
                    pdf_path.name,
                    pix.width * pix.height,
                    settings.auto_meta_pdf_max_render_pixels,
                )
                return None
            with tempfile.NamedTemporaryFile(suffix=".preview.png", delete=False) as tmp:
                tmp_png = Path(tmp.name)
            pix.save(str(tmp_png))

        data_url, _ = _encode_image(tmp_png)
        text = "这是 PDF 第一页截图,只输出一个合法 JSON 对象,字段 title/description/tags/page_summary。"
        raw = _vlm_chat_json(
            text=text,
            image_data_url=data_url,
            prompt_override=settings.auto_meta_pdf_prompt or DEFAULT_PDF_PROMPT,
            max_tokens=settings.auto_meta_max_tokens,
        )
    except Exception as exc:
        log.warning("auto_meta_pdf_first_page(%s) failed: %s", pdf_path.name, exc)
        return None
    finally:
        if tmp_png is not None and tmp_png.exists():
            with contextlib.suppress(OSError):
                tmp_png.unlink()

    return AutoMeta(
        title=_clean_optional_str(raw.get("title")),
        description=_clean_optional_str(raw.get("description")),
        tags=_clean_str_list(raw.get("tags")),
        dominant_objects=_clean_str_list(raw.get("dominant_objects")),
        page_summary=_clean_optional_str(raw.get("page_summary")),
        raw_response=None,
    )
