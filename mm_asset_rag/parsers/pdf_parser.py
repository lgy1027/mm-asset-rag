"""PDF and image parsers.

PDF parsing supports two backends:

* **PyMuPDF** (default, local, free) — extracts text per page directly from the
  PDF stream. Fast and good enough for text-based PDFs.
* **PaddleOCR-VL** (online API, optional) — layout-aware OCR via
  ``https://paddleocr.aistudio-app.com``. Useful for scanned PDFs or when
  you need structured Markdown with table/chart recovery.

Image parsing supports OCR HTTP and an OpenAI-compatible VLM caption.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
from pathlib import Path

import fitz
import requests

from ..assets import Asset
from ..paths import get_parsed_dir
from ..schema import ParsedDocument
from ..settings import get_settings

# ─── PyMuPDF ──────────────────────────────────────────────────────────────


def parse_pdf_with_pymupdf(asset: Asset) -> list[ParsedDocument]:
    """Local fallback parser for text-based PDFs.

    Each page is split by detected heading boundaries (ATX ``#``,
    font-size heuristic, standalone short line). A page that yields
    multiple sections becomes multiple chunks, each with a
    ``metadata.section`` field carrying the heading text. The keyword
    enrichment footer (Chinese jieba TextRank) is appended to every
    chunk's text so the BM25 channel can match short user queries
    like ``联宝 ESG`` against long PDF bodies.
    """
    from .chunk_splitter import split_by_heading

    output_dir = get_parsed_dir() / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    docs: list[ParsedDocument] = []

    with fitz.open(asset.file_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_dict = page.get_text("dict")
            # Build a parallel list of per-line font sizes for the
            # heading heuristic. Each ``block`` -> ``line`` -> ``span``
            # carries a size; we collapse to one size per line by
            # averaging spans.
            font_sizes: list[float] = []
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        font_sizes.append(0.0)
                        continue
                    avg = sum(s.get("size", 0.0) for s in spans) / len(spans)
                    font_sizes.append(avg)
            text = page.get_text("text").strip()
            if not text:
                continue
            markdown_path = output_dir / f"page_{page_index}.md"
            markdown_path.write_text(text, encoding="utf-8")
            sections = split_by_heading(text, font_sizes=font_sizes)
            for chunk_index, section in enumerate(sections):
                # Skip sections with no body — empty chunks (e.g. a
                # bare "1" heading with no following text) would
                # otherwise pollute the BM25 channel with a placeholder
                # payload that drags down dense ranking.
                if not section.body.strip():
                    continue
                enriched_text = _maybe_enrich_with_keywords(section.body)
                docs.append(
                    ParsedDocument(
                        text=enriched_text,
                        metadata={
                            "asset_id": asset.asset_id,
                            "asset_title": asset.title,
                            "source_type": asset.source_type,
                            "source_path": asset.relative_path,
                            "source_url": asset.source_url,
                            "page": page_index,
                            "chunk_index": chunk_index,
                            "section": section.heading,
                            "parser": "pymupdf",
                            "markdown_path": str(markdown_path),
                            "tags": asset.tags,
                        },
                    )
                )
    return docs


def _maybe_enrich_with_keywords(text: str) -> str:
    """Append a "关键词: ..." footer to a chunk when Settings enables it.

    The footer is a deterministic injection point — BM25 tokenises it
    like any other text, but the leading label makes it easy to spot
    in retrieval debug output. Pure function (no I/O) so it's cheap
    to call inside the per-page loop.
    """
    from ..settings import get_settings
    from ..text_keywords import enrich_chunk_text, extract_keywords

    s = get_settings()
    if not s.enrich_chunk_with_keywords:
        return text
    kws = extract_keywords(
        text, top_k=s.enrich_chunk_keyword_top_k, language=s.enrich_chunk_language
    )
    return enrich_chunk_text(text, kws)


# ─── PaddleOCR-VL ──────────────────────────────────────────────────────────


def _is_url(value: Path | str) -> bool:
    return isinstance(value, str) and urllib.parse.urlparse(value).scheme in {"http", "https"}


def submit_paddleocr_vl_job(file_path: Path | str) -> str:
    """Submit a PaddleOCR-VL job. Accepts a local Path or an http(s) URL string."""
    settings = get_settings()
    token = settings.paddleocr_vl_api_token
    if not token:
        raise RuntimeError("PADDLEOCR_VL_API_TOKEN is required for PaddleOCR-VL parsing")

    job_url = settings.paddleocr_vl_job_url
    model = settings.paddleocr_vl_model
    timeout = settings.paddleocr_vl_timeout

    optional_payload = {
        "useDocOrientationClassify": settings.paddleocr_vl_use_doc_orientation_classify,
        "useDocUnwarping": settings.paddleocr_vl_use_doc_unwarping,
        "useChartRecognition": settings.paddleocr_vl_use_chart_recognition,
    }

    headers = {"Authorization": f"bearer {token}"}

    if _is_url(file_path):
        # URL mode — JSON body with fileUrl
        headers["Content-Type"] = "application/json"
        payload = {
            "fileUrl": str(file_path),
            "model": model,
            "optionalPayload": optional_payload,
        }
        response = requests.post(job_url, json=payload, headers=headers, timeout=timeout)
    else:
        # Local file mode — multipart upload
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        with path.open("rb") as file_obj:
            response = requests.post(
                job_url,
                headers=headers,
                data={
                    "model": model,
                    "optionalPayload": json.dumps(optional_payload, ensure_ascii=False),
                },
                files={"file": file_obj},
                timeout=timeout,
            )

    if response.status_code != 200:
        raise RuntimeError(f"PaddleOCR-VL submit failed: {response.status_code} {response.text}")
    return str(response.json()["data"]["jobId"])


def poll_paddleocr_vl_job(job_id: str) -> str:
    """Poll the job until done. Logs ``total_pages / extracted_pages`` progress.

    Returns the URL of the JSONL result document.
    """
    settings = get_settings()
    token = settings.paddleocr_vl_api_token
    job_url = settings.paddleocr_vl_job_url
    timeout = settings.paddleocr_vl_timeout
    poll_interval = settings.paddleocr_vl_poll_interval
    headers = {"Authorization": f"bearer {token}"}
    deadline = time.time() + timeout
    retry_attempts = settings.paddleocr_vl_poll_retry

    attempt = 0
    while time.time() < deadline:
        try:
            response = requests.get(f"{job_url}/{job_id}", headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            attempt += 1
            if attempt >= retry_attempts:
                raise RuntimeError(
                    f"PaddleOCR-VL poll network failure after {attempt} retries: {exc}"
                ) from exc
            print(f"PaddleOCR-VL poll network error (retry {attempt}/{retry_attempts}): {exc}")
            time.sleep(poll_interval * attempt)
            continue

        attempt = 0  # reset on success
        if response.status_code != 200:
            raise RuntimeError(f"PaddleOCR-VL poll failed: {response.status_code} {response.text}")
        payload = response.json()["data"]
        state = payload["state"]
        if state == "done":
            return str(payload["resultUrl"]["jsonUrl"])
        if state == "failed":
            raise RuntimeError(f"PaddleOCR-VL job failed: {payload.get('errorMsg')}")

        progress = payload.get("extractProgress") or {}
        total = progress.get("totalPages")
        done = progress.get("extractedPages")
        if isinstance(total, int) and isinstance(done, int):
            print(f"PaddleOCR-VL job {job_id}: {state} ({done}/{total} pages)")
        else:
            print(f"PaddleOCR-VL job {job_id}: {state}")
        time.sleep(poll_interval)
    raise RuntimeError(f"PaddleOCR-VL job timeout: {job_id}")


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _download_ocr_images(page_result: dict, images_dir: Path) -> dict[str, str]:
    """Download images referenced by the OCR page result.

    Returns a mapping ``remote_url -> "images/xxxx.png"`` (relative path
    suitable for embedding in the markdown body of the same directory).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    md_images = (page_result.get("markdown") or {}).get("images") or {}
    output_images = page_result.get("outputImages") or {}

    for url in list(md_images.keys()) + list(output_images.keys()):
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if url in mapping:
            continue
        try:
            digest = hashlib.md5(url.encode()).hexdigest()[:12]
            # Try to keep the original suffix; default to .png.
            suffix = Path(url.split("?", 1)[0]).suffix.lower()
            if suffix not in _IMAGE_SUFFIXES:
                suffix = ".png"
            local_abs = images_dir / f"img_{digest}{suffix}"
            if not local_abs.exists():
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                local_abs.write_bytes(resp.content)
            mapping[url] = f"images/{local_abs.name}"
        except Exception as exc:
            print(f"OCR image download skipped ({url}): {exc}")
    return mapping


def parse_with_paddleocr_vl(asset: Asset) -> list[ParsedDocument]:
    """Real PaddleOCR-VL OCR via the online API.

    The asset's ``source_url`` is preferred when set, so the file is
    uploaded once to PaddleOCR's storage instead of streamed through us.
    """
    output_dir = get_parsed_dir() / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    raw_jsonl_path = output_dir / "raw.jsonl"

    if raw_jsonl_path.exists() and raw_jsonl_path.stat().st_size > 0:
        jsonl_text = raw_jsonl_path.read_text(encoding="utf-8")
    else:
        if not asset.file_path.exists() and not asset.source_url:
            raise FileNotFoundError(f"PDF not found: {asset.file_path}")

        # Prefer source_url (offloads upload to PaddleOCR's storage), but
        # fall back to local-file upload if the URL is rejected by the
        # backend (common when the URL is behind anti-bot, e.g. arXiv).
        job_id: str | None = None
        if asset.source_url:
            try:
                job_id = submit_paddleocr_vl_job(asset.source_url)
            except RuntimeError as exc:
                print(f"PaddleOCR-VL URL submit failed ({exc}); falling back to local-file upload")
                job_id = None
        if job_id is None:
            job_id = submit_paddleocr_vl_job(asset.file_path)

        jsonl_url = poll_paddleocr_vl_job(job_id)
        response = requests.get(
            jsonl_url,
            timeout=get_settings().paddleocr_vl_timeout,
        )
        response.raise_for_status()
        jsonl_text = response.text
        raw_jsonl_path.write_text(jsonl_text, encoding="utf-8")

    docs: list[ParsedDocument] = []
    page_num = 0
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        result = json.loads(line).get("result", {})
        for parsed_page in result.get("layoutParsingResults", []):
            markdown = parsed_page.get("markdown", {})
            markdown_text = str(markdown.get("text", "")).strip()
            if not markdown_text:
                page_num += 1
                continue

            # Download any embedded images for this page and rewrite
            # the markdown body's remote URLs to local relative paths.
            url_to_local = _download_ocr_images(parsed_page, images_dir)
            if url_to_local:
                for remote_url, local_rel in url_to_local.items():
                    markdown_text = markdown_text.replace(remote_url, local_rel)

            markdown_path = output_dir / f"page_{page_num}.md"
            markdown_path.write_text(markdown_text, encoding="utf-8")
            enriched_text = _maybe_enrich_with_keywords(markdown_text)
            docs.append(
                ParsedDocument(
                    text=enriched_text,
                    metadata={
                        "asset_id": asset.asset_id,
                        "asset_title": asset.title,
                        "source_type": asset.source_type,
                        "source_path": asset.relative_path,
                        "source_url": asset.source_url,
                        "page": page_num,
                        "parser": "paddleocr-vl-api",
                        "markdown_path": str(markdown_path),
                        "images_dir": str(images_dir) if images_dir.exists() else "",
                        "tags": asset.tags,
                    },
                )
            )
            page_num += 1
    return docs


def parse_pdf(asset: Asset, parser: str) -> list[ParsedDocument]:
    if parser == "paddleocr_vl":
        return parse_with_paddleocr_vl(asset)
    if parser == "pymupdf":
        return parse_pdf_with_pymupdf(asset)
    if parser == "auto":
        if get_settings().paddleocr_vl_api_token:
            return parse_with_paddleocr_vl(asset)
        return parse_pdf_with_pymupdf(asset)
    raise ValueError(f"Unsupported PDF parser: {parser}")
