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
from .document_ir import Block, DocumentIR, ImageRef, PageHint

# ─── PyMuPDF ──────────────────────────────────────────────────────────────


def build_ir_pymupdf(asset: Asset) -> DocumentIR:
    """Local PyMuPDF extraction → ``DocumentIR`` (text blocks + images).

    The "format → IR" half of the parse. Walks ``page.get_text("dict")``
    once per page to build text blocks (each line's bbox + font size feeds
    the heading heuristic downstream) and calls ``extract_page_images`` +
    ``detect_figure_captions`` to collect images with their captions.
    Chunking, image↔chunk association, keyword enrichment and metadata
    assembly happen in the shared ``ir_to_documents`` layer, not here —
    so this function does only what PyMuPDF uniquely knows: page geometry
    and font sizes.
    """
    from .chunk_splitter import _make_token_counter  # noqa: F401  (kept for callers)
    from .pdf_images import (
        LineItem,
        detect_figure_captions,
        extract_page_images,
    )

    settings = get_settings()
    extract_images = settings.pdf_extract_images
    output_dir = get_parsed_dir() / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    blocks: list[Block] = []
    images: list[ImageRef] = []
    markdown_paths: list[str] = []
    page_hints: dict[int, PageHint] = {}

    with fitz.open(asset.file_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_dict = page.get_text("dict")
            # Build per-line geometry + font info in one dict walk. The
            # heading detection itself runs later (split_with_recursion →
            # split_by_heading); here we only preserve the raw per-line
            # text + bbox + avg font size so the splitter's font-size and
            # standalone-short-line heuristics keep working.
            font_sizes: list[float] = []
            line_bboxes: list = []
            line_items: list[LineItem] = []
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_bbox = line.get("bbox")
                    bbox_tuple = tuple(line_bbox) if line_bbox else None
                    if not spans:
                        font_sizes.append(0.0)
                        line_bboxes.append(bbox_tuple)
                        line_items.append(LineItem(text="", bbox=bbox_tuple))
                        continue
                    avg = sum(s.get("size", 0.0) for s in spans) / len(spans)
                    font_sizes.append(avg)
                    line_bboxes.append(bbox_tuple)
                    line_text = "".join(s.get("text", "") for s in spans)
                    line_items.append(LineItem(text=line_text, bbox=bbox_tuple, size=avg))
            text = page.get_text("text").strip()
            if not text:
                continue
            markdown_path = output_dir / f"page_{page_index}.md"
            markdown_path.write_text(text, encoding="utf-8")
            markdown_paths.append(str(markdown_path))

            # One Block per page carrying the whole-page text. The heading
            # boundaries and per-line bboxes are passed via page_hints so
            # split_with_recursion keeps working unchanged — the splitter's
            # font-size / standalone-short-line heuristics need the parallel
            # per-line lists, which don't fit the Block shape.
            blocks.append(Block(text=text, page=page_index, bbox=None))
            page_hints[page_index] = PageHint(font_sizes=font_sizes, line_bboxes=line_bboxes)

            # Extract embedded images + resolve figure captions for this
            # page. Captions are detected from line_items + image bboxes.
            page_images = []
            page_figures: dict = {}
            if extract_images:
                page_images = extract_page_images(
                    page, pdf, page_index, images_dir, min_dim=settings.pdf_image_min_dim
                )
                if page_images:
                    page_figures = detect_figure_captions(line_items, page_images)
            for pi in page_images:
                fig = next(
                    (f for f in page_figures.values() if f.image_path == pi.path),
                    None,
                )
                images.append(
                    ImageRef(
                        path=pi.path,
                        page=pi.page,
                        bbox=pi.bbox,
                        figure_id=fig.number if fig else None,
                        caption=fig.caption if fig else "",
                    )
                )

    return DocumentIR(
        blocks=blocks,
        images=images,
        asset=asset,
        parser="pymupdf",
        markdown_paths=markdown_paths,
        page_hints=page_hints,
    )


def parse_pdf_with_pymupdf(asset: Asset) -> list[ParsedDocument]:
    """PyMuPDF parse — ``build_ir_pymupdf`` + shared ``ir_to_documents``.

    Thin wrapper kept so existing callers (registry, tests) keep working
    after the IR refactor; the real work moved to the IR layer.
    """
    from .document_ir import ir_to_documents

    return ir_to_documents(build_ir_pymupdf(asset))


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


def build_ir_paddleocr_vl(asset: Asset) -> DocumentIR:
    """Remote PaddleOCR-VL OCR → ``DocumentIR`` (markdown blocks + images).

    The "format → IR" half. Submits/polls/downloads the OCR JSONL (with
    ``raw.jsonl`` caching), rewrites remote image URLs to local paths,
    and turns each page's markdown into one ``Block``. Chunking, image↔
    chunk association (span-based, re-scanned per sub-chunk) and metadata
    assembly happen in ``ir_to_documents`` — so this function does only
    what the remote OCR uniquely knows: layout-aware markdown + remote
    image URLs. The ``source_url`` is preferred for submit so the file is
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

    blocks: list[Block] = []
    markdown_paths: list[str] = []
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
            # The markdown text (with local paths) is what gets chunked,
            # so extract_markdown_image_refs finds local ``![]()`` spans.
            url_to_local = _download_ocr_images(parsed_page, images_dir)
            if url_to_local:
                for remote_url, local_rel in url_to_local.items():
                    markdown_text = markdown_text.replace(remote_url, local_rel)

            markdown_path = output_dir / f"page_{page_num}.md"
            markdown_path.write_text(markdown_text, encoding="utf-8")
            markdown_paths.append(str(markdown_path))
            # One Block per page. The markdown already carries ATX headings
            # which recursive_split's separator hierarchy respects; no
            # page_hints needed (the splitter runs on the markdown text
            # alone, unlike PyMuPDF which feeds per-line font sizes).
            blocks.append(Block(text=markdown_text, page=page_num, bbox=None))
            page_num += 1

    return DocumentIR(
        blocks=blocks,
        images=[],  # PaddleOCR images are resolved per-sub-chunk by span
        # in ir_to_documents (the ref must physically appear in the chunk
        # text, which only exists after splitting) — not collected here.
        asset=asset,
        parser="paddleocr-vl-api",
        markdown_paths=markdown_paths,
        images_dir=str(images_dir) if images_dir.exists() else "",
    )


def parse_with_paddleocr_vl(asset: Asset) -> list[ParsedDocument]:
    """PaddleOCR-VL parse — ``build_ir_paddleocr_vl`` + shared ``ir_to_documents``.

    Thin wrapper kept so existing callers (registry, tests) keep working
    after the IR refactor; the real work moved to the IR layer.
    """
    from .document_ir import ir_to_documents

    return ir_to_documents(build_ir_paddleocr_vl(asset))


def parse_pdf(asset: Asset, parser: str) -> list[ParsedDocument]:
    """Dispatch a PDF parse by backend name.

    ``auto`` (the default) now does scanned-PDF detection: it runs the
    fast local PyMuPDF extraction first, and if the resulting text density
    is below ``Settings.pdf_scan_text_threshold`` (image-only / scanned
    pages) it falls back to an OCR backend chosen by
    ``Settings.pdf_scan_fallback_parser`` (default ``paddleocr_vl``,
    optionally ``docling`` when the extra is installed). This replaces the
    old "token-configured → always OCR, else always local" behaviour,
    which sent text PDFs through OCR unnecessarily and silently dropped
    scanned PDFs to zero chunks when no token was configured.
    """
    from .document_ir import ir_to_documents, looks_scanned

    if parser == "paddleocr_vl":
        return parse_with_paddleocr_vl(asset)
    if parser == "pymupdf":
        return parse_pdf_with_pymupdf(asset)
    if parser == "auto":
        settings = get_settings()
        ir = build_ir_pymupdf(asset)
        if settings.pdf_scan_fallback_enabled and looks_scanned(
            ir, text_threshold_per_page=settings.pdf_scan_text_threshold
        ):
            fallback = settings.pdf_scan_fallback_parser
            if fallback == "docling":
                return parse_with_docling(asset)
            # Default fallback: PaddleOCR-VL. Only fall back when its API
            # token is configured — otherwise we'd raise where the old
            # code silently produced zero chunks. A bare token-less scan
            # stays on the (empty) PyMuPDF output to preserve behaviour.
            if settings.paddleocr_vl_api_token:
                return parse_with_paddleocr_vl(asset)
        return ir_to_documents(ir)
    if parser == "docling":
        return parse_with_docling(asset)
    raise ValueError(f"Unsupported PDF parser: {parser}")


def parse_with_docling(asset: Asset) -> list[ParsedDocument]:
    """docling parse — lazy, raises a friendly error when the extra is missing.

    docling is the optional heavy backend (torch / transformers); the
    ``[docling]`` extra must be installed. The lazy import means dispatch
    is wired even without the extra — the error surfaces at parse time.
    """
    try:
        from .docling_parser import build_ir_docling  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via parse_with_docling
        raise RuntimeError(
            'docling parsing requires the [docling] extra: pip install -e ".[docling]"'
        ) from exc
    from .document_ir import ir_to_documents

    return ir_to_documents(build_ir_docling(asset))


def parse_with_markitdown(asset: Asset) -> list[ParsedDocument]:
    """MarkItDown parse — ``build_ir_markitdown`` + shared ``ir_to_documents``.

    MarkItDown is the default ``document`` backend (core dependency, so
    normally importable). The lazy import + friendly error mirrors
    ``parse_with_docling`` so a stripped install still reports the install
    hint rather than a bare ``ImportError``.
    """
    try:
        from .markitdown_parser import build_ir_markitdown
    except ImportError as exc:  # pragma: no cover - core dep, exercised via test
        raise RuntimeError(
            "MarkItDown parsing requires the markitdown package: "
            'pip install -e "."  (or pip install "markitdown[docx,pptx,xlsx]")'
        ) from exc
    from .document_ir import ir_to_documents

    return ir_to_documents(build_ir_markitdown(asset))
