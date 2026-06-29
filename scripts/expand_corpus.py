"""Expand the bundled sample corpus with diversified PDFs + image-only variants.

Strategy (chosen to maximise cross-scenario stability testing without
committing to one domain):

- 12 Wikipedia EN topics (food / tech / culture / geography / brands) — every
  page has figures, varied length, all under ~3 MB.
- 4 Wikipedia ZH topics (Beijing / Coffee / Tea / Hong Kong).
- 3 arXiv short papers (non-ML where possible to break the bundled ML
  bias — economics, physics, social science).
- 2 IRS forms (public domain, tiny).
- For 2 of the Wikipedia pages we also write an "image-only" variant
  (text layer stripped via PyMuPDF) to mimic the structure of a scanned
  PDF — every page still has visuals but no searchable text. This is
  the closest proxy for a true scanned document we can assemble from
  fully public-domain sources in this environment.

The output is appended to ``asset_manifest.json`` so the existing CI
sample set stays untouched. After this script:

  mmrag parse --pdf-parser pymupdf    # adds new chunks to documents.jsonl
  mmrag reindex --text-only          # rebuilds the Qdrant text collection
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path

USER_AGENT = "mm-asset-rag/0.1 (https://github.com/lgy1027/mm-asset-rag)"
TIMEOUT = 60

# ─── Source roster ───────────────────────────────────────────────────────
# Picked for topic diversity, public-domain licensing, and small size.
# Each tuple: (asset_id, title, source_url, category, tags).
#
# License tag convention (added by this file):
#   license-cc-by-sa      Wikipedia EN/ZH (CC-BY-SA 4.0)
#   license-arxiv-perp    arXiv (perpetual, non-exclusive)
#   license-public-domain IRS forms (US federal government work)
# A consolidated ``LICENSE-NOTES.md`` in the same directory spells out
# the per-source license mapping for users who redistribute the bundled
# data.

WIKI_EN = [
    ("wiki_en_monalisa", "Mona Lisa (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Mona_Lisa",
     "wiki_en", ["wikipedia", "art", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_bicycle", "Bicycle (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Bicycle",
     "wiki_en", ["wikipedia", "transport", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_pizza", "Pizza (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Pizza",
     "wiki_en", ["wikipedia", "food", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_cocacola", "Coca-Cola (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Coca-Cola",
     "wiki_en", ["wikipedia", "brand", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_piano", "Piano (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Piano",
     "wiki_en", ["wikipedia", "music", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_mushroom", "Mushroom (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Mushroom",
     "wiki_en", ["wikipedia", "biology", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_telescope", "Telescope (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Telescope",
     "wiki_en", ["wikipedia", "science", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_photography", "Photography (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Photography",
     "wiki_en", ["wikipedia", "media", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_qrcode", "QR code (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/QR_code",
     "wiki_en", ["wikipedia", "tech", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_pdf", "PDF (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/PDF",
     "wiki_en", ["wikipedia", "tech", "license-cc-by-sa"]),
    ("wiki_en_apollo11", "Apollo 11 (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Apollo_11",
     "wiki_en", ["wikipedia", "history", "image-rich", "license-cc-by-sa"]),
    ("wiki_en_sony", "Sony (Wikipedia EN)",
     "https://en.wikipedia.org/api/rest_v1/page/pdf/Sony",
     "wiki_en", ["wikipedia", "company", "image-rich", "license-cc-by-sa"]),
]

WIKI_ZH = [
    ("wiki_zh_beijing", "北京 (Wikipedia ZH)",
     "https://zh.wikipedia.org/api/rest_v1/page/pdf/%E5%8C%97%E4%BA%AC",
     "wiki_zh", ["wikipedia", "city", "image-rich", "chinese", "license-cc-by-sa"]),
    ("wiki_zh_kfc", "肯德基 (Wikipedia ZH)",
     "https://zh.wikipedia.org/api/rest_v1/page/pdf/%E8%82%AF%E5%BE%B7%E5%9F%BA",
     "wiki_zh", ["wikipedia", "brand", "image-rich", "chinese", "license-cc-by-sa"]),
    ("wiki_zh_coffee", "咖啡 (Wikipedia ZH)",
     "https://zh.wikipedia.org/api/rest_v1/page/pdf/%E5%92%96%E5%95%A1",
     "wiki_zh", ["wikipedia", "food", "image-rich", "chinese", "license-cc-by-sa"]),
    ("wiki_zh_panda", "大熊猫 (Wikipedia ZH)",
     "https://zh.wikipedia.org/api/rest_v1/page/pdf/%E5%A4%A7%E7%86%8A%E7%8C%AB",
     "wiki_zh", ["wikipedia", "biology", "image-rich", "chinese", "license-cc-by-sa"]),
]

ARXIV = [
    # Short papers, deliberately non-ML where possible. arXiv grants a
    # perpetual, non-exclusive licence to distribute; specific papers may
    # carry additional CC-BY or other marks declared by their authors.
    ("arxiv_phys_thermo", "Thermodynamics for economists (arXiv)",
     "https://arxiv.org/pdf/physics/0507011",
     "arxiv", ["arxiv", "economics", "physics", "license-arxiv-perp"]),
    ("arxiv_stat_econ", "Statistical mechanics of money (arXiv)",
     "https://arxiv.org/pdf/cond-mat/0212011",
     "arxiv", ["arxiv", "economics", "stat-mech", "license-arxiv-perp"]),
    ("arxiv_qbio_neuro", "Simple model of spiking neurons (arXiv)",
     "https://arxiv.org/pdf/q-bio/0312024",
     "arxiv", ["arxiv", "biology", "short", "license-arxiv-perp"]),
]

IRS = [
    # US federal-government works are in the public domain in the
    # United States (17 U.S.C. § 105).
    ("irs_w9", "IRS Form W-9",
     "https://www.irs.gov/pub/irs-pdf/fw9.pdf",
     "irs", ["form", "tax", "table", "license-public-domain"]),
    ("irs_w4", "IRS Form W-4",
     "https://www.irs.gov/pub/irs-pdf/fw4.pdf",
     "irs", ["form", "tax", "table", "license-public-domain"]),
]

ALL_SOURCES = WIKI_EN + WIKI_ZH + ARXIV + IRS


def download(url: str, dest: Path, max_bytes: int = 6 * 1024 * 1024) -> bool:
    """Download ``url`` to ``dest`` if size is under ``max_bytes``."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
        if len(data) > max_bytes:
            print(f"  SKIP (too large {len(data)//1024} KB): {url}")
            return False
        if not data.startswith(b"%PDF"):
            print(f"  SKIP (not a PDF, head={data[:6]!r}): {url}")
            return False
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  FAIL ({type(exc).__name__}: {str(exc)[:60]}): {url}")
        return False


def strip_text_layer(src: Path, dest: Path) -> bool:
    """Write ``dest`` as a copy of ``src`` with every text annotation removed.

    The images stay — this produces a PDF that, structurally, is
    indistinguishable from a scanned image-only document from the
    parser's perspective (PyMuPDF / Tesseract / etc. will see no text
    layer and fall back to OCR). Used here as the public-domain
    surrogate for scanned PDFs.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"  scan-strip SKIP (PyMuPDF not available): {src}")
        return False
    doc = fitz.open(src)
    for page in doc:
        # Remove text while keeping the rasterized content.
        page.add_redact_annot(page.rect, fill=None)
        page.apply_redactions()
        # Re-render the page as an image-only PDF page so the visual
        # structure is preserved without any hidden text layer.
        pix = page.get_pixmap(dpi=150)
        page.clean_contents()
        page.insert_image(page.rect, pixmap=pix)
    doc.save(dest, garbage=4, deflate=True, clean=True)
    doc.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--assets-dir",
        default="examples/data/chapter11_assets",
        help="Root of the bundled asset directory.",
    )
    parser.add_argument(
        "--strip-text-ids",
        default="wiki_en_pizza,wiki_en_monalisa",
        help="Comma-separated asset_ids whose PDFs should also be written "
             "as image-only (no text layer) variants for scan-PDF testing.",
    )
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    pdfs_dir = assets_dir / "pdfs"
    manifest_path = assets_dir / "asset_manifest.json"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: check which sources are downloadable and small enough.
    new_records: list[dict] = []
    for asset_id, title, url, category, tags in ALL_SOURCES:
        dest = pdfs_dir / f"{asset_id}.pdf"
        if dest.exists():
            print(f"  EXISTS, skipping download: {dest.name}")
        else:
            print(f"  GET {url[:90]}")
            if not download(url, dest):
                continue
        new_records.append({
            "id": asset_id,
            "title": title,
            "type": "pdf",
            "path": f"pdfs/{asset_id}.pdf",
            "source_url": url,
            "tags": tags,
        })

    # Scan-style variants: take an existing PDF and strip its text layer.
    strip_ids = {s.strip() for s in args.strip_text_ids.split(",") if s.strip()}
    for rec in new_records:
        if rec["id"] in strip_ids:
            src = pdfs_dir / f"{rec['id']}.pdf"
            scan_dest = pdfs_dir / f"{rec['id']}__scan.pdf"
            if scan_dest.exists():
                print(f"  scan variant EXISTS: {scan_dest.name}")
            elif strip_text_layer(src, scan_dest):
                print(f"  scan variant WROTE: {scan_dest.name}")
            scan_id = f"{rec['id']}__scan"
            new_records.append({
                "id": scan_id,
                "title": f"{rec['title']} (image-only / scan variant)",
                "type": "pdf",
                "path": f"pdfs/{scan_id}.pdf",
                "source_url": rec["source_url"],
                "tags": list(rec["tags"]) + ["scan", "image-only"],
            })

    if not new_records:
        print("No new PDFs downloaded — nothing to append.")
        return

    # Append to asset_manifest.json. Use forward slashes for cross-platform.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {rec["id"]: rec for rec in manifest["records"]}
    appended: list[dict] = []
    merged: list[tuple[str, list[str]]] = []
    for rec in new_records:
        if rec["id"] in by_id:
            existing = by_id[rec["id"]]
            old_tags = list(existing.get("tags", []))
            new_tags = list(rec.get("tags", []))
            # Union preserving insertion order; new tags win on duplicates.
            seen = set()
            merged_tags: list[str] = []
            for t in old_tags + new_tags:
                if t not in seen:
                    seen.add(t)
                    merged_tags.append(t)
            if merged_tags != old_tags:
                existing["tags"] = merged_tags
                merged.append((rec["id"], merged_tags))
        else:
            manifest["records"].append(rec)
            appended.append(rec)
    manifest["total"] = len(manifest["records"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nAppended {len(appended)} new records to {manifest_path}.")
    if merged:
        print(f"Merged tags into {len(merged)} existing records:")
        for aid, tags in merged:
            print(f"  {aid}: {tags}")
    print(f"Manifest now has {manifest['total']} assets.")


if __name__ == "__main__":
    main()