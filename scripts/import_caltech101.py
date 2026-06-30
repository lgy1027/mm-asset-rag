"""Import a labelled subset of Caltech-101 into the bundled assets.

The 20 OpenCV sample images bundled with ``mm-asset-rag`` give us a small
ground-truth base for the multimodal eval, but statistical claims from
4-13 cases per category are weak. Caltech-101 ships 101 labelled object
categories (MIT licence, ~9K images, ~130 MB on disk) under names like
``accordion``, ``airplanes``, ``sunflower`` that map cleanly onto natural
language queries. This script selects ~50 categories (skipping
``BACKGROUND_Google`` which is the negative class), takes 2-3 images per
category, and appends them to ``asset_manifest.json`` so the multimodal
eval can exercise a much larger, statistically meaningful set.

Each new record carries:

- ``id``: prefixed ``caltech_`` (so it never collides with OpenCV or
  Picsum records).
- ``title``: a short human-readable description of the category.
- ``type``: ``image``.
- ``path``: relative path under the assets dir.
- ``source_url``: the Caltech-101 page so attribution survives.
- ``tags``: ``[category, "license-caltech101", "label-source-caltech101"]``
  — the last tag distinguishes the new ground truth from the 151
  Picsum fillers that have no labels.

After running, re-index the image collection (``mmrag reindex --image-only``)
and add the new cases to ``scripts/eval_multimodal.py`` to see meaningful
hit-rate / precision numbers across 50+ categories.

Usage::

    python scripts/import_caltech101.py --per-category 2
    python scripts/import_caltech101.py --per-category 3 --only accordion,sunflower,chair
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CALETCH_ROOT = Path("/tmp/caltech101/caltech-101/101_ObjectCategories")
SOURCE_URL = "https://data.caltech.edu/records/mzrjq-6wc02"

# Picked 50 categories for topic diversity. Skipped ``BACKGROUND_Google``
# (negative class), ``Faces`` / ``Faces_easy`` (privacy-sensitive), and
# duplicates (``cougar_body`` / ``cougar_face``, ``flamingo`` /
# ``flamingo_head``, ``Leopards`` etc). Names are the on-disk directory
# names — also the natural-language label for retrieval queries.
SELECTED_CATEGORIES: list[str] = [
    # Vehicles
    "airplanes",
    "Motorbikes",
    "car_side",
    "helicopter",
    "ketch",
    "schooner",
    "ferry",
    "laptop",
    "cellphone",
    "watch",
    # Animals
    "beaver",
    "dolphin",
    "dalmatian",
    "elephant",
    "kangaroo",
    "llama",
    "panda",
    "platypus",
    "rhino",
    "rooster",
    "scorpion",
    "sea_horse",
    "snoopy",
    "starfish",
    # Objects (everyday)
    "accordion",
    "bonsai",
    "brain",
    "brontosaurus",
    "buddha",
    "camera",
    "cannon",
    "chair",
    "cup",
    "electric_guitar",
    # Food / plants
    "strawberry",
    "sunflower",
    "water_lilly",
    "lotus",
    "pizza",
    # Landmarks / scenes
    "pagoda",
    "pyramid",
    "minaret",
    "stop_sign",
    "saxophone",
    # Tools
    "stapler",
    "wrench",
    "scissors",
    "lamp",
    "revolver",
]


def _caltech_image_paths(category: str, per_category: int) -> list[Path]:
    """Return the first ``per_category`` images of ``category`` in order."""
    cat_dir = CALETCH_ROOT / category
    if not cat_dir.is_dir():
        return []
    images = sorted(cat_dir.glob("image_*.jpg"))[:per_category]
    return images


def _new_record(
    image_path: Path,
    category: str,
    index: int,
    assets_dir: Path,
) -> dict:
    """Copy the image into the assets dir and emit a manifest record."""
    safe_cat = category.lower().replace(" ", "_")
    asset_id = f"caltech_{safe_cat}_{index:02d}"
    rel_path = f"images/{asset_id}.jpg"
    target = assets_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target)
    return {
        "id": asset_id,
        "title": f"Caltech-101 / {category} (sample {index})",
        "type": "image",
        "path": rel_path,
        "source_url": f"{SOURCE_URL}/files/caltech-101.zip",
        "tags": [
            category,
            "license-caltech101",
            "label-source-caltech101",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--per-category",
        type=int,
        default=2,
        help="Number of images to import per category (default 2).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated category list to import (default: SELECTED_CATEGORIES).",
    )
    parser.add_argument(
        "--assets-dir",
        default="examples/data/chapter11_assets",
        help="Path to the assets directory (default: examples/data/chapter11_assets).",
    )
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).resolve()
    manifest_path = assets_dir / "asset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    selected = (
        [c.strip() for c in args.only.split(",") if c.strip()]
        if args.only
        else list(SELECTED_CATEGORIES)
    )

    existing_ids = {r["id"] for r in payload["records"]}
    new_records: list[dict] = []
    for category in selected:
        paths = _caltech_image_paths(category, args.per_category)
        for i, src in enumerate(paths, start=1):
            rec = _new_record(src, category, i, assets_dir)
            if rec["id"] in existing_ids:
                print(f"  skip (already in manifest): {rec['id']}")
                continue
            new_records.append(rec)
            existing_ids.add(rec["id"])
            print(f"  + {rec['id']} <- {src.name}")

    if not new_records:
        print("No new records; manifest unchanged.")
        return

    payload["records"].extend(new_records)
    payload["total"] = len(payload["records"])
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"\nImported {len(new_records)} Caltech-101 images into "
        f"{manifest_path} (total records: {payload['total']})."
    )


if __name__ == "__main__":
    main()
