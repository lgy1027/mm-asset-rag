"""End-to-end client example for mm-asset-rag's HTTP API.

Prerequisite: start the server in another shell:

    $ mmrag-api
    # → http://127.0.0.1:8011

This script:
  1. calls /health
  2. uploads files via /upload/preview, then /upload/confirm to parse + index
  3. calls /search in four modes
  4. calls /answer (falls back to evidence summary if no LLM configured)
  5. calls /eval

Usage:
    python examples/api_client.py path/to/file1.pdf path/to/file2.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8011"


def show(label: str, payload: dict) -> None:
    print(f"\n=== {label} ===")
    if isinstance(payload, dict) and "hits" in payload and len(payload["hits"]) > 0:
        for hit in payload["hits"][:3]:
            ev = hit.get("evidence", "")
            if len(ev) > 120:
                hit["evidence"] = ev[:120] + "..."
    print(payload)


def upload_and_confirm(session: requests.Session, files: list[str]) -> None:
    """Two-phase upload: /upload/preview then /upload/confirm.

    If no files are passed the step is skipped (the rest of the demo still
    runs against whatever is already indexed).
    """
    if not files:
        print("\n=== upload (skipped: no files passed) ===")
        return
    multipart = []
    for f in files:
        p = Path(f)
        multipart.append(("files", (p.name, p.read_bytes())))
    preview = session.post("/upload/preview", files=multipart).json()
    show("upload/preview", preview)
    edits = [
        {
            "preview_id": pv["preview_id"],
            "title": pv.get("effective_title") or pv["sniff"].get("title", ""),
            "tags": pv.get("effective_tags", []),
        }
        for pv in preview["previews"]
        if not pv.get("rejected_reason")
    ]
    if edits:
        confirm = session.post(
            "/upload/confirm",
            json={"cache_id": preview["cache_id"], "edits": edits},
        ).json()
        show("upload/confirm", confirm)


def main() -> None:
    files = sys.argv[1:]
    with requests.Session() as session:
        show("health", session.get(f"{BASE}/health").json())

        upload_and_confirm(session, files)

        show(
            "search (text)",
            session.post(
                f"{BASE}/search",
                json={"query": "retrieval augmented generation", "mode": "text", "top_k": 3},
            ).json(),
        )

        show(
            "search (text-to-image)",
            session.post(
                f"{BASE}/search",
                json={"query": "diagram", "mode": "text-to-image", "top_k": 3},
            ).json(),
        )

        # image-to-image needs a real image path; only call if you have one.
        # show("search (image-to-image)", session.post(
        #     f"{BASE}/search",
        #     json={"query": "x", "mode": "image-to-image",
        #           "image_path": "/abs/path/to/query.png", "top_k": 3},
        # ).json())

        show(
            "answer",
            session.post(
                f"{BASE}/answer",
                json={
                    "question": "which document covers retrieval-augmented generation?",
                    "top_k": 3,
                },
            ).json(),
        )

        show("eval", session.post(f"{BASE}/eval", json={}).json())


if __name__ == "__main__":
    main()
