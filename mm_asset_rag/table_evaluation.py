"""Build row-level v2 evaluation cases from question-and-answer CSV files."""

from __future__ import annotations

import csv
from pathlib import Path


def build_csv_cases(
    source: str | Path,
    *,
    document_id: str,
    question_column: str,
    evidence_column: str,
) -> dict[str, object]:
    """Create document qrels plus row evidence checks for a CSV Q&A table."""
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    path = Path(source).expanduser()
    with path.open(encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        headers = reader.fieldnames or []
        missing = [name for name in (question_column, evidence_column) if name not in headers]
        if missing:
            raise ValueError(f"CSV column(s) not found: {', '.join(missing)}")
        rows = list(reader)

    cases: list[dict[str, object]] = []
    qrels: dict[str, dict[str, int]] = {}
    for row_index, row in enumerate(rows, start=1):
        query = str(row.get(question_column) or "").strip()
        evidence = str(row.get(evidence_column) or "").strip()
        if not query or not evidence:
            continue
        query_id = f"{document_id}-row-{row_index:04d}"
        cases.append(
            {
                "query_id": query_id,
                "query": query,
                "evidence_contains": [evidence],
            }
        )
        qrels[query_id] = {document_id: 1}
    if not cases:
        raise ValueError("CSV has no non-empty question and evidence rows")
    return {"version": "v2", "groups": {"table": cases}, "qrels": qrels}
