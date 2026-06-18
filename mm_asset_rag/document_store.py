import json
from pathlib import Path

from .paths import get_documents_jsonl
from .schema import ParsedDocument


def write_documents(documents: list[ParsedDocument], path: Path | None = None) -> None:
    target = path or get_documents_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        for document in documents:
            file_obj.write(json.dumps(document.to_json(), ensure_ascii=False) + "\n")


def read_documents(path: Path | None = None) -> list[ParsedDocument]:
    target = path or get_documents_jsonl()
    if not target.exists():
        raise RuntimeError(f"Document JSONL not found: {target}")
    documents = []
    with target.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            documents.append(
                ParsedDocument(
                    text=str(payload["text"]),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
    return documents
