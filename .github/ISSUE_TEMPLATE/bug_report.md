---
name: Bug report
about: Report something that doesn't work as expected
title: "[bug] "
labels: bug
assignees: ""
---

## What happened

<!-- A clear, concise description of the bug. -->

## Reproduction

```python
# Minimal code or command that triggers the bug
```

```bash
# Exact command(s) you ran
```

## Expected behavior

<!-- What you expected to happen. -->

## Actual behavior

<!-- What actually happened. Include any traceback. -->

```text
Paste full traceback here
```

## Environment

- OS:
- Python version: `python --version`
- mm-asset-rag version: `pip show mm-asset-rag`
- Backend: `qdrant` (local file mode by default; Qdrant server if `QDRANT_URL` is set)
- Parser(s) used: `pymupdf` / `paddleocr_vl` (per `PDF_PARSER`)
- Image embedding: `lite` / `sentence-transformers` (per `IMAGE_PROVIDER`)
- How you ran the server: `mmrag-api` / `uvicorn` / `python -m mm_asset_rag.api --host ... --port ...`
- Which endpoint you hit (`/upload/preview`, `/upload/confirm`, `/chat/stream`, `/search`, etc.) and the exact request body / curl command

## Additional context

<!-- Anything else relevant: sample documents, upload preview response, .env (REDACTED of any API keys!), logs from `$MM_ASSET_RAG_HOME/tasks.jsonl`, etc. -->

> Never paste real API keys, tokens, or other secrets. The maintainers will not need them to reproduce a bug — `paddleocr_vl` and most embedding providers accept a stub token in dev.