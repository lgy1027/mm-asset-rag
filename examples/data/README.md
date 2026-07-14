# Sample data

This directory holds optional sample assets for manually exercising the
upload-first pipeline. **The binary assets themselves are not tracked in
git** — they bloat every clone and the project is upload-first (users
bring their own files). Only this README is committed; restore the
samples locally if you need them (see below).

## What used to live here

A `chapter11_assets/` subset (10 PDFs + Caltech-101 image labels, ~170 MB)
was tracked in earlier history. It has been removed from the index so
`.gitignore`'s `/examples/data/` rule takes effect. The files may still
exist on disk if you pulled them before the cleanup; they are purely
local now.

## Restore / add your own samples

The project no longer ships a default corpus. To run `mmrag eval` or the
`examples/api_client.py` script against real data, upload your own files
through the web UI or CLI:

```bash
# CLI equivalent of the web upload flow: preview → confirm → parse + index
mmrag parse ./your.pdf ./your.jpg

mmrag search "your query"
mmrag answer "your question"
mmrag eval
```

The `EVAL_CASES` in `mm_asset_rag/evaluation.py` target specific asset
titles/ids, so `mmrag eval` returns useful hit/miss signals only when a
matching corpus is indexed. Treat the bundled eval cases as a template
to adapt to your own corpus, not as a fixed benchmark.

## Layout (when populated locally)

```
examples/data/
└── chapter11_assets/     # local-only, git-ignored
    ├── pdfs/             # sample PDF documents
    └── images/           # sample PNG / JPG images
```
