# Opt-in eval case sets

These JSON files are **not** loaded by the default `mmrag eval`. They ship the
project's internal chapter11 regression baseline (80+ cases across the
联宝 / Codex / Caltech-101 / arxiv-paper corpus) for two purposes:

1. **Reproducibility** — anyone who wants to compare against the numbers in
   `docs/eval-report.md` can load the exact case set that produced them.
2. **Template** — a larger, real-world example of how a case file is shaped,
   to copy when authoring your own.

## Running them

```bash
mmrag eval --cases examples/eval_cases_chapter11_v1.json
mmrag eval --v2 --cases examples/eval_cases_chapter11_v2.json
```

## You must bring your own corpus

The cases reference asset titles/ids for documents that are **not** in this
repo (`examples/data/chapter11_assets/` is intentionally gitignored — the
project is upload-first). Without those assets ingested, every case returns
`hit: false`. Ingest matching files first:

```bash
mmrag parse ./your-papers...
mmrag eval --cases examples/eval_cases_chapter11_v1.json
```

## Schema

```json
{
  "version": "v1",
  "groups": {
    "en": [{"query": "...", "expected_asset_ids": ["..."]}],
    "zh": [...],
    "zh_doc": [...],
    "text_to_image": [{"query": "...", "expected_asset_ids": ["..."]}],
    "image_to_image": [{"image_path": "...", "expected_asset_ids": ["..."]}]
  }
}
```

`expected_asset_ids` are prefix-tolerant — a bare title matches any
`<title>_<hash>` variant the index returns. `negative` cases carry an empty
`expected_asset_ids` list. The default minimal set shipped with the package
(`mm_asset_rag/eval_data/`) covers only text→text; these opt-in files add the
image routes and the full internal corpus.
