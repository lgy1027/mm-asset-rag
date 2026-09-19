# Eval case sets

`eval_cases_images_v2.json` is the **current** image qrels sample: generated
from `image_eval_manifest_v1.json`, it backs `mmrag eval --image` (the default
when `--cases` is omitted) and scores the primitive `TEXT_TO_IMAGE` /
`IMAGE_TO_IMAGE` routes. See `examples/image_eval_manifest_v1.json` for the
manifest that is the semantic source of truth for the corpus and queries.

The `eval_cases_chapter11_*.json` files are **archived legacy fixtures** and
are not loaded by the default `mmrag eval`. They use the removed
`expected_asset_ids` schema, so the qrels-only v2 evaluator intentionally
rejects them. Keep them only for historical comparison with the chapter11
reports.

Do not use these files as templates. New case files must assign every case a
unique `query_id` and provide a top-level graded `qrels` mapping.

## Current schema

```json
{
  "version": "v1",
  "groups": {
    "en": [{"query_id": "q1", "query": "..."}]
  },
  "qrels": {"q1": {"exact-document-id": 3}}
}
```

Document IDs are matched exactly and case-sensitively. Positive integer grades
mean relevant; a negative case must still have a qrels entry whose value is
`{}`. See the working bundled samples in `mm_asset_rag/eval/eval_data/`.
