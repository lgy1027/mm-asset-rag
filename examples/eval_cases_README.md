# Legacy eval case sets

These JSON files are **archived legacy fixtures** and are not loaded by the
default `mmrag eval`. They use the removed `expected_asset_ids` schema, so the
qrels-only v2 evaluator intentionally rejects them. Keep them only for
historical comparison with the chapter11 reports.

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
`{}`. See the working bundled samples in `mm_asset_rag/eval_data/`.
