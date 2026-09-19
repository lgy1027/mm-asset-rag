# Examples

## `api_client.py`

A minimal HTTP client for a running local server.

```bash
# Terminal 1
mmrag-api

# Terminal 2
python examples/api_client.py
```

The project is upload-first: use the web UI or `/upload/preview` + `/upload/confirm` before running search/answer examples. The `image-to-image` block in the script is commented out — uncomment and supply a real `image_path` to try it.

## Eval case sets

`eval_cases_chapter11_v{1,2}.json` are **archived legacy fixtures** (the project's internal chapter11 baseline), kept only for historical comparison with the chapter11 reports. The default `mmrag eval` does **not** use them — it loads the small generic sample shipped with the package. They use the removed `expected_asset_ids` schema, so the qrels-only v2 evaluator intentionally **rejects them — they cannot be loaded via `--cases`**. Do not use these files as templates; see [`eval_cases_README.md`](eval_cases_README.md) for the current schema and the corpus requirement.
