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

`eval_cases_chapter11_v{1,2}.json` are opt-in regression case files (the project's internal chapter11 baseline). The default `mmrag eval` does **not** use them — it loads the small generic sample shipped with the package. Load one explicitly with `--cases`; see [`eval_cases_README.md`](eval_cases_README.md) for the schema and the corpus requirement.
