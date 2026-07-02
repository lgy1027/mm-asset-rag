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