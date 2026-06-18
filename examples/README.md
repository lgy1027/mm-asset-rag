# Examples

## `api_client.py`

A minimal end-to-end client that exercises every HTTP endpoint.

```bash
# Terminal 1
mmrag-api

# Terminal 2 (with assets prepared in $MM_ASSET_RAG_HOME/assets)
python examples/api_client.py
```

Requires the server to be running locally and `MM_ASSET_RAG_HOME/assets/asset_manifest.json` to point to actual files. The `image-to-image` block in the script is commented out — uncomment and supply a real `image_path` to try it.