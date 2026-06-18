# Sample data

This directory contains the sample assets used by the LlamaIndex multimodal
RAG tutorial chapters (11–13) — 10 PDFs + 23 images + an `asset_manifest.json`
describing 30 records. About **67 MB** total.

## Origin

Extracted from `lgy1027/ai-tutorial` (`llamaindex/data/media_project/chapter11_assets/`).
Original sources include arXiv PDFs (RAG, CLIP, BERT, LLaMA, …) and a mix of
diagrams and dataset samples. See `asset_manifest.json` `source_url` fields
for the original URLs.

## Layout

```
examples/data/
└── chapter11_assets/
    ├── asset_manifest.json   # 30 records, each with id / type / path / tags
    ├── pdfs/                 # 10 PDF documents
    └── images/               # 23 PNG / JPG images
```

## Usage

Point the runtime at this directory with:

```bash
export MM_ASSET_RAG_HOME=/path/to/mm-asset-rag
cp examples/data/chapter11_assets/asset_manifest.json $MM_ASSET_RAG_HOME/assets/asset_manifest.json
mkdir -p $MM_ASSET_RAG_HOME/assets/pdfs $MM_ASSET_RAG_HOME/assets/images
cp examples/data/chapter11_assets/pdfs/*   $MM_ASSET_RAG_HOME/assets/pdfs/
cp examples/data/chapter11_assets/images/* $MM_ASSET_RAG_HOME/assets/images/
mmrag parse --pdf-parser pymupdf
mmrag index
mmrag search "which document covers retrieval-augmented generation?"
```

Or in a single `MM_ASSET_RAG_HOME` directly without copying (preferred):

```bash
mkdir -p $MM_ASSET_RAG_HOME
ln -s "$(pwd)/examples/data/chapter11_assets" $MM_ASSET_RAG_HOME/assets
mmrag parse
mmrag index
mmrag eval
```

The third `EVAL_CASES` in `mm_asset_rag/evaluation.py` (`pdf_rag`, `pdf_clip`,
`pdf_layoutlm`) target specific asset IDs in this manifest, so `mmrag eval`
will only return useful hit/miss signals when this full set is in use.