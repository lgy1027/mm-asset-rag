# mm-asset-rag(中文)

> 多模态检索引擎 — 把混合素材(PDF / Office 文档 / 图片)统一索引,然后跨四种路由检索:text→text、text→image、image→image、weighted hybrid,全部用 RRF 融合;检索之上可选叠加基于证据的 LLM 回答层。

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-723%20passed-orange)](.github/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25-yellow)](tests/)

[English README](README.md) | 中文(本文档)

## 一眼看懂

```
                  ┌─────────────────────────────────────────────────┐
                  │          $ mmrag-api  (FastAPI + Web UI)        │
                  └───────────────┬─────────────────────────────────┘
                                  drag / POST /upload/preview
                                          ▼
   ┌──────────────────────┐   POST /upload/confirm    ┌──────────────────┐
   │  .preview-cache/<id> │ ─────────────────────────▶│  assets/pdfs     │
   │   (sniff + VLM meta) │   background task        │  assets/images   │
   └──────────────────────┘                          │  assets/documents │
                                                      └─────────┬────────┘
                                                                │ parse
                                                                ▼
                                                  ┌──────────────────────┐
                                                  │   documents.jsonl    │
                                                  └─────────┬────────────┘
                                                                │ embed
                                                                ▼
                  ┌─────────────────────────────────────────────────┐
                  │                  Qdrant (本地 / 服务)            │
                  │  multimodal_text_<dim>d    multimodal_image_<dim>d│
                  │   dense · bm25 · bm25_zh      CLIP / CN-CLIP     │
                  └───────────────┬─────────────────────────────────┘
                                  │ query (text / image / hybrid)
                                  ▼
                  ┌─────────────────────────────────────────────────┐
                  │  RRF 融合 → 可选 rerank → /answer 或 /chat      │
                  └─────────────────────────────────────────────────┘
```

四种检索路由,统一由 `mmrag search "..."` 入口按 query 形状自动分派:

```
  query ─┬─ text                 ──▶ qdrant_text_search          (dense + bm25 + bm25_zh)
         ├─ text  + image_path   ──▶ + qdrant_text_to_image_search  (CLIP text → image)
         ├─ image                ──▶ qdrant_image_to_image_search   (CLIP image → image)
         └─ hybrid (默认)        ──▶ 三路按权重合并,RRF 融合
```

## 这是什么?

一个小而自洽的 Python 包,做**多模态检索**:PDF、Office 文档(docx/pptx/xlsx)、图片。检索是核心,生成是可选层。支持:

- **四种检索路由** — text→text(dense + BM25 稀疏向量 RRF 融合)、text→image(CLIP)、image→image(CLIP),以及按权重混合三路的 hybrid。一次 dispatch 按 query 形状自动分派路由。
- **跨模态检索** — PDF / Office 文档里嵌的图会被抽出来,可选让 VLM 打 caption,这样纯文本 query 也能命中"只有图的 slide";`find images similar to this one` 这种 query 走 CLIP image collection。同一份素材库同时喂两条线。
- **Upload-first 摄入** — 不再需要 `asset_manifest.json`。`/upload/preview` 嗅探文件魔数,提取维度 / PDF 元数据,可调用 VLM 取 title / description / tags,`/upload/confirm` 才真正解析 + 索引。
- **解析** — PDF:PyMuPDF(本地,默认)或 PaddleOCR-VL(API,扫描件更准)或 docling(本地,版面感知);Office 文档:MarkItDown(默认)或 docling;图片:OCR + VLM caption。
- **索引** — Qdrant(本地文件或 server)。文本点携带 dense + BM25 + 中文 BM25-zh 三个稀疏通道;图片点带 CLIP 向量。
- **可选生成** — OpenAI 兼容 chat completion,严格基于证据,支持 NDJSON 流式;没配 LLM 时 `/answer` / `/chat` 返回 evidence 摘要而不是报错 — 检索本身永远能工作。
- **Web UI** — 自带单页 HTML(`mm_asset_rag/web/index.html`),FastAPI 直接 serve,做上传预览、任务状态、聊天。

VLM 自动打 tag 也是可选的;不上 VLM 时只用 sniff 出的元数据,上传照样能跑。

## 为什么要做这个?

如果你手里有一堆混合素材 — 论文、slide deck、照片、示意图 — 想问"找出像这张的照片","哪个文档讲 RAG","给我那个只有路线图的 slide",这个项目是个能用的起点。重心在**检索**:四种路由 + 跨模态 + rank 融合,每一层都可替换。

不是研究级系统;是个**模块化的多模态检索引擎**,把可动的地方都暴露出来 — 你想换 parser / embedder / backend / reranker / LLM,不用改其他地方。

对比几个更重的框架:

- **vs LlamaIndex / Verba**:自带 Web UI;**多模态检索优先**(不是文本 RAG 优先);每个模块都 ≤ 2k 行,从头读到尾容易。
- **vs Haystack / txtai**:表面积更小;四种路由从第一天就内置;端到端可读。

## 安装

从 PyPI 装最新版:

```bash
pip install mm-asset-rag   # core:文本检索 + FastAPI web UI(图检索需 [clip])
```

可选 CLIP 图像 embedding(要做 text→image / image→image 推荐装):

```bash
pip install "mm-asset-rag[clip]"     # sentence-transformers CLIP
```

可选中文 CLIP(中文 zero-shot,768d,中文语料强烈推荐):

```bash
pip install "mm-asset-rag[cn_clip]"  # transformers + OFA-Sys/chinese-clip-vit-base-patch16
```

可选 docx/pptx/xlsx/html 复杂版面解析(默认 MarkItDown 已够用,docling 是更准但更重的备选):

```bash
pip install "mm-asset-rag[docling]"  # docling(heavy,会拉 torch/transformers)
```

可选本地 OCR(`[ocr]` extra,纯 ONNX,零外网):

```bash
pip install "mm-asset-rag[ocr]"      # PP-OCRv6 + onnxruntime
```

可叠加,如 `[clip,ocr]` 或 `[clip,docling,ocr]`。

源码本地开发:

```bash
git clone https://github.com/lgy1027/mm-asset-rag
cd mm-asset-rag
pip install -e ".[dev,clip]"
```

或用 [uv](https://docs.astral.sh/uv/)(commit 进 `uv.lock`,reproducible):

```bash
uv sync --extra dev --extra clip
```

## 快速上手

> **第一次用?** 先看 [docs/quickstart.md](docs/quickstart.md) — 从零搭环境(ollama + bge-m3 + Qdrant 本地)到第一次 `mmrag search` 出结果的 30 分钟路径,含新手常见坑。下面假定环境已就绪。

```bash
# 1. 启 API + Web UI
mmrag-api
# → http://127.0.0.1:8011/
# → http://127.0.0.1:8011/docs

# 2. 打开 Web UI,拖 PDF / 图片,审 preview card,
#    必要时改 title / tags,点 Confirm & Ingest。

# 3. CLI 检索 / 问答(等 ingest 完成后)
mmrag search "哪篇文档讲 RAG?"
mmrag answer "哪篇文档讲 RAG?"
```

CLI 也走 upload-first(PDF / 图片 / Office 文档都支持):

```bash
mmrag parse ./paper.pdf ./photo.jpg ./deck.pptx
mmrag reindex
mmrag search "找到那张海滩照片"
```

> ⚠️ **Qdrant 本地文件锁是单进程的。** `mmrag-api` 跑着时,另一个终端跑 `mmrag reindex` 会报 `storage already accessed`。要么先停 API,要么把 `QDRANT_URL` 指到独立的 Qdrant server。

**任务控制** — 长 parse / index 任务可以协作式取消:`POST /tasks/{id}/cancel` 设 stop flag,worker 在两个 asset 之间检查(完成当前 asset,再停止并把任务标 `cancelled`)。`mmrag retry` 跑剩下的 asset。

**健康检查** — `GET /health` 返回存活 + 索引状态;`GET /health?deep=true` 再加 `llm_configured` / `embedder_configured`(只看配置齐不齐,**不**触发 LLM 调用 / 配额),让编排器能区分"`/answer` 能不能工作"。

## 上传流程

```
POST /upload/preview (multipart files)
  ├─ stream 到 .preview-cache/
  ├─ 嗅探魔数:pdf / image / unsupported
  ├─ 抽本地元数据:PDF /Info、页数、图片尺寸、EXIF
  ├─ 可选 VLM JSON mode:title / description / tags
  └─ 返回可编辑 preview cards

POST /upload/confirm (cache_id + 编辑过的 previews)
  ├─ 把确认的文件搬到 assets/pdfs、assets/images 或 assets/documents
  ├─ 解析 PDF / image / document → documents.jsonl
  ├─ 文本 chunk upsert 到 Qdrant text collection
  └─ 图片向量 upsert 到 Qdrant image collection
```

## 配置

所有配置走环境变量(当前目录下的 `.env` 自动加载)。最常用的几个:

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | 上传素材、parsed data、索引、任务历史放哪 | `~/.mm_asset_rag` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | `/answer` 和 `/chat` 用的 LLM | — |
| `EMBEDDING_*` | 文本 embedding provider(默认 OpenAI 兼容) | — |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server 模式(不填走本地文件) | — |
| `CLIP_MODEL` | sentence-transformers CLIP 模型名(配 `[clip]` extra) | `clip-ViT-B-32` |
| `IMAGE_PROVIDER` | `lite` / `sentence_transformers` / `cn_clip` | `lite` |
| `OCR_BACKEND` | 图片 OCR:`local`(PP-OCRv6,`[ocr]` extra)或 `http` | `local` |
| `OCR_HTTP_URL` | 自建 OCR 端点(只 `OCR_BACKEND=http` 时用) | — |
| `AUTO_META_ENABLED` | 上传 preview 时是否走 VLM title / description / tag | `true` |
| `PADDLEOCR_VL_API_TOKEN` | PaddleOCR-VL API token(扫描 PDF) | — |

完整列表见 [`.env.example`](.env.example) 和 [`docs/configuration.md`](docs/configuration.md)。

## 评估

`mmrag eval` 拿一组 `query → expected_asset_ids` 对照活索引跑,报 hit-rate / MRR。cases 写在 JSON 里(`{"version","groups":{group:[{query,expected_asset_ids}]}}`)。**默认** 走包内置小样例(`mm_asset_rag/eval_data/`)— 一个文本→text 模板,对着几篇经典 arxiv 论文。前提是**语料已 ingest**,否则全是 `hit: false`。

想评估自己的语料,写自己的 case 文件,传 `--cases`(或设 `EVAL_CASES_PATH`):

```bash
# 1. 先 ingest 你的评估语料(cases 引用 asset title/id,
#    把 mmrag parse 指到你要评估的那批)
mmrag parse ./my_eval_corpus/*.pdf
# 2. 跑评估
mmrag eval                              # 默认内置样例
mmrag eval --cases my_cases.json        # 自定义
mmrag eval --v2                         # v2:多维度,中文为主
mmrag eval --v2 --cases examples/eval_cases_chapter11_v2.json   # 内部基线
```

没配 LLM 也能跑(只评检索),`/answer` 相关 case 优雅降级。

### 跑性能基准

语料到一定量后,在自己机器上跑真实 p50 / p95 / QPS,再去调权重:

```bash
# 跑前先停 mmrag-api(Qdrant local 是单进程锁)
uv run python scripts/benchmark.py --top-k 5 --n-runs 50
# → 写 $MM_ASSET_RAG_HOME/benchmark_report.json + stdout 表
```

基准只走公开 `hybrid_search` — 不依赖内部 `_` 私有符号 — 调 `Settings` 改了 reranker / `MAX_CHUNKS_PER_PDF` 后数字立刻反映。完整路径见 [`docs/quickstart.md`](docs/quickstart.md)。

## 项目结构

```
mm-asset-rag/
├── mm_asset_rag/         # 一个 Python 包(扁平布局 + 三个子包)
│   ├── api.py            # FastAPI app:薄路由层,委托给 service.py
│   ├── cli.py            # `mmrag` / `mmrag-api` 入口脚本
│   ├── service.py        # IngestService:parse / index / 任务历史
│   ├── upload_pipeline.py# preview → confirm 上传流水线
│   ├── sniff.py          # 文件魔数 + 本地元数据
│   ├── auto_meta.py      # VLM JSON-mode 元数据抽取
│   ├── settings.py       # pydantic-settings:env var 集中一处
│   ├── protocols.py      # Parser / Embedder / VectorBackend 协议
│   ├── registry.py       # parser / embedder / backend 全局 registry
│   ├── paths.py          # $MM_ASSET_RAG_HOME 下磁盘布局
│   ├── assets.py         # Asset dataclass
│   ├── schema.py         # SearchHit / ParsedDocument
│   ├── document_store.py # 统一的 ParsedDocument JSONL 存储
│   ├── answer.py         # 基于证据的回答(流式 + 同步)
│   ├── evaluation.py     # 小型回归套件
│   ├── retrieval.py      # hybrid merge + normalize
│   ├── parsers/          # PDF / image 解析实现
│   ├── embedders/        # text / image embedding 实现
│   └── backends/         # Qdrant backend 实现
├── tests/unit/           # 离线单元测试
├── tests/integration/    # 标记 @pytest.mark.integration
├── docs/                 # architecture、configuration、api、quickstart
└── scripts/              # benchmark.py(性能)
```

### 加新模态(audio、video)

三行改动,不用动中央 dispatch:

1. 写 `parsers/audio_parser.py`,类满足 `protocols.Parser`。
2. 在 `parsers/__init__.py` 末尾 `register_parser(AudioParser())`。
3. 写 `embedders/audio_embedder.py`,类满足 `protocols.Embedder`,同样 `register_embedder(...)`。

FastAPI、CLI、Qdrant backend 全部从 registry 运行时读。

## 文档

- [Quickstart(从零到第一次搜索)](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Data flow(文本线 vs 图片线对照)](docs/data-flow.md)
- [Configuration](docs/configuration.md)
- [HTTP API](docs/api.md)
- [Upload flow](docs/upload-flow.md)
- [FAQ & 故障排查](docs/faq.md)

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)。

## 协议

Apache-2.0。见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。