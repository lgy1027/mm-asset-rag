# mm-asset-rag(中文)

> 多模态知识库 — 统一索引文档与图片，自动选择资料检索、图片检索或以图搜图；回答附带证据与文档内关联图片。

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](.github/workflows/test.yml)
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
                  │          向量后端（内置 Qdrant）                 │
                  │  multimodal_text_<dim>d    multimodal_image_<dim>d│
                  │   dense · bm25 · bm25_zh      CLIP / CN-CLIP     │
                  └───────────────┬─────────────────────────────────┘
                                  │ 查询（自动 / 资料 / 图片）
                                  ▼
                  ┌─────────────────────────────────────────────────┐
                  │  RRF 融合 → 可选 rerank → /answer 或 /chat      │
                  └─────────────────────────────────────────────────┘
```

前端只提供三种面向用户的选择，系统再自动选择相应的内部检索路径：

```
  查询 ──┬─ 资料                 ──▶ 文档证据检索
         ├─ 图片                 ──▶ 图像感知检索
         ├─ 上传查询图片          ──▶ 以图搜图
         └─ 自动（默认）          ──▶ 选择并融合相关证据
```

## 这是什么?

一个小而自洽的 Python 包,做**多模态检索**:PDF、Office 文档(docx/pptx/xlsx)、图片。检索是核心,生成是可选层。支持:

- **意图检索** — 用户只需选择“自动 / 资料 / 图片”；自动模式会按问题同时利用文字证据与图片元数据，上传图片时自动以图搜图。
- **跨模态检索** — PDF / Office 文档里嵌的图会被抽出来,可选让 VLM 打 caption,这样纯文本 query 也能命中"只有图的 slide";`find images similar to this one` 这种 query 走 CLIP image collection。同一份素材库同时喂两条线。
- **Upload-first 摄入** — 不再需要 `asset_manifest.json`。`/upload/preview` 嗅探文件魔数,提取维度 / PDF 元数据,可调用 VLM 取 title / description / tags,`/upload/confirm` 才真正解析 + 索引。
- **解析** — PDF:PyMuPDF(本地,默认)或 PaddleOCR-VL(API,扫描件更准)或 docling(本地,版面感知);Office 文档:MarkItDown(默认)或 docling;图片:OCR + VLM caption。
- **索引** — Qdrant 是内置后端（本地文件或 server）。可通过 `VECTOR_BACKEND` 选择其他已注册后端；Qdrant 文本点携带 dense、BM25 和 BM25-zh 向量，图片点携带 CLIP 向量。
- **可选生成** — OpenAI 兼容 chat completion,严格基于证据,支持 NDJSON 流式;没配 LLM 时 `/answer` / `/chat` 返回 evidence 摘要而不是报错 — 检索本身永远能工作。
- **Web UI** — 自带单页 HTML(`mm_asset_rag/api/web/index.html`),FastAPI 直接 serve,做上传预览、任务状态、聊天。

VLM 自动打 tag 也是可选的;不上 VLM 时只用 sniff 出的元数据,上传照样能跑。

## 为什么要做这个?

如果你手里有一堆混合素材 — 论文、slide deck、照片、示意图 — 想问"找出像这张的照片","哪个文档讲 RAG","给我那个只有路线图的 slide",这个项目提供面向意图的检索工作流，各层都能独立验证。

这是一个**模块化的多模态知识库**，检索和问答层都可独立验证。

对比几个更重的框架:

- **vs LlamaIndex / Verba**:自带 Web UI;**多模态检索优先**(不是文本 RAG 优先);每个模块都 ≤ 2k 行,从头读到尾容易。
- **vs Haystack / txtai**:表面积更小；提供面向文档与图片的检索；端到端可读。

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
mmrag search "哪篇文档讲 RAG?" --collection default --principal local-user
mmrag answer "哪篇文档讲 RAG?" --collection default --principal local-user --min-confidence 0.5
```

CLI 也走 upload-first(PDF / 图片 / Office 文档 / 表格都支持，含 csv / tsv):

```bash
mmrag parse ./paper.pdf ./photo.jpg ./deck.pptx --collection default --principal local-user
mmrag reindex
mmrag search "找到那张海滩照片" --collection default --principal local-user
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
  └─ 通过当前后端索引文本 chunk 和图片向量（默认 Qdrant）
```

## 配置

所有配置走环境变量（当前目录下的 `.env` 自动加载）。先配置模型能力和 RAG 档位；底层调参统一收进高级参考。

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `MM_ASSET_RAG_HOME` | 上传素材、parsed data、索引、任务历史放哪 | `~/.mm_asset_rag` |
| `MODEL_API_KEY` / `MODEL_BASE_URL` | LLM、VLM、文本 embedding 共用的 OpenAI 兼容连接 | — |
| `EMBEDDING_MODEL` / `EMBEDDING_*` | 必填的文本 embedding 模型及可选 provider 覆盖 | — |
| `LLM_MODEL` | `/answer`、`/chat` 和查询改写的可选 LLM | — |
| `VLM_MODEL` / `VLM_*` | 上传元数据和图片 caption 的可选 VLM | — |
| `RERANKER_*` | 可选的二阶段重排 provider 与模型 | 关闭 |
| `INGESTION_PROFILE` | `fast`、`balanced`、`precision` 三档摄入成本/质量策略 | `balanced` |
| `RETRIEVAL_PROFILE` | `fast`、`balanced`、`precision` 三档检索策略 | `balanced` |
| `VECTOR_BACKEND` | 运行时选用的已注册检索/索引后端 | `qdrant` |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server 模式(不填走本地文件) | — |
| `CLIP_MODEL` | sentence-transformers CLIP 模型名(配 `[clip]` extra) | `clip-ViT-B-32` |
| `IMAGE_PROVIDER` | `clip` / `cn_clip` | `clip` |
| `OCR_BACKEND` | 图片 OCR:`local`(PP-OCRv6,`[ocr]` extra)或 `http` | `local` |

档位只会补足未显式设置的高级变量，已有 `.env` 中的显式值保持原行为。简洁模板见 [`.env.example`](.env.example)，高级调优见 [`docs/configuration.md`](docs/configuration.md)。

## 评估

`mmrag eval` 用分组查询和逻辑文档 qrels 对照活索引评测,报告文档级 Recall、MRR、MAP 和分级 NDCG。每个 case 包含 `query_id` 和 `query`;顶层 `qrels` 把每个查询 ID 映射到 `{document_id: relevance}`。**默认** 走包内置的 qrels 小样例(`mm_asset_rag/eval/eval_data/`)。文档必须已用 qrels 中完全一致、区分大小写的 `document_id` ingest,否则正例会记为未命中。

```json
{
  "version": "v1",
  "groups": {"zh": [{"query_id": "q1", "query": "..."}]},
  "qrels": {"q1": {"document-id": 3}}
}
```

想评估自己的语料,写自己的 case 文件,传 `--cases`(或设 `EVAL_CASES_PATH`):

```bash
# 1. 先 ingest 你的评估语料,document_id 要与 qrels 完全一致
mmrag parse ./my_eval_corpus/*.pdf --collection default --principal local-user
# 2. 跑评估
mmrag eval --collection default --principal local-user                              # 默认内置样例
mmrag eval --cases my_cases.json --collection default --principal local-user        # 自定义
mmrag eval --v2 --collection default --principal local-user                         # v2:多维度,中文为主
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
├── mm_asset_rag/         # 一个 Python 包,按职责分包
│   ├── cli.py            # `mmrag` / `mmrag-api` 入口脚本
│   ├── service.py        # IngestService 门面:parse / index / 任务历史
│   ├── core/             # 契约 + 基础设施:settings, schema, protocols,
│   │                     #   registry, paths, llm_transport, observability
│   ├── ingest/           # 上传 → 解析:upload_pipeline, sniff, auto_meta,
│   │                     #   document_store, ingest_workflow, task_store
│   ├── query/            # 检索:search_service, retrieval, query_rewrite,
│   │                     #   query_intent, query_preprocess, evidence_policy
│   ├── answer/           # 基于证据的回答生成 + 回答质量评估
│   ├── eval/             # eval  harness + 内置 eval_data/
│   ├── api/              # FastAPI 薄路由层 + 自带 web UI
│   ├── parsers/          # PDF / image 解析实现
│   ├── embedders/        # text / image embedding 实现
│   └── backends/         # 后端适配器（内置 Qdrant）
├── tests/unit/           # 离线单元测试
├── tests/integration/    # 标记 @pytest.mark.integration
├── docs/                 # architecture、configuration、api、quickstart
└── scripts/              # benchmark.py(性能)
```

### 加新模态(audio、video)

1. 实现并注册满足 `protocols.Parser` 的解析器。
2. 实现并注册满足 `protocols.Embedder` 的嵌入器。
3. 为新 source type 增加 API/CLI 路由。
4. 扩展当前后端，使其能索引和查询该模态。

Registry 消除了中心化实现查找；路由和后端能力仍需显式实现。

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

GNU Affero General Public License v3.0 或更高版本（AGPL-3.0-or-later）。见
[LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
