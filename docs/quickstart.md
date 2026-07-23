# 快速上手(Quickstart)

> 目标:**30 分钟内**从零到第一次 `mmrag search` 出结果。
> 这条路径只用最少的依赖:本地 ollama(bge-m3 embedding)+ Qdrant 本地文件(自动,无需起服务)+ PyMuPDF(纯文本 PDF)。
> reranker / CLIP 图像 / docling / OCR / VLM 都是**可选**的,先跑通再按需开。

## 前置条件

| 依赖 | 说明 |
| --- | --- |
| Python 3.10 / 3.11 / 3.12 | |
| **ollama** | 本地 embedding + 可选 LLM。安装:`curl -fsSL https://ollama.com/install.sh \| sh` |
| **Qdrant** | **不用单独装**。本仓库默认用 Qdrant 的 local-file 模式,数据写在 `~/.mm_asset_rag/indexes/qdrant/`,进程内嵌,无需起服务。 |
| 一两个 PDF 文件 | 用来试跑。随便找篇论文即可。 |

不需要的(可选,新手先跳过):sentence-transformers(CLP 图像)、docling(复杂版面)、PaddleOCR-VL(扫描件 OCR)、云 LLM key。

## 第 1 步:装包

```bash
git clone <repo> mm-asset-rag
cd mm-asset-rag
pip install -e .
```

> 如果要用 uv(推荐,本仓库用 `.venv`):`uv pip install -e .`
> 验证:`mmrag --help` 能打印子命令列表即成功。

## 第 2 步:用 ollama 拉 embedding 模型

bge-m3 是默认的文本向量模型,支持多语言 + 自带稀疏向量(BM25 那一路)。

```bash
ollama pull bge-m3          # ~1.2GB,一次即可
ollama serve                # 启动本地服务(已运行则跳过)
```

可选:想要 LLM 回答(不配也行,`mmrag answer` 会回退成 evidence 摘要):

```bash
ollama pull gemma3:4b       # 或任何 OpenAI 兼容的聊天模型
```

## 第 3 步:写最小 `.env`

在仓库根目录创建 `.env`,只填 3 行:

```bash
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_MODEL=bge-m3
EMBEDDING_API_KEY=ollama          # ollama 不校验 key,随便填即可占位
```

> 为什么这么填?embedding 走 OpenAI 兼容协议,ollama 暴露在 `127.0.0.1:11434/v1`。`EMBEDDING_API_KEY` 字段必填但值随便——ollama 不看。详见 [configuration.md](configuration.md)。

可选:接上一步的 LLM(不接则 `/answer` 返回 evidence 摘要):

```bash
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_MODEL=gemma3:4b
OPENAI_API_KEY=ollama
```

**建议新手先把 reranker 关掉**(默认是开的,但需要 `sentence-transformers`,没装时会自动降级——不过为了干净起见):

```bash
RERANKER_ENABLED=false
```

## 第 4 步:索引一个 PDF

```bash
mmrag parse ./your_paper.pdf
```

这条命令会:嗅探文件 → PyMuPDF 抽文本 + 分块 → bge-m3 向量化 → 写进 Qdrant local collection。
看到任务结束、无报错即成功。

> 进度在哪看?任务历史存在 `~/.mm_asset_rag/` 下的 SQLite 里(`mmrag retry` 可重跑失败/中断的资产)。
> 想看 web UI?`mmrag-api` 起 FastAPI 服务(`http://127.0.0.1:8011`),浏览器打开 `/` 可视化上传/查看任务。

## 第 5 步:搜索

```bash
mmrag search "retrieval augmented generation"
# 默认 mode=hybrid(text dense + BM25 + BM25-zh 三路 RRF 融合)
```

输出是 top-k 个 `SearchHit`,含 `asset_id` / `title` / `score` / `evidence`(命中的文本片段)。

其他模式:

```bash
mmrag search "your query" --mode text        # 纯文本路
mmrag search "your query" --mode hybrid --top-k 10
```

## 第 6 步:问个问题(可选)

```bash
mmrag answer "这篇论文讲了什么?"
```

配了 LLM → 返回 grounded 回答;没配 → 返回检索到的 evidence 摘要,不报错。

## 常见坑(新手最容易踩的)

| 现象 | 原因 / 解法 |
| --- | --- |
| `TextEmbedder requires api_key, base_url, and model` | `.env` 没配全 3 个 `EMBEDDING_*`,或 ollama 没起(`ollama serve`) |
| `Storage folder ... is already accessed by another instance` | Qdrant local 是**单进程锁**。停掉 API server / 别的 `mmrag` 进程再跑。换 `QDRANT_URL` 可并发。 |
| 改了 embedding 维度后搜不到东西 | collection 名按向量维度加后缀,换维度会建新空集合。跑 `mmrag reindex` 重建。 |
| 单测莫名红(本机有 `.env`) | 本机根目录 `.env` 会被单测读进去覆盖默认值。**跑测试前挪开 `.env`**:`mv .env .env._testparked` |
| `mmrag search` 慢、没重排 | reranker 默认开但没装 `sentence-transformers` 会静默降级。要开重排:`pip install -e ".[clip]"` 并设 `RERANKER_ENABLED=true` |
| 搜中文论文召不对 | 试试 `--mode text`,或开 reranker(对"近亲论文混淆"的提升最大,见 [eval-report.md](eval-report.md)) |

## 下一步

跑通后,按需解锁可选能力(都在 `pip install -e ".[extra]"`):

| 想要 | 装什么 | 开什么 |
| --- | --- | --- |
| 图像检索 / 以图搜图 | `[clip]`(sentence-transformers + CLIP) | `IMAGE_PROVIDER=sentence_transformers` |
| 更好的 reranker(本地) | `[clip]`(同上,bge-reranker-v2-m3) | `RERANKER_ENABLED=true` |
| 云 reranker(不装本地模型) | 无额外依赖 | `RERANKER_ENABLED=true` + `RERANKER_PROVIDER=siliconflow`(或 `dashscope`)+ `RERANKER_API_KEY=sk-xxx` |
| docx/pptx/xlsx 复杂版面 | `[docling]` | `DOCUMENT_PARSER=docling` |
| 扫描件 PDF OCR | PaddleOCR-VL token | `PDF_PARSER=paddleocr_vl` |
| VLM 给图片打标 | ollama vision 模型 | `ENABLE_VLM=true` |

完整配置参考:[configuration.md](configuration.md);架构总览:[architecture.md](architecture.md);上传流程:[upload-flow.md](upload-flow.md)。
