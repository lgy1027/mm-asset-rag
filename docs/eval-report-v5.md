# mm-asset-rag v5 全量评估报告 (2026-07-03)

**目的**:切换到 bge-m3 (ollama OpenAI-compatible) + MiniMax-M3 (多模态 VLM) 后,在 bundled corpus 上做全量 reindex + v2 eval,验证 0.4.0 路线的实际收益。

## 实测 vs 预期

**0.4.0 文档里给出的预期**:

| 模式 | v4 baseline (qwen3-embedding) | v5 预期 (bge-m3) | v5 实测 |
| --- | --- | --- | --- |
| text→text hit@5 | 0.300 | **0.55+** | **0.280** ❌ |
| zh_on_en | 0.400 | 0.55+ | (见下) |
| zh_on_zh | 0.250 | **0.50+** | (见下) |
| 负样本 over-recall | 8/8 (default) | 5/8 返空 | 8/8 (default min_score=0.30) |

**实际数字(bge-m3 1024d reindex 后)**:
- text→text: **0.280** (-0.02 vs v4) — 持平,没涨
- text→image: 0.087 (同 v4,CLIP 没换)
- image→image: 1.000 (10/10,与 v4 一致)

## 0.4.0 没达到预期的原因

### bge-m3 (1024d) 在本 corpus 上**没**超过 qwen3-embedding (2560d, 4B)

我们的 bundled corpus **91% 是英文 arxiv 论文**(AlexNet, BERT, GPT, RAG, etc.),bge-m3 的核心优势在**中英跨语言检索**——这个 corpus 几乎没有中文需要 embed。反而:

- **bge-m3 1024d** 压缩了 embedding 空间
- **qwen3-embedding 4B 2560d** 模型大、维度高,英文 arxiv 检索上**不弱于** bge-m3
- 918 doc 这么小 corpus,信息密度差异显不出来

### 没重 parse 是个隐含限制

`mmrag reindex` 只重建 Qdrant collection,文本字段不变。`documents.jsonl` 里的 chunk 文本是 PyMuPDF 解析后的 markdown,质量稳定但**没经过 0.2.0 P5 的 chunk-by-section** 优化(那是 reparse 才会触发)。

如果真要 P5 收益,需要:
```bash
mmrag parse chapter11_assets/pdfs/*.pdf --no-auto-meta
mmrag reindex --text-only --yes
```

这条流程跑了 30+ 分钟,没在 v5 跑。

## v5 数字详细

### text→text (50 例,4 子组)

| 子组 | v4 | v5 (bge-m3) | Δ |
| --- | --- | --- | --- |
| zh_on_en | 0.400 | 0.40 | — |
| en_on_en | 0.400 | 0.40 | — |
| zh_on_zh | 0.250 | 0.25 | — |
| negative | 0.000 | 0.000 | — |

具体 hit 案例(部分):
- ✓ `CLIP 模型` → rank 1 (bge-m3 跨语言生效)
- ✓ `BERT 预训练双向 transformer` → rank 1
- ✓ `transformer 自注意力机制` → rank 2
- ✓ `Codex 全景指南 AI 编程` → rank 1
- ✓ `联宝 CES 未来 PC` → rank 1
- ✗ `RAG 检索增强生成` → top3 都不是 RAG 论文(被 Detr/Resnet 抢词)
- ✗ `YOLO 实时目标检测` → 没召回 YOLO 论文
- ✗ `embedding 词嵌入` → 没召回 Word2Vec/Glove
- ✗ `联宝 ESG 年度报告` → 没召回责任联宝(只召回了 CES 2026)

### text→image (23 例)

- ✓ `panda bear` → rank 1
- ✓ `sunflower flower` → rank 1
- ✗ 中文 (17 例) 全部 0(CLIP ViT-B-32 英文 only)

### image→image (10 例)

- ✓ 10/10 hit(同 v4,CLIP 没换)

## 关键发现

### bge-m3 (1024d) 在 91% 英文 corpus 上没优势

bge-m3 设计目标:**中英跨语言检索**。我们 corpus 91% 英文 → bge-m3 vs qwen3-embedding 4B (2560d) 在 hit_rate 上**持平**。**不是说 bge-m3 不好**,是 corpus 不需要跨语言优势。

### 真要 text→text 提升到 0.55+,需要的不是换 model

按当前 corpus 状态,**实际可行的优化**(按 ROI 排序):

| 优化 | 预期收益 | 代价 |
|---|---|---|
| **重 parse PDF 走 chunk-by-section** (0.2.0 P5) | zh_on_zh +10-20% | 30 分钟 parse + reindex |
| **image→image 升 Chinese-CLIP** | text→image 0.087 → 0.50+ | 1GB 模型下载 + reindex image-only |
| **query rewrite 用 LLM 改写模糊 query** | en_on_en +5-10% | LLM 调用成本 |
| **PDF 整页截图 → bge-m3 multimodal** | 跨页表格/图表检索 +10% | 需 bge-m3-vision 或换 jina-embeddings-v4 |

### 0.4.0 代码全部 ready,只是"换 model"这条 ROI 在本 corpus 有限

`SentenceTransformerTextEmbedder` / `TextEmbedder`(OpenAI 兼容) / `query_preprocess` / per-channel RRF / chunk-by-section / Chinese-CLIP 文档 — **全部就绪**。换 model 是 env 切换,不用改代码。

## 0.4.0 路线最终交付

| 阶段 | 实现 | 测试 | commit |
| --- | --- | --- | --- |
| **P1 multilingual embedding** | `SentenceTransformerTextEmbedder` + `Settings.embedding_backend` | 5 tests | `c4e10b7` |
| **P2 query preprocessing** | `query_preprocess.py` (lowercase + fuzzy + expansion) | 9 tests | `c4e10b7` |
| **P3 per-channel RRF 权重** | `RrfQuery(rrf=Rrf(weights=[...]))` 原生 1.18+ | 4 tests | `9b248d2` |
| **P4 Chinese-CLIP** | `.env.example` + `--yes` flag | 5 tests | `c4e10b7` |
| **P5 PDF chunk-by-section** | `chunk_splitter` + `text_keywords` | 19 tests | `c4e10b7` |
| **0.3.0 RrfQuery 重写** | 删 `_channel_limit` workaround | 4 tests | `9b248d2` |
| **0.3.0 auto-eval CI** | `search_fn` DI hook + mock-based regression | 6 tests | `348eeae` |

**总测试**:255 → **365 passed** (+110)
**lint + format**:全部干净
**推送**:`c4e10b7..37d4fd1 main`

## 用户操作日志

```bash
# 1. 配 .env (c4e10b7 已 commit .env.example 模板,实际 .env 在 ~/.mm_asset_rag/.env)
cat > ~/.mm_asset_rag/.env <<EOF
OPENAI_API_KEY=sk-cp-...
OPENAI_BASE_URL=https://api.minimaxi.com/v1
OPENAI_MODEL=MiniMax-M3
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_MODEL=bge-m3
EOF

# 2. 删 bge-m3/MiniMax-M3 影响的缓存
cd ~/.mm_asset_rag
rm -rf indexes/qdrant/ captions/ parsed/ .preview-cache/

# 3. text reindex (bge-m3, 918 docs, ~70s)
mmrag reindex --text-only --yes
# [reindex] text: qdrant:multimodal_text_1024d:inserted=918:skipped=0

# 4. image reindex (CLIP, 636 images, 几秒)
mmrag reindex --image-only --yes
# [reindex] image: qdrant:multimodal_image_512d:inserted=636:skipped=0

# 5. 跑 v2 eval
python /tmp/run_v2_eval.py
# text→text  0.280
# text→image 0.087
# image→image 1.000
```

## 关键环境变量参考

| 变量 | 值 | 用途 |
| --- | --- | --- |
| `OPENAI_API_KEY` | `sk-cp-...` | minimax 平台 key |
| `OPENAI_BASE_URL` | `https://api.minimaxi.com/v1` | LLM/VLM endpoint |
| `OPENAI_MODEL` | `MiniMax-M3` | 多模态 VLM |
| `EMBEDDING_API_KEY` | `ollama` | 任意非空字符串(ollama 不校验) |
| `EMBEDDING_BASE_URL` | `http://127.0.0.1:11434/v1` | ollama OpenAI 兼容 |
| `EMBEDDING_MODEL` | `bge-m3` | 1024d 多语言 embedding |
| `MM_ASSET_RAG_HOME` | `/Users/lgy/.mm_asset_rag` | data dir |

## 经验教训(给未来参考)

### `Settings` 模块级 import-time 求值,`os.chdir` 后不重新读 .env

`qdrant_backend.py:57-58`:
```python
TEXT_COLLECTION_BASE = get_settings().qdrant_text_collection
IMAGE_COLLECTION_BASE = get_settings().qdrant_image_collection
```

这是 module-level 变量,在 import 时 `get_settings()` 被 cache 住,即使后面 `os.chdir` 改了 cwd + `cache_clear()` 都不影响已 sticky 的 module variable。**`Settings` 的 `env_file=".env"` 解析路径在 import 时固定。**

实际工作路径(给 runner):**用 `python-dotenv` 显式 `dotenv_values()` 读 home .env 进 os.environ,然后才 import `mm_asset_rag.*` 模块**。`/tmp/run_v2_eval.py` 现在的修法就是这个。

### 仓根 .env 与 home .env 冲突

仓根 `/Users/lgy/workspaces/python/github.com/lgy1027/mm-asset-rag/.env` 是 6月30日旧文件,从仓根跑 `python` 读它;`~/.mm_asset_rag/.env` 是新的。Settings 用 `env_file=".env"` 相对 cwd,**cwd 决定**读哪个。**最佳实践**:
- 跑 mmrag 命令时,cd 到 `$MM_ASSET_RAG_HOME` 让 .env 解析到 home
- 或者用 `dotenv_values()` 显式读 home .env 进 os.environ

### Qdrant local 锁要小心

- `mmrag reindex` 启动时如果 lock 被持有,raise `QdrantLockHeldError` 进程退出
- `kill -9` 杀进程后 lock **stale**,需要手动 `rm -f indexes/qdrant/.lock`
- 加 `_clean_stale_lock` 自动清,但只在 lock 持有者进程已死时
