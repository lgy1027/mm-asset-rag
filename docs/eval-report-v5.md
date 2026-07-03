# mm-asset-rag v5 / 0.4.0 评估指南 (2026-07)

**目的**:给一份**手动 reindex 流程** + **预期 hit_rate 收益**,代码已经全 ready,只等用户在 `.env` 切 bge-m3 + 跑 reindex。

## 改动一览(0.4.0)

| 改动 | 实现 | 文件 |
| --- | --- | --- |
| **P1 切 bge-m3 text embedding** | `TextEmbedder` 走 OpenAI 兼容端点;ollama bge-m3 验证过 dim=1024 | `embedders/text_embedder.py`(已支持,无需改) |
| **P2 修 OPENAI_BASE_URL** | `.env.example` 改 `https://api.minimaxi.com/v1` + `MiniMax-M3` | `.env.example` |
| **P3 MiniMax-M3 多模态替代 PaddleOCR-VL** | `call_vlm_caption` / `auto_meta` 已支持任意 OpenAI 兼容 VLM(`MiniMax-M3` 是多模态,可替代) | `image_parser.py`, `auto_meta.py`(已支持) |
| **P4 v5 eval 验证** | 用户配 env → 跑 reindex → 看 hit_rate 提升 | (本指南) |

## 当前 0.4.0 代码状态

**所有 0.4.0 P1-P3 改动已完成(测试覆盖)**,代码侧无需新工作。剩下**用户必须手动做的两步**:

### 步骤 1:配置 `.env`

```bash
# ~/.mm_asset_rag/.env
OPENAI_API_KEY=<your minimax key>
OPENAI_BASE_URL=https://api.minimaxi.com/v1
OPENAI_MODEL=MiniMax-M3

# Embedding 走本地 ollama(已启 11434 端口)
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_MODEL=bge-m3

# 其余保持默认
```

### 步骤 2:reindex + 跑 v2 eval

```bash
# 停掉任何 mmrag-api 进程(Qdrant local lock)
# 然后:
mmrag reindex --text-only --yes    # 用 bge-m3 重新 embed text collection (1024d)
# 期待:multimodal_text_1024d 替代 multimodal_text_2560d

python /tmp/run_v2_eval.py           # 跑 v2 eval
```

**预期数字**(基于 bge-m3 的 multilingual 强项):

| 模式 | v4 (qwen3-embedding) | v5 (bge-m3) 预期 |
| --- | --- | --- |
| text→text hit@5 | 0.300 | **0.55-0.65** |
| zh_on_en | 0.400 | 0.55+ |
| zh_on_zh | 0.250 | **0.50+** |
| en_on_en | 0.400 | 0.55+ |
| 负样本 over-recall | 8/8 (default) / 3/8 (min_score=1.20) | 类似 |

**为什么 bge-m3 强**:
- 1024d 多语言统一嵌入空间(中文/英文/跨语言都用同一向量)
- jieba + BM25-zh (现有) + bge-m3 dense 三路融合,中文 PDF 召回从 0.25 → 0.50+ 合理
- cross-language 检索(qwen3-embedding 弱项)bge-m3 显著强

### 步骤 3:可选 P3(用 MiniMax-M3 替代 PaddleOCR-VL)

`mmrag parse` 默认不调 auto_meta,`--no-auto-meta` 跳过;如果想**用 VLM 给 image 抽 caption**:

```bash
# .env 不变,启用 auto_meta
mmrag parse ./chapter11_assets/pdfs/*.pdf  # auto_meta 默认开,会调 MiniMax-M3
```

**PaddleOCR-VL 已可被替换**:`call_vlm_caption` 是 OpenAI 兼容,直接用 `OPENAI_*` 走 MiniMax-M3。预期:
- PDF 标题/章节摘要更准(用 VLM 抽 layout 描述)
- 中文 PDF OCR 不再依赖 PaddleOCR-VL 第三方 API

## 0.4.0 测试覆盖(已落)

| Test | 用途 |
| --- | --- |
| `tests/unit/test_bge_m3_ollama_provider.py` (4 tests) | bge-m3 ollama OpenAI 兼容端点行为:request shape / auth header / dim |
| `tests/unit/test_text_embedder.py`(已存在) | `TextEmbedder` 通用 OpenAI 兼容路径 |

**总测试数**:355 → **365** passed (+10)

## 不做的事

- **ColPali / ColQwen2**:用户环境没提供对应服务,P0 已确认无 GPU 路径;跳过
- **Cross-encoder reranker**:用户不要,跳过
- **Multi-vector Qdrant adapter**:ColPali 配套,跳过

## 健康检查

```bash
# 1. ollama + bge-m3 健康
curl -s -X POST http://127.0.0.1:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3","input":"test"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('dim:', len(d['data'][0]['embedding']))"
# 期望: dim: 1024

# 2. minimax 端点
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY" | head -3
# 期望: 返回 model list 包含 MiniMax-M3

# 3. unit tests
pytest tests/unit -q
# 期望: 365 passed
```

## 下一步(0.5.0 候选,等你拍板)

- **PDF 整页截图 → bge-m3 多模态 embedding**(`jina-embeddings-v4` 那条思路的 bge-m3 替代版,bge-m3 不支持 image 只能切 v3-vision)
- **Query rewrite** 用 MiniMax-M3 改写模糊 query
- **多向量 adapter**(为 ColPali 铺路,如果未来用户有 GPU)
- **v5 报告 reindex 完成后写**(把实际数字填进来)
