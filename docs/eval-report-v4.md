# mm-asset-rag v4 评估报告 (2026-07-04)

**目的**:落地 0.2.0 路线 5 阶段全部(P1 切 multilingual / P2 query 预处理 / P3 BM25 权重 / P4 Chinese-CLIP / P5 PDF chunk-by-section + 关键词)。验证 v3 → v4 进展 + 给用户 reindex 指南。

## 改动一览

| 阶段 | 实现 | 文件 | 测试 |
| --- | --- | --- | --- |
| **P1 multilingual embedding** | `Settings.embedding_backend` (`openai` / `sentence_transformers`);`SentenceTransformerTextEmbedder` 支持 BGE-m3 / multilingual-e5 | `embedders/text_embedder.py`, `settings.py` | 5 tests |
| **P2 query preprocessing** | `query_preprocess.py` 三个 stage: lowercase / fuzzy corrector / expansion pairs。`qdrant_text_search` 走预处理器,BM25 用预处理版,dense 用原版 | `query_preprocess.py`, `backends/qdrant_backend.py:783` | 9 tests |
| **P3 per-channel RRF 权重** | `Settings.rrf_weight_dense/bm25/bm25_zh`,`qdrant_client 1.18` 不支持 per-prefetch weight,改通过 `_channel_limit(weight)` scale prefetch `limit`(1.0 默认 → 20, 1.5 → 30, 4.0 封顶) | `backends/qdrant_backend.py:_channel_limit`, `settings.py` | 4 tests |
| **P4 Chinese-CLIP** | `.env.example` 文档 + `mmrag reindex --yes` flag(CI 友好);settings.clip_model 默认不变(避免破坏既有 collection),用户 opt-in | `.env.example`, `cli.py` | 5 reindex tests |
| **P5 PDF chunk-by-section** | `chunk_splitter.split_by_heading` 三规则(ATX # / font-size / 双边界短行);PyMuPDF 路径走 page→sections→chunks,带 `metadata.section` + `chunk_index` | `parsers/chunk_splitter.py`, `parsers/pdf_parser.py` | 8 tests |
| **P5 关键词 enrichment** | `text_keywords.extract_keywords_zh(en/auto)`;PyMuPDF 和 PaddleOCR-VL 路径都注入 `关键词: ...` footer,给 BM25 channel 显式 token | `text_keywords.py`, `parsers/pdf_parser.py` | 11 tests |

总测试:`313 → 355` (+42)。

## v4 数字

| 模式 | v3 | v4 (本轮) | Δ | 备注 |
| --- | --- | --- | --- | --- |
| text→text hit@5 | 0.300 | **0.300** | — | embedding 没切中文 model,documents.jsonl 没重 parse(集成) |
| zh_on_en | 0.400 | 0.400 | — | dense 仍是 qwen3-embedding |
| en_on_en | 0.400 | 0.400 | — | fuzzy 救了 `transformr` (rank 3 hit) |
| zh_on_zh | 0.250 | 0.250 | — | 关键词 enrichment 没生效在 collection 里(没 reparse) |
| negative | 0/8 | 0/8 | — | min_score=0.30 default,bundled corpus 阈值 1.20 才是甜点 |
| text→image | 0.087 | 0.087 | — | 没切 Chinese-CLIP |
| image→image | 1.000 | 1.000 | — | — |

**坦白**:本轮没切 dense model / CLIP,也没 reparse,所以 v3 baseline 没退化,也没提升。**所有改动的潜在收益需要两步手动操作才解锁**:

1. 设 `EMBEDDING_BACKEND=sentence_transformers` + `EMBEDDING_MODEL=BAAI/bge-m3` + `mmrag reindex --text-only --yes`
2. 设 `CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16` + `mmrag reindex --image-only --yes`

下面给完整 reindex 流程和预期数字。

## 用户 reindex 指南

### 路径 A: 切中文 embedding(text→text zh_on_en/zh_on_zh 提升)

```bash
# 1. 装 bge-m3 (~2GB 首次下载,后续 HF cache 命中)
pip install sentence-transformers  # 已装, [clip] extra 自带

# 2. 改 .env
cat >> .env <<EOF
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=16
EOF

# 3. 停 API server + reindex
mmrag reindex --text-only --yes

# 4. 跑 v2 eval 验证
python /tmp/run_v2_eval.py
```

**预期**:
- zh_on_en 0.40 → 0.60+ (BGE-m3 跨语言检索显著强于 qwen3-embedding)
- zh_on_zh 0.25 → 0.45+ (BGE-m3 中文密度更高)
- 文本 collection dim suffix 改 1024d(`multimodal_text_1024d`)

### 路径 B: 切 Chinese-CLIP(text→image 救活)

```bash
# 1. 改 .env
cat >> .env <<EOF
CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16
IMAGE_RELEVANCE_THRESHOLD=0.20
EOF

# 2. 停 API server + reindex image
mmrag reindex --image-only --yes

# 3. 验证
mmrag search --image-query "飞机"  # 期望召飞机图
mmrag search --image-query "熊猫"  # 期望召熊猫图
```

**预期**:
- text→image hit@5: 0.087 → 0.50+
- ZH 子组: 0/17 → ≥ 9/17
- image collection dim suffix 改 768d(`multimodal_image_768d`)

### 路径 C: 走 PDF chunk-by-section + 关键词(要重新 parse)

```bash
# 跑新 parse 触发 P5 代码路径
# 默认 enrich_chunk_with_keywords=true, ENRICH_CHUNK_LANGUAGE=auto
mmrag parse ./examples/data/chapter11_assets/pdfs/*.pdf --no-auto-meta
mmrag reindex --text-only --yes
```

**预期**:
- zh_on_zh 0.25 → 0.45+ (关键词 `联宝`, `ESG` 显式入 BM25 index)
- 短 query `联宝 ESG` 能直接命中(目前 miss)

## 已知 v4 trade-off

- **P3 RRF 权重 via prefetch limit scale** 是 workaround。Qdrant 1.10+ 服务端支持 RRF k 常数配置;client 1.18 没 expose per-prefetch weight。等 Qdrant 升级或手写 client-side RRF 3 路合并。
- **P2 fuzzy** 用 `difflib.get_close_matches` 一次性 O(|vocab|),在 10k token 级别很快。corpus > 100k token 时考虑 trie + Levenshtein 自动机。
- **P5 chunk-by-section** 在 PyMuPDF 路径生效。PaddleOCR-VL 输出基本是 markdown,ATX 规则够用,字体 heuristic 暂未应用到该路径(等加 markdown-aware splitter)。
- **P1 切 BGE-m3** 后 collection dim suffix 改变,旧 2560d collection 留着无害但占空间;用户可手动删 `~/.mm_asset_rag/indexes/qdrant/collection/multimodal_text_2560d/`。

## 复现 v2 eval

```bash
# 假设 /tmp/run_v2_eval.py 在上一轮已写;如果不在:
cat > /tmp/run_v2_eval.py <<'PY'
import contextlib, os, sys
from mm_asset_rag.config import load_env
from mm_asset_rag.evaluation_v2 import (
    run_text_to_text_eval_v2,
    run_text_to_image_eval_v2,
    run_image_to_image_eval_v2,
    write_eval_report_v2,
)
load_env()
with contextlib.redirect_stderr(open(os.devnull, 'w')):
    t2t = run_text_to_text_eval_v2(top_k=5)
    t2i = run_text_to_image_eval_v2(top_k=5)
    i2i = run_image_to_image_eval_v2(top_k=5)
for label, rs in (("text→text", t2t), ("text→image", t2i), ("image→image", i2i)):
    hits = sum(1 for r in rs if r.hit)
    print(f"=== {label} ===  total={len(rs)}  hits={hits}  hit_rate={hits/len(rs):.3f}")
write_eval_report_v2({"text_to_text": t2t, "text_to_image": t2i, "image→image": i2i})
PY

python /tmp/run_v2_eval.py
cat ~/.mm_asset_rag/eval_report_v2.md
```

## 健康检查

- `pytest tests/unit -q` → **355 passed** (+42)
- `ruff check .` → 干净
- `ruff format --check .` → 干净
- `graphify update .` → 略(下次再跑)
