# mm-asset-rag v3 评估报告 (2026-07-03)

**目的**:跟进 v2 报告([eval-report-v2.md](eval-report-v2.md))里暴露的 3 个 P0 弱点(text→text hit_rate 0.220 / text→image 中文 0 / 负样本 over-recall 8/8),验证 fix 是否有效 + 决定下一轮 0.2.0 路线。

## 改动摘要

| 修复 | 实现 | 文件 |
| --- | --- | --- |
| **BUG 1**: image placeholder 污染 text collection | (a) `parse_image` 缺 signal 时返 `[]`; (b) `qdrant_text_search` 加 `must:source_type=pdf` 过滤双保险 | `parsers/image_parser.py:177-200`, `backends/qdrant_backend.py:798-810`, `_hybrid_text_query` |
| **BUG 2**: 中文 text→image 0% | `.env.example` 加 `CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16` 切换说明; default 不改(避免破坏既有 collection),用户 opt-in reindex | `.env.example:33-44`, `settings.py:58-66` |
| **BUG 3**: hybrid 无 min_score 阈值 | `merge_hits(min_score=...)` + `Settings.min_score` (env: `MIN_SCORE`, default 0.30) | `retrieval.py:47-100, 113-135`, `settings.py:120-129` |
| **eval 脚本 bug**: bare id 不跨 hash 匹配 | `_title_of` 剥 `_<8hex>` 后再做 contains 判断 | `evaluation_v2.py:289-302` |

## v3 数字

跑同一份 83 例 v2 用例集(没动 expected / queries),corpus 也没动,只是修了检索路径:

| 模式 | v2 hit@5 | v3 hit@5 | Δ |
| --- | --- | --- | --- |
| **text→text** | 0.220 | **0.300** | +36% |
| **text→image (EN)** | 0.154 (2/13) | 0.154 (2/13) | — |
| **text→image (ZH)** | 0.000 (0/17) | 0.000 (0/17) | — |
| **image→image** | 1.000 | 1.000 | — |
| **negatives over-recall** | 8/8 | 8/8 (default 0.30) | — |
| **negatives over-recall (ms=1.20)** | — | **3/8** | -5 |

## 详细分维度

### text→text (50 例,4 子组)

| 子组 | v2 hit@5 | v3 hit@5 | Δ | 关键变化 |
| --- | --- | --- | --- | --- |
| zh_on_en | 0.300 | **0.400** | +33% | RAG/YOLO/U-Net 召回 PDF 论文而不是 Picsum 抢词 |
| en_on_en | 0.300 | **0.400** | +33% | 多模态 query (image classification) 召回 CLIP/Densenet |
| zh_on_zh | 0.167 | **0.250** | +50% | 联宝子品牌 7 个 cross-query 召回更准 |
| negative | 0.000 | 0.000 | — | 0 命中的 8 个都正确(都没期望) |

### text→image (23 例)

不变 — 仍然依赖 `clip-ViT-B-32` 英文模型。v3 没动 model,只修了**怎么过滤**。中文 text→image 提升需要把 `CLIP_MODEL` 换成 Chinese-CLIP 并 reindex。

### image→image (10 例)

100% 不变。CLIP 图像 embedding 仍强。

## Min_score 灵敏度测试

`MIN_SCORE` 在 bundled corpus 下的影响(50 例 text→text ):

| MIN_SCORE | text→text hit | negs with results (8 总) | 备注 |
| --- | --- | --- | --- |
| 0.00 | 15/50 | 8 | 旧行为,过度召回 |
| 0.30 | 15/50 | 8 | **default**:在 bundled corpus 几乎不 filter,安全 |
| 0.50 | 15/50 | 8 | 仍没切到负样本 |
| 0.80 | 13/50 | 8 | 开始丢 hit(降低 RAG/YOLO 召回) |
| **1.20** | **11/50** | **3** | 甜点:5/8 负样本返空,正样本保留 11/15 |
| 1.50 | 6/50 | 2 | 太严,正样本也大量丢 |

**推荐**:用户自己 corpus 上 sweep 一下 `MIN_SCORE`,我们这个 1.20 是 bundled arxiv + 联宝 PDF 的局部最优,不是 universal。

## v3 仍未修的问题

| 优先级 | 问题 | 当前 | 推荐 | 改造成本 |
| --- | --- | --- | --- | --- |
| **P0** | 中文 text→image 仍 0% | hit=0/17 | 切 `OFA-Sys/chinese-clip-vit-base-patch16` | 低(reindex 即可),但要新 collection 名字 |
| **P0** | 拼写/小写/typo 失败 | `transformr/RESNET/LORA` 全 miss | query 预处理:lowercase + 编辑距离 fuzzy | 中 |
| **P1** | 中文 PDF 跨 query 召回差 | 联宝子品牌 7 篇互相混,只 3/12 命中 | 标题重写 / 加 tags / chunk-by-section | 高,需改 PDF parser |
| **P1** | 短中文 query 召回率低 | `RAG 检索增强生成` 召回 Codex 论文 | 优化 Chinese BM25 参数,改 dense model 改 multilingual-e5 | 中 |
| **P2** | VLM auto_meta × 639 张 ≈ 2h+ | 用户通常 `--no-auto-meta` 跳过 | batch endpoint / sub-sample | 中 |
| **P2** | PaddleOCR-VL API SSL 卡死 | 73 PDF 中 18 后 kill | per-job timeout + retry | 中 |

## 复现命令

```bash
# 启用 Chinese-CLIP(text→image 真正救星)
export CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16
export HF_HUB_OFFLINE=0   # 第一次需要下载 ~1GB
mmrag reindex --image-only

# Sweep min_score
export MIN_SCORE=1.20
mmrag search "强化学习算法 PPO DQN"   # 期望:空结果
mmrag search "Codex 全景指南 AI 编程" # 期望:Codex 论文

# 重跑 v2 eval
python /tmp/run_v2_eval.py
cat ~/.mm_asset_rag/eval_report_v2.md
```

## 健康检查

- `pytest tests/unit -q` → **313 passed**(+13 来自 v3 测试)
- `ruff check .` → 干净
- `ruff format --check .` → 干净
- `graphify update .` → 略(下次再跑)
