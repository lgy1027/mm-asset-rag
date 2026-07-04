# mm-asset-rag v6 评估报告 (2026-07-04)

**目的**:解锁 v4 已写但未生效的代码路径(P5 chunk-by-section + 关键词 footer、bge-m3、Chinese-CLIP),测量真实 Δ vs v5。

## 解锁动作
- reparse 146 个 PDF(`--pdf-parser pymupdf --no-auto-meta`)→ 触发 P5 chunk-by-section + 关键词 footer
- reindex text(bge-m3 1024d)
- 切 Chinese-CLIP(`OFA-Sys/chinese-clip-vit-base-patch16`)+ reindex image

## 索引状态变化
- documents.jsonl: 918 → **4794 chunks**(P5 chunk-by-section 生效,77% chunk 带 `关键词:` footer)
- text collection: `multimodal_text_1024d`(bge-m3)
- image collection: 切到 Chinese-CLIP(768d)

## v5 → v6 Δ(hit_rate@5)
| 模式 | v5 | v6 | Δ |
| --- | --- | --- | --- |
| text→text | 0.280 | **0.673** | **+0.393** |
| text→image | 0.087 | **0.652** | **+0.565** |
| image→image | 1.000 | **1.000** | +0.000 |

> 注:0.673 是修正 eval 假 miss 后的真实基线(见 §假 miss 修正)。修正前为 0.620,有两个用例 expected 数据与 corpus 实际 title 不符(空格 / 中英译名)导致的假 miss 拖低了数字。

## 详细指标(v6)
### text→text
- hit_rate={'1': 0.5, '3': 0.62, '5': 0.62, '10': 0.62}, precision={'1': 0.5, '3': 0.39, '5': 0.39333333333333337, '10': 0.39333333333333337}, recall={'1': 0.25233333333333335, '3': 0.389, '5': 0.39899999999999997, '10': 0.39899999999999997}, f1={'1': 0.324, '3': 0.37214285714285716, '5': 0.37747619047619047, '10': 0.37747619047619047}, ndcg={'1': 0.5, '3': 0.4145846675898883, '5': 0.4107451618597088, '10': 0.4107451618597088}, mrr=0.560, map=0.351

### text→image
- hit_rate={'1': 0.6086956521739131, '3': 0.6521739130434783, '5': 0.6521739130434783, '10': 0.6521739130434783}, precision={'1': 0.6086956521739131, '3': 0.5797101449275363, '5': 0.5652173913043478, '10': 0.5652173913043478}, recall={'1': 0.10144927536231883, '3': 0.2898550724637681, '5': 0.4710144927536232, '10': 0.4710144927536232}, f1={'1': 0.17391304347826086, '3': 0.3864734299516908, '5': 0.5138339920948617, '10': 0.5138339920948617}, ndcg={'1': 0.6086956521739131, '3': 0.5882922293033583, '5': 0.5755433920664622, '10': 0.5135060564434492}, mrr=0.623, map=0.451

### image→image
- hit_rate={'1': 1.0, '3': 1.0, '5': 1.0, '10': 1.0}, precision={'1': 1.0, '3': 1.0, '5': 0.9800000000000001, '10': 0.9800000000000001}, recall={'1': 0.15, '3': 0.45, '5': 0.7333333333333334, '10': 0.7333333333333334}, f1={'1': 0.2593406593406593, '3': 0.6133333333333333, '5': 0.8267379679144385, '10': 0.8267379679144385}, ndcg={'1': 1.0, '3': 1.0, '5': 0.9868794922487659, '10': 0.8318487321329643}, mrr=1.000, map=0.733

## 关键发现
- **text→text 翻倍多(0.280→0.620)**:P5 chunk-by-section + 关键词 footer + bge-m3 全部生效。`RAG 检索增强生成`、`YOLO 实时目标检测`、`联宝 ESG` 等此前 miss 的 query 全部 rank=1 命中
- **text→image 7 倍(0.087→0.652)**:Chinese-CLIP 解锁中文,`熊猫/向日葵/海豚/直升机/手风琴` 等 12/13 中文 query rank=1 命中(此前中文 0/17)
- **image→image 持平 1.000**:CLIP 图像 embedding 同 category 召回稳定
- **negative 仍 8/8 over-recall**:`MIN_SCORE=0.30` 下 negative 全部命中(详见 §阈值实验)

## 阈值实验:MIN_SCORE=1.20 实测(v6b)

v3 报告曾记录 `MIN_SCORE=1.20` 是 bundled corpus 甜点(5/8 negative 返空,正样本保留 11/15)。**v6 解锁 P5 后重测,该结论失效**:

| 模式 | v6 (0.30) | v6b (1.20) | Δ |
| --- | --- | --- | --- |
| text→text hit@5 | 0.620 | **0.540** | -0.080 |
| negative 返空 | 0/8 | **1/8** | +1 |
| text→image | 0.652 | 0.652 | — |
| image→image | 1.000 | 1.000 | — |

**为什么 v3 的 1.20 在 v6 失效**:
- v3 时代 918 chunks(qwen3-embedding 2560d,无 chunk-by-section)
- v6 是 4794 chunks(bge-m3 1024d + P5 chunk-by-section),chunk 更小更内聚,**score 分布整体下移且更分散**
- 1.20 阈值卡掉了 4 个正样本(`扩散模型论文`/`变分自编码器VAE`/`联宝媒眼安徽外贸`/`transformer论文中文`,它们 top3 本有正确结果)
- negative 只救回 1 个(`推荐系统`),其余 7 个 negative 的误命中分数 > 1.20 —— negative 是**高分误命中**,阈值法拦不住

**结论**:已改回 `MIN_SCORE=0.30`。**negative over-recall 不能靠全局阈值解**,因为 negative 误命中分数与正样本重叠。下一轮需用更精准手段(query rewrite / classifier / per-route 阈值)。

## 仍未达标的维度(下一轮靶点,数据驱动)

- **negative over-recall**:全局阈值无效 → 候选:per-route 阈值、query 改写后 rerank、轻量 relevance classifier
- `embedding 词嵌入` / `目标检测模型` / `CLIP 中文版` 等 query 仍 miss:可能需 query rewrite 或 contextual retrieval(LLM 注入上下文前缀)
- `扩散模型 论文` miss 但 `Stable Diffusion` 在 top1:eval 期望匹配逻辑可能需 prefix 容忍(疑似 eval 脚本而非 retriever,类似 v2 报告的 bare id 问题)
- `transformr self attention` 仍 miss:fuzzy corrector 词典未覆盖

## 结论

v4 代码全部 ready,只是没 reparse/reindex。解锁后 text→text +0.393、text→image +0.565,**零代码改动**即拿到巨幅提升。印证 v5 报告判断:"换 model 这条 ROI 在本 corpus 有限" 不准确——解锁 P5 + Chinese-CLIP 后收益巨大。

**经验教训**:`MIN_SCORE` 甜点随索引状态变化(918→4794 chunks 后旧阈值失效),旧 sweep 数据不能直接套用,改阈值后必须重测正样本不丢。

## 假 miss 修正(P0,2026-07-04)

逐个 miss 分析发现 3 个用例 expected 数据与 corpus 实际 asset title 不符(已用 `asset_index.jsonl` 核实),导致假 miss:

| 用例 | expected(修前) | corpus 实际 | 修法 |
|---|---|---|---|
| `联宝 2026 财年 启幕` | `敢 AI 敢为`(带空格) | `敢AI敢为 志在必行...`(连写) | expected 改 `敢AI敢为`(去空格) |
| `CLIP 中文版` | `学习从自然语言监督中获取可迁移视觉模型`(中译名) | `Learning Transferable Visual Models...`(英文原题) | expected 改英文原题 |
| `2026 年 AI 技术趋势` | `2026 年 AI 技术趋势` | corpus 无此文档 | 删除用例 |

修正后 text→text: **0.620 → 0.673**(+0.053),两个假 miss 均 rank=1 命中。`_match` 匹配逻辑未动(空格/中英是字符串包含盲点,但改逻辑风险高于改数据)。

**此 0.673 是 contextual retrieval / reranker 等后续优化的对照基准。**

## Contextual Retrieval 试点(P1,2026-07-04)

实现 `mm_asset_rag/contextual.py`(方案 D:文档摘要 + chunk context,MiniMax-M3,opt-in `--contextual`)。小规模试点:对 10 个中文 PDF(联宝/Codex/Obsidian,153 chunks)生成 context,reindex,eval。

**整体 hit_rate 没动(0.673 → 0.673),但分维度看有大效果:**

| 子组 | v6-corrected | v7(中文PDF加context) | Δ |
| --- | --- | --- | --- |
| **zh_on_zh** | 0.583 | **1.000 (11/11)** | **+0.417** |
| zh_on_en | 0.700 | 0.700 | 持平(英文PDF未加context) |
| en_on_en | 0.800 | 0.800 | 持平 |
| negative | 0/8 | 0/8 | 持平 |
| 整体 | 0.673 | 0.673 | 被算术掩盖 |

**关键结论**:
- **contextual retrieval 对中文 PDF 场景效果显著**(zh_on_zh +0.417 到 100%),验证 ROI。生成的 context 质量高(样本:"本片段是'环境(E)'章节的战略引言,确立双碳目标的宏观视角...")
- **B 类 miss 未改善**,因为它们是"中文 query → 英文 arxiv 论文",而英文 PDF 没加 context。`扩散模型论文`→Stable Diffusion 抢词、`embedding词嵌入`→ViT 抢词等,根因在英文论文侧
- **要改善 B 类 miss,必须给英文 arxiv 论文也加 context** —— 这是全量 4158 chunk 的理由
- **negative over-recall 仍 8/8**:context 没治 over-recall(意料之中,negative 是检索期判别问题,需 reranker)

**下一步决策点**:
- 全量 4158 chunk 加 context(成本 ~9.4M token,30min-1h),预期改善 B 类 miss,text→text 0.673 → 0.72+
- 或先做 reranker(bge-reranker-v2)治 negative over-recall,再决定全量

**代码改动**(已落地,全绿):
- `mm_asset_rag/contextual.py`:generate_doc_summary + generate_chunk_context + enrich_docs_with_context(并发 + 缓存)
- `Settings`:contextual_enabled/concurrency/model/timeout/chunk_max_chars(opt-in 默认关)
- `service._do_parse`:写盘前调 enrich_docs_with_context,缓存 parsed/\<id\>/context.jsonl
- `qdrant_backend.build_qdrant_text_index`:L594 拼 context 前缀喂 embedder,payload text 保持原始正文
- `cli.py`:`mmrag parse --contextual` flag
- 5 单测(mock LLM + qdrant 拼前缀),370 passed,lint+format 干净

## 二阶段 Reranker(P1b,2026-07-04)

实现 `mm_asset_rag/embedders/reranker.py`(bge-reranker-v2-m3,sentence-transformers CrossEncoder 本地跑,opt-in)。在 `hybrid_search` 的 `merge_hits` 后插入 rerank:取 `reranker_top_n`(20)候选 → CrossEncoder 用 `(query, evidence)` 精排 → 返回 `reranker_top_k`(5)。不需 reindex(检索期组件)。

**环境踩坑**:用户本地 ollama 有 `qllama/bge-reranker-v2-m3`,但实测官方 ollama **无 rerank 端点**(`/v1/rerank` / `/api/rerank` 都 404),把 cross-encoder 当 embedder 跑(`/api/embed`)直接让 llama-server 崩溃(`GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)`)。改用 sentence-transformers CrossEncoder(项目已有 `sentence-transformers 5.6.0`,与 bge-m3 embedder 同源同依赖)。

**v6-corrected → v8(reranker 启用)Δ**:

| 模式 | v6-corrected | v8(reranker) | Δ |
| --- | --- | --- | --- |
| text→text | 0.673 | **0.714** | **+0.041** |
| text→image | 0.652 | 0.652 | 持平 |
| image→image | 1.000 | 1.000 | 持平 |

**rescue 的关键 B 类 miss**(cross-encoder 区分了语义相邻论文):
- `讲 diffusion 去噪扩散的论文` → rank 1 = **DDPM**(之前 Stable Diffusion 抢 top1)✅
- `U-Net 跳跃连接 skip connection` → rank 1 = **U-Net**(之前 Pix2Pix 抢)✅
- `残差网络 ResNet` → rank 2 = Resnet(之前 rank 1 是 Resnext,reranker 把 Resnet 提上来)

**negative over-recall 仍 8/8**:reranker 治不了——negative(`强化学习 PPO DQN`)在 corpus 里有 token 重叠(`Ssd`/`Gan` 等),cross-encoder 仍打正分。negative over-recall 的真正解是"I don't know" 判别或更高 min_score,不是 reranker。但 reranker 显著改善了**正样本精度**(B 类 miss rescue)。

**累计提升**(v5 → v8):
| 模式 | v5 | v8 | 累计 Δ |
| --- | --- | --- | --- |
| text→text | 0.280 | 0.714 | **+0.434** |
| text→image | 0.087 | 0.652 | **+0.565** |

**代码改动**(已落地,全绿):
- `mm_asset_rag/embedders/reranker.py`:Reranker 类(lazy-load CrossEncoder + 模块级单例 + 失败 sticky)+ get_default_reranker + reset_reranker
- `Settings`:reranker_enabled/model/top_n/top_k(opt-in 默认关)
- `embedders/__init__.py`:导出 reranker API
- `retrieval.hybrid_search`:reranker 启用时 fetch_k=reranker_top_n,merge 后 rerank,降级透明
- 7 单测(mock CrossEncoder + hybrid_search 开关),377 passed,lint+format 干净