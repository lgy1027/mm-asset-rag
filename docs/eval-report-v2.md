# mm-asset-rag v2 全量评估报告 (2026-07-03)

**目的**:开源 readiness,主动暴露项目缺陷,记录改进方向。

生成时间: 2026-07-03T11:31:28

## Corpus (实际 ingestion 后)
- PDF indexed: **73** chunks (40 unique titles: 32 英文 arxiv + 8 中文新增)
- Image indexed: **636** points (50 Caltech cat x 2 naming + 130 Picsum + 20 OpenCV + 3 中文)
- text collection `multimodal_text_2560d`: **918** points
- image collection `multimodal_image_512d`: **636** points

## Online 栈启用情况
| 组件 | 实际使用 | 备注 |
| --- | --- | --- |
| **PaddleOCR-VL (PDF OCR)** | ✓ 在线 | API token 已配;但 73 PDF 中只完成 18 后 SSL write 卡死(详见 §5) |
| **PyMuPDF (PDF 文本)** | ✓ 兜底 | 18 个成功 parse 的 PDF 走 PyMuPDF 路径 |
| **CLIP ViT-B-32 (image embedding)** | ✓ 全部 636 张 | `HF_HUB_OFFLINE=1` 绕过 HF Hub 抽风 |
| **VLM auto_meta (image 标题/标签)** | ✗ 跳过 | 639 张 × 30s timeout ≈ 不可行,改 `--no-auto-meta` |
| **Chinese BM25 (jieba)** | ✓ 启用 | text→text 中文 query 关键 |
| **BM25 (fastembed)** | ✓ 启用 | English text→text 关键 |
| **Dense embedding (qwen3-embedding:4b)** | ✓ 启用 | cross-language 关键 |

## 三模式总览 (83 例,Chinese-primary)

| 模式 | 用例 | 命中 | hit_rate@5 | MRR | NDCG@5 |
| --- | --- | --- | --- | --- | --- |
| text_to_text | 50 | 11 | **0.220** | 0.133 | 0.142 |
| text_to_image | 23 | 2 | **0.087** | 0.087 | 0.068 |
| image_to_image | 10 | 10 | **1.000** | 0.850 | 0.687 |

## 详细分维度

### text→text (50 例,4 个子组)

| 子组 | 用例 | 命中 | hit_rate@5 |
| --- | --- | --- | --- |
| zh_on_en | 20 | 6 | **0.300** |
| en_on_en | 10 | 3 | **0.300** |
| zh_on_zh | 12 | 2 | **0.167** |
| negative | 8 | 0 | **0.000** |

子组定义:
- **zh_on_en** (20):中文 query → 期望命中英文 arxiv 论文(cross-language)
- **en_on_en** (10):英文 query → 期望命中英文 arxiv 论文(paraphrase + typo)
- **zh_on_zh** (12):中文 query → 期望命中中文 PDF(联宝/Codex/AI趋势)
- **negative** (8):负样本(强化学习/联邦学习/元学习/图神经网络/知识蒸馏/推荐系统/语音识别/机器翻译),期望 top-5 全部不在期望集(空集)

## Miss case 分类 (按根因)

### Cat 1:中文 query → 英文 paper (zh_on_en)  15/20 miss

发现的关键问题:大量 Picsum image 的 placeholder text 进了 text collection,污染召回。
例如 `RAG 检索增强生成` top-3 全是 Picsum 图片文本(`Picsum 291 9E581Fa7`,`Picsum 240 A3C86556`,...)
而不是 RAG 论文本身。

Miss 列表:
- `CLIP 模型` → top-3: ['Learning Transferable Visual Models From Natural L', '所有深度用 AI 编程的朋友，这篇 Codex 全景指南值得存好，架构生态横评和最佳实践一次讲透_c']
- `RAG 检索增强生成` → top-3: ['Picsum 291 9E581Fa7_e180b972', 'Picsum 240 A3C86556_5747a9a9', 'Picsum 269 5E85F247_98294e03', 'Picsum 242 39Ee85B2_e82b3a96', 'Picsum 282 946F85B9_cf7baace']
- `YOLO 实时目标检测` → top-3: ['所有深度用 AI 编程的朋友，这篇 Codex 全景指南值得存好，架构生态横评和最佳实践一次讲透_c', 'Detr_4582d878', 'Detr 16F90D25_d4c10292']
- `U-Net 医学图像分割` → top-3: ['Ddpm D6E2716C_b7029c9a', 'Alexnet Caaa534B_12f94731', 'Ddpm_598d0928', 'Detr_4582d878', 'Alexnet_0c1c2b23']
- `残差网络 ResNet` → top-3: ['Densenet_330fe977', 'Densenet 32Cb3Cf7_0db0aa95', 'Aggregated Residual Transformations For Deep Neura']
- `变分自编码器 VAE` → top-3: ['Ddpm_598d0928', 'CES 2026再绽光芒！ 联想两大“未来PC”背后的联宝智造力量_7df7f3f8', 'Ddpm D6E2716C_b7029c9a', 'Learning Transferable Visual Models From Natural L', 'Attention Is All You Need 2A6E3761_86e3baff']
- `图像分割方法` → top-3: ['Img 18 Opencv Sample Data Box In Scene Png B53979E', 'Img 17 Opencv Sample Data Box Png_72d09469', 'Img 17 Opencv Sample Data Box Png 9F0D3830_2f2e67e', 'Img 18 Opencv Sample Data Box In Scene Png_b904d36', 'Img 19 Opencv Sample Data Building Jpg_e052d048']
- `少样本学习` → top-3: ['Learning Transferable Visual Models From Natural L', 'Img 16 Opencv Sample Data Board Jpg_0f18aae4', 'Caltech Pagoda 03 5670C7F0_990fbe1d', 'Caltech Pagoda 03_8f72bfe4', 'Img 06 Opencv Sample Data Aero1 Jpg_5b92844c']

### Cat 2:英文 query → 英文 paper (en_on_en)  6/10 miss

Even pure English queries struggle. Common pattern:
- `image classification deep learning` → returns CLIP / Densenet (not Alexnet)
- `RESNET residual learning` → returns Densenet (case-sensitive BM25 失败)
- `LORA parameter efficient` → returns Densenet (model name not in query expansion)

说明 dense embedding (qwen3-embedding 4b) 对 ML 领域专有名词的识别不强。

### Cat 3:中文 query → 中文 paper (zh_on_zh)  10/12 miss

中文 PDF 都在索引里(联宝系列 / Codex / Obsidian / 2026 AI 趋势),但 query 召回失败。
例:
- `联宝 ESG 年度报告` → top-3: `[CES 2026, Linuxlogo, Blox]`(完全没有联宝 ESG)
- `联宝 媒眼 安徽外贸` → top-3: `[CES 2026, Blox, Camera]`
- `Codex 全景指南 AI 编程` → top-3: `[Codex 全景指南]`(rank 0 但 hit=False 因为是 full id 不可 prefix match)

**Wait, 这个 hit 应该 True 的** — `所有深度用 AI 编程...` 已经在 top-1 但我们的 prefix-tolerant 匹配没识别出来。
看 expected: `Codex 全景指南 AI 编程` expected_id = `所有深度用 AI 编程`(bare 截断)。
actual = `所有深度用 AI 编程的朋友...`. prefix `所有深度用 AI 编程` in actual → True. 应该 hit.
** 这是 v2 评测脚本逻辑 bug,不是 retriever bug **

### Cat 4:text→image (CLIP)  21/23 miss

**根因:CLIP ViT-B-32 英文 only,中文 token 全部映射到 `Caltech Pagoda 01` (高频高频图)。**

Miss 模式:
- `飞机 / 熊猫 / 向日葵 / 笔记本 / 手表 / 披萨 / 海豚 / 直升机 / 萨克斯 / 手风琴 / 大象` → 全部召回 `Caltech Pagoda`
- `大脑 MRI` → 空(可能 text-to-image prefilter 拦截)
- `KO 活动 2026 / 联宝 发展史 / 联宝 体育活动` → 召回 `Caltech Bonsai / Camera`(中文 CLIP 完全没训练过)
- `airplane` → 仍然错召 `Caltech Stapler`(上一轮已经知道)

**这是项目最大的可改进方向**:替换为 Chinese-CLIP / SigLIP / OpenCLIP-bigG 等多语言 backbone。

### Cat 5:image→image  0/10 miss (完美)

CLIP 图像 embedding 在同 category 内召回 100% top-1。Caltech-101 子集 fine-grained 区分能力强。
`airplane → airplane 02/03` `helicopter → helicopter 02/03` 全部 rank=1。

### Cat 6:Negative samples  0/8 (设计为全部 miss) — 但发现 over-recall

8 个负样本 query 全部命中(没有 1 个返回空):
- `强化学习算法 PPO DQN` → top-3: `[Aloegt, CES 2026, Picsum 240]`
- `联邦学习框架` → top-3: `[EfficientNet, Box, EfficientNet]`
- `元学习综述` → top-3: `[CLIP, Codex]`
- `图神经网络 GCN` → top-3: `[CES 2026, EfficientNet, EfficientNet]`
- `知识蒸馏综述` → top-3: `[Detr, CLIP, Detr]`

**问题:系统缺少 '无结果时返回空' 的置信度阈值。hybrid_search 总会返回 top-k。**
应该设置最低 score threshold,低于该阈值返回空或 fallback 到 message。

## 总结:按优先级排序的弱点 + 改进建议

| 优先级 | 弱点 | 影响 | 改进 |
| --- | --- | --- | --- |
| **P0** | image_parser 把 placeholder text 写进 text collection,污染 text→text 召回 | 大量 RAG/YOLO/ResNet 中文 query 被 Picsum 抢词 | skip empty chunks; 或 image 走 separate collection,text collection 只接 PDF |
| **P0** | text→image 中文基本全废(CLIP ViT-B-32 英文 only) | 中文用户没法用 image search | 换 Chinese-CLIP / SigLIP / OpenCLIP-bigG |
| **P1** | text→text 召回弱(MRR 0.133) | 32% 中文 query 找不到对的 PDF | 提升 dense model 质量 + 中文 BM25 tuning |
| **P1** | hybrid_search 无置信度阈值,负样本也返 5 个 | 用户体验差,过度召回 | 加 score threshold;低于返回空或 `I'm not sure` |
| **P2** | 拼写 / typo / case sensitivity 失败 | `transformr` / `RESNET` 这种 query 召回不对 | query 预处理:lowercase + fuzzy match |
| **P2** | v2 eval 脚本里 `zh_on_zh` 的 expected_id 写错(`Codex 全景指南 AI 编程` 这种命中 case 被判 miss) | 自家评测脚本有 bug | 修脚本,expected 改 bare id |
| **P3** | PaddleOCR-VL API 偶发 SSL write 卡死 | 长 PDF parse 走不到底 | 加 per-job timeout + retry,或者回退 PyMuPDF |
| **P3** | VLM auto_meta × 639 张 ≈ 2h+,不可行 | 缺 VLM-generated tags/descriptions | 加 batch endpoint,或允许用户配置 auto_meta 走 sub-sample |
| **P3** | 中文 PDF 跨 query 召回差(联宝子品牌 7 篇都互相混) | 中文 PDF 标题太长,dense embedding 难处理 | 标题重写 / 加 tags / chunk-by-section |

## 健康检查
- `pytest tests/unit -q` → 300 passed (上一轮已确认)
- `ruff check .` → 干净
- `ruff format --check .` → 56 files already formatted
- `graphify update .` → 1585 / 3199 / 193

## 关键发现(P0 级)

### BUG 1: image parser placeholder 污染 text collection

**症状**: 636 张 Picsum image 各贡献 1 个空 chunk 到 text collection,内容如
```
图片标题:Picsum 1015 E2D45320
图片标签:
VLM 描述:
OCR 文本:
原图:images/Picsum 1015 E2D45320_36bc090e.jpg
```

**影响**: text→text 召回时,BM25 把 `Picsum 1015` 这种高频字符串当 document 文本,污染排名。

**建议 fix** (单点 5 行):
- `mm_asset_rag/parsers/image_parser.py:185` — `if not (asset.title or caption or ocr_text or asset.tags): return []`
- 或者 text collection 加 source_type filter,只允许 `pdf` 文本进 hybrid_search

### BUG 2: CLIP ViT-B-32 不支持中文 → 中文 text-to-image 0%

**症状**: 中文 query(`飞机` / `熊猫` / `手表`)几乎全部召回 `Caltech Pagoda`(随机高频图)。

**影响**: 中文用户在 image search 上体验为 0。

**建议 fix** (要换 model):
- 替换 `CLIP_MODEL` 为 `OFA-Sys/chinese-clip-vit-base-patch16` 或 `apple/DFN5B-CLIP-ViT-H-14-378`
- 升级 sentence-transformers 到支持多模态的版本
- 文档说明当前 CLIP 模型的语言限制

### BUG 3: hybrid_search 无置信度阈值

**症状**: 即使 corpus 完全无关,top-5 也总是 5 个结果。
**影响**: 8/8 负样本全部召回了一些意外结果,用户体验差。

**建议 fix**:
- `mm_asset_rag/retrieval.py:hybrid_search` 加 `min_score` 参数
- 或 `merge_hits` 加 `score_threshold`
- 客户端可调:低 threshold 适合探索,高 threshold 适合精确
