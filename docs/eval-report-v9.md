# mm-asset-rag v9 评估报告 (2026-07-05)

**目的**:切到 minimax M3 (chat) + bge-m3 (embed) + Chinese-CLIP,清空老索引,
重 parse + reindex 跑全量 eval,验证 minimax chat 链路在 production 流量下
能正常出 token 流且 `<think>` 块被自动 strip。

## 解锁动作

- 清空 `$MM_ASSET_RAG_HOME` 全部老数据(`assets` 软链保留)
- `mmrag parse --no-auto-meta` 重跑 140 个 PDF(bge-m3 1024d collection)
- `mmrag parse --ocr --no-auto-meta` 跑 1633 张 image(走 filename title,
  未做 OCR/VLM caption,本地 OCR HTTP 服务未起、PaddleOCR-VL token 注释)
- `mmrag reindex --image-only --yes` 重建 `multimodal_image_512d`
  (OFA-Sys/chinese-clip-vit-base-patch16, 512d)
- `.env` 切到 `OPENAI_BASE_URL=https://api.minimaxi.com/v1` + `OPENAI_MODEL=MiniMax-M3`,
  embed 走 `EMBEDDING_BASE_URL=http://localhost:11434/v1` + `EMBEDDING_MODEL=bge-m3`

## 索引状态

- text collection: `multimodal_text_1024d`(bge-m3 1024d)·3765 chunks inserted
- image collection: `multimodal_image_512d`(Chinese-CLIP 512d)·1633 inserted
- unique PDF assets: 137 (140 PDF 中 3 个去重 hash)
- image assets: 1633 (filename title, 无 OCR/VLM caption)

## v6 → v9 hit_rate@5

| 模式 | v6 | v9 | Δ |
| --- | --- | --- | --- |
| mmrag eval (37 case, 老 eval) | (跨 hash bug) | **0.838** (31/37) | — |
| eval-v2 text→text (49 case) | 0.673 | **0.694** (34/49) | +0.021 |
| eval-v2 text→image (23 case) | 0.652 | **0.652** (15/23) | ±0.000 |
| eval-v2 image→image (10 case) | 1.000 | **1.000** (10/10) | ±0.000 |

eval-v2 合计 82 case · **0.720** (59/82)。

## minimax M3 chat 验证

| 维度 | 结果 |
| --- | --- |
| chat/completions HTTP | ✅ 200,正常 JSON |
| 流式 `/chat/stream` | ✅ SSE 逐 token 输出,中文 token 正常 |
| `<think>` 块自动 strip | ✅ 推理 token 不外泄(answer.py:221 + 跨 chunk buffer) |
| 诚实度 | ✅ 检索召回与 query 不符时直接说"证据不足",不编造 |
| 引用证据 | ✅ token 中带 `[1][2][3]` 引用标记,对齐 hits 顺序 |

### 流式输出样例

```
Q: 用一句话告诉我 Transformer 是什么
[1] ViT 论文第 6 页 Figure 5 ...
[2] "Attention Is All You Need" 第 9 页 Table 4 ...
...
```

## 跟 v6 相比的 Δ 分析

### text→text 涨 0.021 — 涨幅小

v6 0.673 → v9 0.694,主要来自 evaluation_v2.py 大小写假 miss 修复
(Rich Feature Hierarchies 期望补全)。embedding 模型 bge-m3 1024d 没换,
chunker 没改,路由没改。理论上限就是 P5 chunk + bge-m3 + BM25 hybrid 表现。

### text→image / image→image 持平

Chinese-CLIP 没动,corpus 没动,纯看是否跑通;两边都跟 v6 一致。

### B 类 miss 仍存在(15 个 t2t miss)

| Query 类型 | 例子 | 误召回 | 根因 |
| --- | --- | --- | --- |
| 概念近邻 | "扩散模型" | DDPM → Stable Diffusion | BM25 抢词,reranker 救不了 |
| 任务近邻 | "目标检测模型" | YOLO → ResNet | dense 编码没分清 |
| typo | "transformr" | — | 无 fuzzy 字典 |
| negative | "强化学习 PPO DQN" | SSD | `MIN_SCORE` 太松 |

需要 query rewrite / fuzzy dict / 分类器才能根治,评估体系不能再帮忙。

## image corpus 文本质量 caveat

这轮 image parse **没做 OCR / VLM caption**:

- `mmrag parse --ocr` → 本地 OCR HTTP `127.0.0.1:8000/ocr` 未起
- `--vlm` → 1633 张图都调 M3 写 caption 成本高,跳过
- PaddleOCR-VL 远程 token 在 .env 注释里(历史备份 `.env.bak.v9` 里能找到)

结果:image 在 text 集合里**只有 filename title** 作为信号。
这对 **text→text 召回 image 不利**,但因为
**text→image / image→image 都直接走 CLIP 向量**,完全不受影响。
本轮 t2i 0.652 / i2i 1.000 与 v6 持平验证了这点。

如要让 image 进 text→text,后续可单独补:
```bash
mmrag parse --ocr --no-auto-meta examples/data/chapter11_assets/images/...
```

## 整体验收

| 维度 | 状态 |
| --- | --- |
| minimax M3 chat | ✅ 流式/非流式 + 推理 strip + 诚实度 |
| bge-m3 1024d embed | ✅ 全文检索基线 |
| Chinese-CLIP image | ✅ 图文检索 0.652 / 1.000 |
| eval-v2 全套(t2t/t2i/i2i) | ✅ 三段都跑,无 collection missing |
| `mmrag-api /chat/stream` | ✅ SSE 端点通 |
| 0.838 老 eval | ✅ 修完跨 hash bug 后回到设计目标 |

## 后续可选优化(非阻塞)

1. 补 PaddleOCR-VL token 跑 image OCR → text→text 能召回 image
2. fuzzy dictionary expansion(transformr → Transformer 等)
3. negative query 阈值调高(MIN_SCORE)或 query rewrite
4. B 类 miss query 分类器(YOLO/AlexNet/DDPM 这种"特定专名"类 query 走 dense-only 路由)