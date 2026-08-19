# 数据流:文本 vs 图片两条线

> 本文从**数据怎么流**的角度看 mm-asset-rag:`architecture.md` 是"分层组件视角",这里是"端到端数据流视角"。两者互补,可对照阅读。

## 一、上传:进来先识别

```
用户上传文件
   │
   ├─ sniff(本地,纯 magic bytes)判类型 + 抽元数据(页数/尺寸/EXIF)
   ├─ 可选 VLM 抽标题/标签/描述(auto_meta,需配 LLM)
   └─ 用户可编辑预览卡 → confirm
        │
        └─ 落盘到 assets/{pdfs|images}/,起后台 parse+index 线程
```

## 二、解析:按文件类型选解析器

| 文件 | 默认解析 | 抽出来的内容 |
| --- | --- | --- |
| **文本型 PDF** | PyMuPDF | 逐页文本 + 行级位置;页内嵌图单独抽出 |
| **扫描型 PDF**(几乎无文字) | 自动 fallback **本地 PP-OCRv6** | 逐页 200dpi 渲图 → OCR 出文字 |
| **Office/HTML/MD**(docx/pptx/xlsx/html/md) | MarkItDown | 转文本;内联 base64 图解码落地 |
| **独立图片**(jpg/png) | 本地 PP-OCRv6 + 可选 VLM 描述 | 图里的文字 + 图的语义描述 |

> 扫描 PDF 有百度 token 才走在线 PaddleOCR-VL,否则本地零外网。
> docling 是可选重栈(PDF/Office 都能用),版面结构更细但更慢,需 `[docling]` extra。

## 三、索引:文本和图片分两条独立的线

```
解析产物
   │
   ├──────────── 文本线 ──────────────
   │  切块(递归,500/800 token,overlap 60)
   │  可选:Contextual Retrieval 给每块加 LLM 前言
   │  可选:VLM 给嵌入图生成中文 caption 拼进块文本
   │  → 三种向量进【文本 collection】:
   │      • dense   (bge-m3)
   │      • BM25 英文 (fastembed)
   │      • BM25 中文 (jieba + Okapi)
   │
   └──────────── 图片线 ──────────────
      只对 独立图片(source_type=image)生效
      PDF/Office 里的嵌入图 不进 这条线
      → CLIP 图像向量 进【图片 collection】
```

> 两个 collection 完全独立:文本 collection 存文本三路向量,图片 collection 存 CLIP 向量。
> Schema 不匹配时(如切换 BM25-zh 开关)会**快速失败**提示 `mmrag reindex`,而非静默漏写。

## 四、检索:按模式走不同路

```
query 预处理(小写/纠错/同义词,均可开关)
   │
   ├─ mode=text (文→文,默认)
   │    三路并行: dense + BM25英 + BM25中  → RRF 融合 → 重排
   │
   ├─ mode=text-to-image (文→图)
   │    query → CLIP 文本向量 → 图片 collection 找近邻
   │
   ├─ mode=image-to-image (图→图)
   │    一张图 → CLIP 图像向量 → 图片 collection 找相似图
   │
   └─ mode=hybrid (混合)
        text 路必跑 + 传了 image_path 才带 image-to-image 路
        → RRF 融合 → 重排
```

**RRF 融合**(`retrieval.merge_hits`):把各路原始分统一到 rank 空间
`score = Σ weight/(RRF_K + rank)`,避免某路量纲(如 CLIP cosine)压死其他路。
各路原分仍保留在 `metadata.raw_score` 供重排器读取。

**两阶段重排**(可选,默认开):各路取候选 → RRF 融合 → bge-reranker
cross-encoder 对 `(query, evidence)` 精排 → `top_k`。重排器加载失败自动降级
单阶段。图像源 hit 不被文本重排器打分(保留 CLIP 分)。

## 五、回答:检索结果喂 LLM

```
top-k 证据
   │
   ├─ 拼 evidence(每条带 asset_id/title/source/page + 关联图 hint)
   ├─ 命中图片可作 base64 配图喂多模态 LLM(全局上限 4 张)
   ├─ reasoning 模型自动剥 <think/> 块(流式跨 chunk 缓冲剥)
   ├─ 无 LLM → 退化为证据摘要(前 3 hit 各 300 字)
   └─ 可流式(/chat/stream NDJSON 逐 token)
```

---

## 两条线对照

| | 文本线 | 图片线 |
| --- | --- | --- |
| **谁进** | PDF/Office/MD 的文字 + 图片里 OCR/描述出的文字 | 独立上传的图片 |
| **向量** | dense(bge-m3)+ BM25 英 + BM25 中 | CLIP |
| **存哪** | 文本 collection | 图片 collection |
| **检索模式** | `text` / `hybrid` | `text-to-image` / `image-to-image` |
| **PDF 内嵌图** | 经 VLM caption / OCR 转文字进这条线 | ❌ 不进 |

**核心思路**:图片有两条命运——

1. **独立图**走 CLIP 语义检索(文→图 / 图→图);
2. **嵌在文档里的图**通过 OCR / VLM caption 转成文字,走文本检索。

文本检索永远是主力(三路混合 + 重排),CLIP 是图片语义检索的补充通道。
默认全本地零外网(ollama 本地 embedding + 本地 OCR),只在主动配百度 token
或在线 LLM 时才联网。
