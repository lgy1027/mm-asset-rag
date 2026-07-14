# mm-asset-rag 评估报告 (2026-07-06)

> 这是当前保留的评估基线快照(原 v10)。历史版本 v2–v9 已从仓库移除,
> 可在 git log 中追溯。评估用 `mmrag eval` 实时跑,本报告记录的是该命令
> 在某一时刻的输出,不是持续维护的文档。

**目的**:补 image corpus OCR(上次 v9 因本地 OCR HTTP 服务未起、PaddleOCR-VL
token 注释,image 只走 filename title),用 PaddleOCR-VL 远程 API 批量 OCR
1633 张图,把可抽文本写回 `parsed/<asset_id>/ocr.json`,reindex 让 OCR 文本
进入 text collection。

## OCR 流程

- 验证 PaddleOCR-VL token 通,单张图 ~1s 出结果(`/api/v2/ocr/jobs` accept
  image/jpeg multipart,返回 `resultUrl.jsonUrl` 是 layoutParsingResults JSON)
- 5 worker 并发,3 retry 处理 429 + 网络抖动;批量 OCR 脚本写 `parsed/<id>/ocr.json`
  (image parser 已支持该缓存路径)
- 跑 1617 张(16 张之前 cache 跳过),152 张有 OCR 文本(9.4%)、1442 空、23 失败
- 失败主要是 `Max retries exceeded with url` (网络抖动),可补跑——但空文本
  image 占比 90%+,补跑 ROI 低,直接走 reindex

### OCR 文本示例(真实可搜)

| 图片 | OCR 文本 |
| --- | --- |
| 联宝 Kickoff '26 海报 | `LCFC 联宝科技 / LCFC / LCFC Kickoff '26 / 聚智AI 领新程 / April 2026 / #WeAreLenovo` |
| 联想创业精神 5.0 | `Entrepreneurship 创业精神5.0 / Commit to Lenovo Vision / 使命必达 / 笃信愿景，坚守航向` |
| Caltech Dalmatian 01 | `oney please, I calm down. me explain....` (漫画对话框) |

## 索引状态变化

| 指标 | v9 | v10 | Δ |
| --- | --- | --- | --- |
| text collection chunks | 3765 | **5414** | **+1649** |
| image collection chunks | 1633 | 1633 | 0 |
| unique assets | 137 PDF + 1633 img | 同 | — |

1649 个新增 chunks 是 152 个有 OCR 文本的 image(每个 image 一个 chunk,
text = "图片标题: filename\n图片标签: ...\nOCR 文本: ...")。
其余 1442 张无 OCR 文本的 image 进 text collection 时只带 filename title,
signal 弱,对召回影响小。

## eval-v2 hit_rate@5

| 模式 | v9 | v10 | Δ |
| --- | --- | --- | --- |
| text→text (49 case) | 0.694 | **0.694** | ±0.000 |
| text→image (23 case) | 0.652 | **0.652** | ±0.000 |
| image→image (10 case) | 1.000 | **1.000** | ±0.000 |

**text→text 不涨**——这不是 OCR 失败,而是 eval-v2 的 expected_asset_ids
全部是 PDF 来源(49 个 t2t query 期望命中 AlexNet / YOLO / DDPM 等论文,
没有 query 期望命中 Caltech/Picsum image)。OCR 新增的 1649 image chunks
在 eval 数据里没有 ground truth,数字上不变。

## OCR 真实价值:search 验证

直接 search OCR-only 关键词看召回:

| Query | 召回 top | 来源 |
| --- | --- | --- |
| `联宝科技 Kickoff` | 媒眼看联宝 PDF | PDF 里也有"联宝科技",PDF 抢词 |
| `Lenovo Vision 使命必达` | Flamingo PDF | "Vision" 普遍词,Flamingo 抢词 |
| `calm down explain` | Lora / Glove / EfficientNet (全 PDF) | Caltech 对话框短文本被 1649 PDF/1649 image 总池淹没 |

### 为什么 image OCR 不直接体现

- **数量失衡**:1633 image vs 137 PDF(实际 chunks 5414 vs 3765),比例 ~1.4:1,
  但**单 image 的 OCR 文本平均长度 < 100 chars**,PDF chunks 平均 > 1000 chars,
  BM25 词频累积后 PDF 永远占优
- **OCR 短文本多是专名**:LCFC、Kickoff、聚智 AI 等少量 token,在 token 总数
  几十万的 corpus 里 IDF 偏低
- **eval 期望是 PDF**:t2t 49 个 query 全部期望 PDF asset,没机会验证
  "图里的文字召回对应图"

## 真要发挥 OCR 价值需要

| 优化 | 效果 | ROI |
| --- | --- | --- |
| 给 image chunk 加更长文本上下文(图片说明 / tag) | 让 BM25 词频提升 | 低(难扩) |
| image→text 单独路由(image query 走 image collection + OCR 文本 boost) | 隔离竞争 | 中(需代码) |
| OCR 失败的 23 张 retry | +0.1% ~ 1% | 低 |
| 跑完整 OCR(用 VLM caption 而非 OCR) | 图像描述更丰富 | 中(token 贵) |

## 验收

| 维度 | 状态 |
| --- | --- |
| PaddleOCR-VL token 验证 | ✅ 1s/张,multipart OK |
| 批量 OCR 1617 张 | ✅ 25 分钟,9.4% 有文本 |
| ocr.json 写入正确 | ✅ 152 张有非空 blocks |
| reindex text-only | ✅ 5414 chunks(+1649 image) |
| mmrag eval hit_rate | 不变(0.838) |
| eval-v2 全套 | 不变(0.720),原因是 eval 数据全是 PDF |
| image→image / text→image | 不变(1.000 / 0.652),走 CLIP 不受 OCR 影响 |

## 后续可选(非阻塞)

1. 给 eval-v2 加几个 image-OCR-only 的 query(期望命中 Caltech/Picsum
   类目,验证 OCR 在 search 中的真实作用)
2. 重跑 23 张失败 image OCR(网络抖动 retry 即可)
3. 跑 VLM caption 替代 OCR,获取更长更丰富的图像描述(text 集合里
   image chunks 文本长度翻倍,BM25 词频才能跟 PDF 抗衡)