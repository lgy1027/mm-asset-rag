# 音视频检索设计方案(Phase 1: 文本侧)

> 目标:支持音频(会议录音/播客)与视频(培训/录屏/监控)资产的语义检索,
> **不改动现有检索核心**。协议层(`core/protocols.py`)早已预留
> `source_type="audio"|"video"` 与 `modality="audio"|"video_frame"`,
> 本方案是填实现,不是改架构。

## 一、核心决策(已定)

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 路线 | **Phase 1 纯文本侧** | ASR/字幕/caption 全部转成带时间戳的 `ParsedChunk` 进现有 text 索引;embedding、BM25、rewrite、rerank、Langfuse 追踪零改动 |
| ASR | **本地 FunASR**(`[asr]` 可选 extra) | 中文场景 Paraformer/SenseVoice 优于 whisper;词级时间戳利于定位;ingest 是离线批量任务,无查询路径延迟敏感;复用 `[ocr]` 的"可选 extra + 静默降级"约定 |
| 视频取文本 | **三级级联** | 内嵌字幕(成本最低) → 无字幕抽音轨走 ASR → 关键帧 VLM caption 补盲。与 PDF 已有的"文本层 → OCR → VLM"降级哲学一致 |
| 远端 ASR | 不选 | edgefn 中转已暴露高延迟(rewrite 单次 9.9s),重负载不宜再压上去 |

## 二、市面调研结论(2026-09)

| 项目 | 做法 | 对我们的启示 |
| --- | --- | --- |
| [VideoContext-Engine](https://github.com/dolphin-creator/VideoContext-Engine) | 场景检测 + Whisper ASR + Qwen3-VL,每场景输出转写+视觉描述+标签的"RAG-Friendly"结构化文档 | 与我们的三级级联设计同构,路线被验证 |
| [deepseek-v4-flash-vision-video-rag](https://github.com/liangdabiao/deepseek-v4-flash-vision-video-rag) | ffmpeg 抽帧 → VLM 逐帧读成时间轴卡片 → 粗筛+精排+深读,带 `[MM:SS]` 回答 | 同厂商栈参考;但**纯视觉无音频模态,音轨被忽略**——反向证明 ASR 级不可省 |
| [HKUDS/VideoRAG](https://github.com/HKUDS/VideoRAG)(KDD'26) | 图谱驱动知识索引,面向数百小时长视频理解 | 学术重方案,企业资产检索属过度设计;确认"时间戳场景索引"是共识 |
| [video-rag-bot](https://github.com/di37/video-rag-bot) | CLIP 帧向量 + 向量库 | 原生多模态向量属以图搜视频的补充路径(Phase 3 备选) |

**共识**:转写 + 字幕 + 画面 caption 进文本索引是事实标准;没有任何主流
方案把原生音频向量(CLAP 类)当主路径(仅音乐/哼唱搜索 niche)。

## 三、数据流设计

```
音频文件(mp3/wav/m4a/flac)
   │
   └─ ffmpeg 规范化(统一 16k 单声道)→ FunASR → 词级时间戳
        │
        └─ 按时间窗(默认 30s)切带 start/end 的 ParsedChunk ──┐
                                                            │
视频文件(mp4/mkv/mov/webm)                                   │
   │                                                          │
   ├─ ① ffprobe 探测内嵌字幕流 ──有──> 字幕解析(srt/vtt/ass/mkv)│
   │                                    │                      │
   │  ② 无字幕:ffmpeg 抽音轨 ──────────┘ (复用上面的 ASR 管线)│
   │                                                          │
   └─ ③ 场景切分抽关键帧(每场景 1-5 帧)→ VLM caption          │
        (复用 image_caption 的 OpenAI 兼容调用模式)            │
        │                                                      │
        └─ 带时间戳的 ParsedChunk ─────────────────────────────┘
                                  │
                                  ▼
                     现有 text 索引(dense + BM25,零改动)
```

- 三级产出统一为 `ParsedChunk(text=转写/字幕/caption, metadata={start, end, ...})`;
  `metadata` 本就 modality-neutral,检索/重排/回答链路完全无感
- 检索命中后,`start/end` 支持回放到"第几分几秒";`SearchHit` 仅加约定字段,不改语义

## 四、任务拆分(全部"纯新增 + 注册",零回归)

| # | 任务 | 改动点 | 验收标准 |
| --- | --- | --- | --- |
| 1 | sniff 识别音视频 | `ingest/sniff.py` 加 magic bytes(mp3/wav/mp4/webm/mkv/mov)与扩展名表 | 上传音/视频文件识别为 `audio`/`video`;pdf/image 行为不变 |
| 2 | ffmpeg 前置检查 | ingest 前置探测,缺失则该资产标记失败原因(不 raise) | 无 ffmpeg 环境下其余资产 ingest 不受影响 |
| 3 | 音频 parser | 新文件 `parsers/audio_parser.py`,`source_type="audio"`,注册 `(audio, funasr)`;`[asr]` extra | 一段中文录音转写为带时间戳 chunks;不配 extra 时优雅降级 |
| 4 | 视频 parser | 新文件 `parsers/video_parser.py`,`source_type="video"`,注册 `(video, ffmpeg)`:字幕解析 → 音轨 ASR(复用任务 3)→ 抽帧 VLM caption | 带字幕视频走①;去字幕视频走②;画面-only 内容有 caption 覆盖 |
| 5 | 检索后链路 | `SearchHit.metadata` 时间戳约定 + web UI 回放定位 + answer 引用带 `[MM:SS]` | 命中音/视频 chunk 可定位播放 |
| 6 | 评测 | 音频/视频语料 + qrels cases(含负例) | recall@5/MRR 可对比 |

依赖顺序:1 → 2 → 3 → 4(依赖 3)→ 5、6 可并行。

## 五、解耦与风险

- **解耦铁律**:新能力只走 registry 注册(`register_parser` / `register_embedder`),
  应用代码不 import 具体实现;不配 `[asr]` / ffmpeg 时对应 `source_type`
  的 ingest 降级报错,其余类型零影响
- **性能**:ASR/VLM 是计算密集,长视频 ingest 耗时显著上升——`INGESTION_PROFILE`
  分级语义天然适配(precision 才开 VLM caption,fast 仅字幕/ASR)
- **Langfuse**:新 parser 在 `ingest.task` 下自动产生子 span,无需插桩
- **明确的非目标**(不在本期):原生音频向量(CLAP)、以图搜视频、实时音频流;
  如需再加 Phase 2/3 设计文档

## 六、参考

- 项目扩展约定:[README「Adding a new modality」](../README.md)
- 同构参考实现:`mm_asset_rag/ingest/image_caption.py`(OpenAI 兼容调用 + 静默降级)
- PDF 降级哲学:`docs/configuration.md`「Scanned-PDF fallback (auto parser)」
