# FAQ & 故障排查

> 面向刚装上 `mm-asset-rag` 的用户,也作为日常排错速查。
>
> 新发现的坑 → 先按表对照 → 仍无解再开 issue,并附上 `/health?deep=true` 输出 + `$MM_ASSET_RAG_HOME/tasks.jsonl` 最近一条 task。

## 目录

- [安装 / 环境](#安装--环境)
- [启动 / 配置](#启动--配置)
- [上传 / 解析](#上传--解析)
- [检索 / 评估](#检索--评估)
- [macOS 特别坑](#macos-特别坑)
- [架构 / 调试](#架构--调试)

---

## 安装 / 环境

### `pip install mm-asset-rag` 装不上 / 装错版本

确认 Python 在 3.10 / 3.11 / 3.12 任一档:

```bash
python --version    # 必须 ≥ 3.10
```

PyPI 包名就是 `mm-asset-rag`(中划线,不是下划线):

```bash
pip install mm-asset-rag          # 只装 core(text + FastAPI)
pip install "mm-asset-rag[clip]"  # + sentence-transformers CLIP(推荐,支持 text→image / image→image)
pip install "mm-asset-rag[docling]"  # + docling 文档解析(更准但重)
pip install "mm-asset-rag[ocr]"   # + 本地 PP-OCRv6(图片 OCR 零外网)
pip install "mm-asset-rag[cn_clip]"  # + Chinese-CLIP(中文 zero-shot)
```

可叠加,例如 `[clip,ocr]`。

### 用 uv 但漏了某个 extra,运行时报 `ModuleNotFoundError`

`uv sync --extra dev` 只装 dev。运行 `mmrag-api` 时需要哪些 extra 就 `--extra` 一次声明完整:

```bash
uv sync --extra dev --extra clip --extra ocr
uv run mmrag-api
```

### 安装时 wheel 编译失败 / `PyMuPDF` / `onnxruntime` 找不到

这俩是预编译 wheel,理论上 3.10+ 都直接有。如果卡编译,通常是你的 Python 是发行版修改版(如 conda-forge 的 `python==3.10.x`),或平台太老(比如 macOS 10.13)。建议改用官方 python.org installer 或 conda 的 `defaults` channel。

---

## 启动 / 配置

### 启动后访问 `http://127.0.0.1:8011/` 是空白页 / 404

先确认服务真起来了:

```bash
curl http://127.0.0.1:8011/health
# 期望:{"status":"ok",...}
```

如果是端口占用,改 `MM_ASSET_RAG_PORT`(在 `.env` 里设)再启。

### `/health` 说 `embedder_configured=false`,但我已经设了 `OPENAI_API_KEY`

`OPENAI_*` 是给 LLM 用的,embedder 默认走 OpenAI-compatible `/v1/embeddings` 端点 — 但需要设的是 `EMBEDDING_*` 系列(见 `.env.example`):

```env
EMBEDDING_BASE_URL=https://api.openai.com/v1   # 或 ollama / 你的代理
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
```

设错了 `OPENAI_*` 只会让 `/answer` / `/chat` 工作,`/search` 检索直接报 `embedder not configured`。

### `.env` 改了不生效

`.env` 在 cwd 被 `config.load_env()` 加载。如果是从 IDE 启动,IDE 可能改了 cwd。检查:

```bash
# 在 mmrag-api 进程的 cwd 下跑
cat .env
```

CI / 测试场景别靠 `.env`,用 `monkeypatch.setenv` 或显式 env。

### 想跑单元测试,但本机 `.env` 干扰

直接跑就行 — `tests/conftest.py` 有 autouse fixture `_isolate_env_file` 把 `.env` 屏蔽、清单例缓存,跑 `pytest` 不用先 `mv .env`。真要覆盖某个值:

```python
def test_x(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    ...
```

### `QDRANT_URL` 怎么选?local file 还是 server?

- **local file**(默认):`MM_ASSET_RAG_HOME/indexes/qdrant/` 单文件,**单进程锁**。适合个人 / 单机 demo。
- **server**:`QDRANT_URL=http://localhost:6333` 走 `docker run qdrant/qdrant`。多进程并发、跨机访问、想要持久化备份时用。

> ⚠️ local-file 单进程锁:`mmrag-api` 在跑时,另一个终端跑 `mmrag reindex` 会报 `storage already accessed`。要么先停 API,要么切 server 模式。

---

## 上传 / 解析

### 上传 PDF 后 web UI 没动静 / 进度卡 0%

后台任务异步跑。打开 `$MM_ASSET_RAG_HOME/tasks.db` 看 `status` / `current`:

```bash
sqlite3 ~/.mm_asset_rag/tasks.db "select id, status, current, error from task where id='<task_id>'"
```

或 CLI:`mmrag tasks`(列出最近的任务)。

### 解析报 `KeyError: parser ('pdf', 'auto') not registered`

只会在装好老 wheel 又升级但没重启时出现。重启 `mmrag-api` 即可,或在代码里调一次 `from mm_asset_rag.parsers import _AutoPdfParser`。

### VLM 抽 title 把文件名覆盖了,eval 被随机性污染

`AUTO_META_ENABLED=true`(默认)时 VLM 会给文件猜 title / description / tags,**覆盖** 你上传时填的 title。这会让 `asset_id` 变成 hash(基于 VLM 改后的 title),eval cases 写的预期 id 就对不上。

正式跑 eval 时关掉:

```env
AUTO_META_ENABLED=false
```

或者命令行:`mmrag parse --no-auto-meta ./corpus/*.pdf`。

### 扫描 PDF 一片白(没字) / OCR 不出文字

扫描 PDF 自动路由:
- 有 `PADDLEOCR_VL_API_TOKEN` → 走 PaddleOCR-VL 在线
- 没 token → 本地 PP-OCRv6(`[ocr]` extra,需 `uv sync --extra ocr`)

如果 token 设了还是白,直接:

```bash
mmrag parse --pdf-parser paddleocr_vl ./paper.pdf
```

确认走的是 API 路径。本地 PP-OCRv6 跑扫描 PDF 较慢(每页 1-2s),几百页文档耐心等。

### 图片 OCR 走了外网 / 我想要纯本地

`.env`:

```env
OCR_BACKEND=local     # 默认就走 PP-OCRv6 + onnxruntime
# OCR_BACKEND=http   # 切到 HTTP 自建后端,需要 OCR_HTTP_URL / OCR_HTTP_TOKEN
```

`local` 模式零外网,模型打包在 `rapidocr` wheel 里,首次启动不下载。

### 上传中文 docx/pptx,内容是乱码 / 丢了表格

默认 MarkItDown 后端,对中文是 UTF-8 直通 — 如果乱码,通常是文件本身就乱码(老 WPS / GBK)。可以试试 docling 后端:

```env
DOCUMENT_PARSER=docling
```

docling 装上慢(~500 MB),但对复杂 docx / pptx / 含公式表格的 PDF 更稳。

### 文档里的内嵌图被加进 image collection,污染了 text→text 召回

这是 0.2.0 修复:文档图默认 **不进** CLIP image collection(`IMAGE_CAPTION_ENABLED=false` 时),只走文本化 caption 喂 BM25。

如果你的语料里 docx/pptx 的图特别多、又确实想按图搜,显式开:

```env
IMAGE_CAPTION_ENABLED=true   # VLM 给文档图打 caption,文本 + 进 image collection
```

> ⚠️ 开 caption 后,每个图走一次 VLM,几百图耗时长,且 reasoning 模型(如 QwQ)会在 caption 里塞 `<think>...</think>`,需要后处理。

---

## 检索 / 评估

### `mmrag search` 跑出 0 结果,但 corpus 里有相关内容

分四步查:

1. **检查 collection 是否有数据**:`mmrag reindex --dry-run` 或直接 `curl http://127.0.0.1:6333/collections/multimodal_text_1024d`(端口来自 QDRANT_URL 或 local file)。
2. **看 route**:`--mode text` 走 dense + bm25 + bm25_zh 三路 RRF;`--mode text-to-image` 走 CLIP。
3. **调 threshold**:`MIN_SCORE`(默认 0.30)。如果召回太严,临时降到 0.10 看是否能回结果,再反向调高。
4. **看 embedding 维度**:切过 `EMBEDDING_MODEL` / 加过 `[cn_clip]` 后,collection 后缀变了 — 老索引没用了,必须 `mmrag reindex`。

### text→image 检索全是英文图,中文 query 没结果

默认 `clip-ViT-B-32`(OpenAI 原版,英文)。中文 query 必须切 Chinese-CLIP:

```env
IMAGE_PROVIDER=cn_clip
CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16   # 768d
```

注意:
1. 需要 `[cn_clip]` extra(`uv sync --extra cn_clip`)或 `[docling]` 间接拉入 `transformers`。
2. image collection 后缀变 `_768d`,需要 `mmrag reindex` drop + rebuild。
3. 首跑会下 ~1 GB 模型,macOS 系统代理坑见下节。

### eval 跑了但 `hit_rate=0`

**几乎一定是 asset_id 不匹配**:
- eval case 写 `expected_asset_ids: ["Alexnet"]`,实际 corpus 里的 asset_id 是 `Alexnet_<8-hex-hash>`。
- `evaluation_v2` 已经做了 trailing hash 剥离;`evaluation`(v1)没有。`evaluation` v1 仍跑过的,可以手动在 case 里把 `Alexnet` 写成 `Alexnet_<hash>`。

**第二种可能**:corpus 没 ingest。`mmrag eval` 不带 ingest 步骤,先把语料喂进 index:

```bash
mmrag parse ./my_eval_corpus/*.pdf
mmrag eval --cases my_cases.json
```

**第三种可能**:embedding dim / collection 没对上。`/health?deep=true` 看 `collection_name`。

### `mmrag reindex` 报 `missing sparse vectors: ['bm25_zh']`

text collection schema 不匹配 — 之前没开 BM25-zh,现在开了。处理:

```bash
mmrag reindex     # 默认 drop + rebuild
```

或者彻底关:`BM25_ZH_ENABLED=false`。

### `mmrag reindex` 报 `storage already accessed`

local-file Qdrant 单进程锁。先停 `mmrag-api`,再 reindex;或切 `QDRANT_URL` 走 server。

---

## macOS 特别坑

### `mmrag-api` 启动后整个进程卡死,什么都不响应

**几乎肯定是 macOS 系统代理 + HuggingFace Hub 卡死。**

症状:`sentence-transformers` 或 `transformers` 首次启动时尝试连 HF Hub 拉模型,在 macOS「网络偏好设置 → 代理」里被劫持,hang 死整个进程(包括 `/health`)。

解决(任选):

```bash
# 方案 A:提前下模型,设 offline
huggingface-cli download sentence-transformers/clip-ViT-B-32
export HF_HUB_OFFLINE=1
mmrag-api

# 方案 B:取消 macOS 系统代理(网络偏好设置 → 取消勾选)
# 然后正常启动,模型下次启动会缓存

# 方案 C:让 HF 走镜像(国内)
export HF_ENDPOINT=https://hf-mirror.com
```

### ollama 在 macOS 上跑 `bge-m3` 慢 / OOM

`bge-m3` 是 1.5 GB 模型,M 系列 Mac 上 MPS 后端偶尔抽风。临时切 CPU:

```bash
OLLAMA_NUM_GPU=0 ollama serve
```

或换更小的 `nomic-embed-text`(137M,256d,够用)。

### `pip install` 装 `onnxruntime` 报 architecture 错

Apple Silicon 上装 x86_64 wheel 会失败。强制 arm64:

```bash
python -m pip install --upgrade --force-reinstall onnxruntime
```

---

## 架构 / 调试

### 我加了一个新 modality(audio),要怎么注册

详见 README 末「Adding a new modality」节。三步:
1. `parsers/audio_parser.py` 写 `class AudioParser:` 实现 `protocols.Parser`。
2. `parsers/__init__.py` 末尾 `register_parser(AudioParser())`。
3. `embedders/audio_embedder.py` + `register_embedder(...)`。

FastAPI、CLI、Qdrant backend 都从 registry 读,不动 dispatch。

### `/chat/stream` 用 reasoning 模型,前端看到一堆 `<think>...</think>`

已经自动 strip。0.2.0 P1 加的:跨 chunk 边界识别 `<think>...</think>`,reasoning token 不下发给客户端。

如果你**希望** reasoning 可见,在 `.env` 设 `CHAT_INCLUDE_REASONING=true`(0.2.0 之后会加)。

### 怎么把 `/health?deep=true` 输出粘 issue 里

```bash
curl -s 'http://127.0.0.1:8011/health?deep=true' | python -m json.tool
```

期望字段:`status`、`embedder_configured`、`llm_configured`、`collection`、`point_count`。

### 怎么把 task 历史清掉 / 重置整个 home

```bash
mmrag reset          # 0.2.0 加的(若存在);或:
rm -rf ~/.mm_asset_rag    # 一切清零,下次启动自动重建
```

### 任务卡 `running` 但进程已死 / `interrupted` 状态异常

启动时自动 recover:daemon 进程 crash 后还在 `running` 的 task 会被标 `interrupted`。重跑:

```bash
mmrag retry <task-id>
```

### `pytest` 跑挂,提示 import 失败

`uv run pytest` 前缀必须带 — `.venv` 才有 `transformers` / `sentence-transformers` 等可选依赖。全局 `pip install -e .` 已被项目显式 uninstall,跑测试必须走 `uv run`。

### coverage 太低,有几个文件 hit 不到

`parsers/pdf_parser.py` 53% / `backends/__init__.py` 51% 是已知的:scan fallback / 远端 backend 路径需要真实 PDF 或联网。整体 80%,足以反映主流程覆盖。再压要重型 fixture,性价比低。

### 在哪看 commit 风格 / PR 模板

- commit:`fix(qdrant): ...` 中英混用,**不带** `Co-Authored-By: Claude`,**不带** `Generated with [tool]`。
- PR:`CONTRIBUTING.md` 末的 PR 模板。
- changelog:`CHANGELOG.md` Keep-a-Changelog 格式,新条目塞 `[Unreleased]` 块。