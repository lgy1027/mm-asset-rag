#!/usr/bin/env bash
# Contextual Retrieval 小规模试点:对 10 个中文 PDF(联宝/Codex/Obsidian,~208 chunks)
# 生成 LLM context,跑 eval 看是否 > v6-corrected 基线 0.673。
#
# 流程:
#   1. 清这 10 个 asset 的 parsed/ 缓存(否则 _do_parse 跳过)
#   2. 从 documents.jsonl 删这 10 个 asset 的旧 chunk(否则重复)
#   3. 重新 parse(带 --contextual,调 MiniMax-M3 生成 context)
#   4. reindex text(让 context 前缀进 dense + BM25)
#   5. 跑 v2 eval,对比 0.673
set -euo pipefail

HOME_DIR="${MM_ASSET_RAG_HOME:-$HOME/.mm_asset_rag}"
export MM_ASSET_RAG_HOME="$HOME_DIR"

# Resolve repo root from this script's absolute path (lives in <repo>/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/..")"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PDFS_DIR="$REPO_ROOT/examples/data/chapter11_assets/pdfs"

echo "════════════════════════════════════════════════════════════════"
echo "  contextual retrieval 试点 (10 中文 PDF, ~208 chunks)"
echo "════════════════════════════════════════════════════════════════"

# 清 stale qdrant lock(8011 空闲时安全)
LOCK="$HOME_DIR/indexes/qdrant/.lock"
if [ -f "$LOCK" ] && ! lsof -i :8011 >/dev/null 2>&1; then
    echo "  清 stale qdrant .lock"
    rm -f "$LOCK"
fi

# 10 个中文 PDF 的 asset_id 前缀(去 hash)
PREFIXES=(
  "CES 2026再绽光芒"
  "Obsidian 的 10 大 AI Skill"
  "创新联宝 会发电的键盘"
  "创新联宝 联宝科技中试基地"
  "受邀参加合肥"
  "媒眼看联宝 \"一台笔记本\""
  "媒眼看联宝 安徽外贸"
  "所有深度用 AI 编程"
  "敢AI敢为 志在必行"
  "责任联宝 ESG"
)

# 1. 清 parsed/ 缓存 + 从 documents.jsonl 删旧 chunk
echo "▶ step 1/5: 清试点 asset 缓存 + 旧 chunk"
python3 - "$HOME_DIR" "${PREFIXES[@]}" <<'PY'
import json, sys, shutil
from pathlib import Path
home = Path(sys.argv[1])
prefixes = sys.argv[2:]
parsed_dir = home / "parsed"
docs_path = home / "documents.jsonl"

# 找出要删的 asset_id 全集(从 asset_index)
idx = [json.loads(l) for l in open(home / "asset_index.jsonl")]
target_aids = set()
for r in idx:
    aid = r.get("asset_id","")
    title = aid.rsplit("_",1)[0] if "_" in aid else aid
    for p in prefixes:
        if title.startswith(p) or p.startswith(title[:len(p)]):
            target_aids.add(aid)
print(f"  target assets: {len(target_aids)}")

# 清 parsed/<id>/
cleared = 0
for aid in target_aids:
    d = parsed_dir / aid
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        cleared += 1
print(f"  cleared parsed/ dirs: {cleared}")

# 从 documents.jsonl 删这些 asset 的 chunk
if docs_path.exists():
    kept = 0
    removed = 0
    lines = docs_path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        aid = d.get("metadata",{}).get("asset_id","")
        if aid in target_aids:
            removed += 1
        else:
            out.append(line)
            kept += 1
    docs_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    print(f"  documents.jsonl: removed {removed} chunks, kept {kept}")
PY

# 2. 重新 parse(带 --contextual)
echo ""
echo "▶ step 2/5: reparse with --contextual (调 MiniMax-M3, ~5-10 min)"
# 收集目标 PDF 文件
PDF_FILES=()
for f in "$PDFS_DIR"/*.pdf; do
    base="$(basename "$f" .pdf)"
    title="${base%_*}"  # 去 _hash
    for p in "${PREFIXES[@]}"; do
        if [[ "$title" == "$p"* ]]; then
            PDF_FILES+=("$f")
            break
        fi
    done
done
echo "  target PDFs: ${#PDF_FILES[@]}"
cd "$REPO_ROOT"
mmrag parse "${PDF_FILES[@]}" --no-auto-meta --pdf-parser pymupdf --contextual 2>&1 | tail -10

# 3. 验证 context 生成
echo ""
echo "▶ step 3/5: 验证 context 已写入 documents.jsonl"
python3 - "$HOME_DIR" "${PREFIXES[@]}" <<'PY'
import json, sys
from pathlib import Path
home = Path(sys.argv[1])
prefixes = sys.argv[2:]
docs = [json.loads(l) for l in open(home / "documents.jsonl")]
target = [d for d in docs if any(d["metadata"].get("asset_id","").startswith(p) or p.startswith(d["metadata"].get("asset_id","")[:len(p)]) for p in prefixes)]
with_ctx = sum(1 for d in target if d["metadata"].get("context"))
print(f"  试点 chunks: {len(target)}, 带 context: {with_ctx} ({100*with_ctx//max(len(target),1)}%)")
if target:
    print(f"  样本 context: {target[0]['metadata'].get('context','(无)')[:120]}")
PY

# 4. reindex text
echo ""
echo "▶ step 4/5: reindex text (context 前缀进索引)"
cd "$HOME_DIR"
mmrag reindex --text-only --yes 2>&1 | tail -3

# 5. 跑 eval
echo ""
echo "▶ step 5/5: 跑 v2 eval 对比 v6-corrected 基线 0.673"
python3 /tmp/run_v2_eval.py 2>&1 | grep -E "===|hit_rate" | head -10

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  试点完成。若 text→text > 0.673 → contextual 有效,可全量 4158 chunk"
echo "════════════════════════════════════════════════════════════════"
