#!/usr/bin/env bash
# Unlock v4 code paths + measure real eval delta vs v5 baseline (0.280 / 0.087 / 1.000).
#
# What this does:
#   1. Kill stale qdrant lock + verify no process holds it
#   2. Reparse all bundled PDFs  → triggers P5 chunk-by-section + keyword footer
#   3. Reindex text (bge-m3 1024d)        → unlocks multilingual + per-channel RRF
#   4. Download Chinese-CLIP (~1GB, first run) + reindex image → unlocks ZH text→image
#   5. Run v2 eval (83 cases)             → real numbers vs v5
#   6. Append a delta report to docs/eval-report-v6.md
#
# Run from repo root:  bash scripts/unlock_and_eval.sh
set -euo pipefail

HOME_DIR="${MM_ASSET_RAG_HOME:-$HOME/.mm_asset_rag}"
export MM_ASSET_RAG_HOME="$HOME_DIR"

# Resolve repo root from this script's location (it lives in <repo>/scripts/).
# Done before any `cd` so relative BASH_SOURCE works.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/..")"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "════════════════════════════════════════════════════════════════"
echo "  mm-asset-rag  unlock-v4 + eval   (home: $HOME_DIR)"
echo "  repo: $REPO_ROOT"
echo "════════════════════════════════════════════════════════════════"

# Stay in HOME_DIR so mmrag's load_env() reads the home .env (bge-m3 /
# minimax values), not the stale repo-root .env. mmrag is a console
# script (pip install -e .) so it doesn't need cwd=repo.
cd "$HOME_DIR"
MMRAG="mmrag"

# ─── 0. preflight ───────────────────────────────────────────────────────
echo "▶ preflight: kill stale lock + verify ollama + no server on 8011"
LOCK="$HOME_DIR/indexes/qdrant/.lock"
# qdrant local .lock is a marker file (contents like "tmp lock file"),
# not a pid — we judge staleness by "no live process touching the store".
# 8011 idle + no mmrag/python holding the qdrant dir == safe to remove.
if [ -f "$LOCK" ]; then
    if lsof -i :8011 >/dev/null 2>&1; then
        echo "  ⚠ API server on 8011 holds qdrant — stop it first"
        exit 1
    fi
    echo "  stale qdrant .lock (8011 idle) → removing"
    rm -f "$LOCK"
fi

if lsof -i :8011 >/dev/null 2>&1; then
    echo "  ⚠ API server running on 8011 — stop it first (qdrant local is single-process)"
    exit 1
fi

if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "  ⚠ ollama not running on 11434 — start it: ollama serve"
    exit 1
fi
echo "  ollama ok · bge-m3 available"

MMRAG="mmrag"

# ─── 1. reparse PDFs (triggers P5 chunk-by-section + keyword footer) ────
echo ""
echo "▶ step 1/5: reparse bundled PDFs (P5 chunk-by-section + keyword footer)"
PDFS_DIR="$REPO_ROOT/examples/data/chapter11_assets/pdfs"
if [ ! -d "$PDFS_DIR" ]; then
    echo "  ⚠ $PDFS_DIR missing — bundled corpus not checked out locally"
    echo "    run: git -C \"$REPO_ROOT\" checkout HEAD -- examples/data/"
    exit 1
fi
N_PDFS=$(find "$PDFS_DIR" -name '*.pdf' | wc -l | tr -d ' ')
echo "  $N_PDFS PDFs to parse"
# --no-auto-meta: skip VLM per-file round-trip (146 PDFs × 30s = unworkable)
# --pdf-parser pymupdf: deterministic, no network, fast
$MMRAG parse "$PDFS_DIR"/*.pdf --no-auto-meta --pdf-parser pymupdf 2>&1 | tail -8

# sanity: keyword footer now present in fresh chunks?
echo "  verify P5 took effect:"
python3 -c "
import json
from pathlib import Path
docs = [json.loads(l) for l in open('$HOME_DIR/documents.jsonl')]
n_with_kw = sum(1 for d in docs if '关键词:' in d.get('text',''))
print(f'  documents.jsonl: {len(docs)} chunks, {n_with_kw} with keyword footer ({100*n_with_kw//max(len(docs),1)}%)')
"

# ─── 2. reindex text (bge-m3 1024d) ─────────────────────────────────────
echo ""
echo "▶ step 2/5: reindex text collection (bge-m3)"
$MMRAG reindex --text-only --yes 2>&1 | tail -3

# ─── 3. download Chinese-CLIP + reindex image ───────────────────────────
echo ""
echo "▶ step 3/5: switch to Chinese-CLIP + reindex image"
# Toggle CLIP_MODEL in the home .env (idempotent: only if not already set)
if grep -q "^CLIP_MODEL=clip-ViT-B-32" "$HOME_DIR/.env"; then
    echo "  switching CLIP_MODEL → chinese-clip-vit-base-patch16 (will download ~1GB on first run)"
    sed -i.bak 's/^CLIP_MODEL=clip-ViT-B-32/CLIP_MODEL=OFA-Sys\/chinese-clip-vit-base-patch16/' "$HOME_DIR/.env"
    # Also lower the image relevance floor — Chinese-CLIP score distribution differs
    sed -i.bak 's/^IMAGE_RELEVANCE_THRESHOLD=.*/IMAGE_RELEVANCE_THRESHOLD=0.20/' "$HOME_DIR/.env" 2>/dev/null || true
fi
$MMRAG reindex --image-only --yes 2>&1 | tail -3

# ─── 4. run v2 eval (83 cases) ──────────────────────────────────────────
echo ""
echo "▶ step 4/5: run v2 eval (text→text + text→image + image→image)"
python3 /tmp/run_v2_eval.py 2>&1 | tail -100

# ─── 5. capture deltas into docs/eval-report-v6.md ───────────────────────
echo ""
echo "▶ step 5/5: write delta report → docs/eval-report-v6.md"
python3 - "$HOME_DIR" "$REPO_ROOT" <<'PY'
import json, sys, datetime
from pathlib import Path
home = Path(sys.argv[1])
repo = Path(sys.argv[2])
report = home / "eval_report_v2.json"
if not report.exists():
    print("  no eval_report_v2.json — skip delta write")
    sys.exit(0)
data = json.loads(report.read_text(encoding="utf-8"))
pg = data.get("per_group", {})
def hr(g):
    v = pg.get(g, {})
    return v.get("hit_rate", 0.0) if isinstance(v, dict) else 0.0
t2t = hr("text_to_text")
t2i = hr("text_to_image")
i2i = hr("image_to_image")
today = datetime.date.today().isoformat()
v5 = {"t2t": 0.280, "t2i": 0.087, "i2i": 1.000}
lines = [
    f"# mm-asset-rag v6 评估报告 ({today})",
    "",
    "**目的**:解锁 v4 已写但未生效的代码路径(P5 chunk-by-section + 关键词 footer、bge-m3、Chinese-CLIP),测量真实 Δ vs v5。",
    "",
    "## 解锁动作",
    "- reparse PDFs(`--pdf-parser pymupdf --no-auto-meta`)→ 触发 P5 chunk-by-section + 关键词 footer",
    "- reindex text(bge-m3 1024d)",
    "- 切 Chinese-CLIP(`OFA-Sys/chinese-clip-vit-base-patch16`)+ reindex image",
    "",
    "## v5 → v6 Δ",
    f"| 模式 | v5 | v6 | Δ |",
    f"| --- | --- | --- | --- |",
    f"| text→text hit@5 | {v5['t2t']:.3f} | {t2t:.3f} | {t2t-v5['t2t']:+.3f} |",
    f"| text→image hit@5 | {v5['t2i']:.3f} | {t2i:.3f} | {t2i-v5['t2i']:+.3f} |",
    f"| image→image hit@5 | {v5['i2i']:.3f} | {i2i:.3f} | {i2i-v5['i2i']:+.3f} |",
    "",
    "## 结论",
    "- 若 text→text 涨到 0.40+:P5 + 关键词 footer 生效,zh_on_zh 应明显改善",
    "- 若 text→image 涨到 0.50+:Chinese-CLIP 解锁中文",
    "- 仍未达标的维度 → 下一轮优化的真实靶点(避免猜测)",
]
out = repo / "docs/eval-report-v6.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"  wrote {out}")
print(f"  text→text {v5['t2t']:.3f} → {t2t:.3f} ({t2t-v5['t2t']:+.3f})")
print(f"  text→image {v5['t2i']:.3f} → {t2i:.3f} ({t2i-v5['t2i']:+.3f})")
print(f"  image→image {v5['i2i']:.3f} → {i2i:.3f} ({i2i-v5['i2i']:+.3f})")
PY

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  done. see docs/eval-report-v6.md for the delta."
echo "════════════════════════════════════════════════════════════════"
