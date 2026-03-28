#!/usr/bin/env bash
set -euo pipefail

ROOT="/desay120T/ct/dev/uid01954/FlashOCC1.5"
cd "$ROOT"

CONFIG="projects/configs/flashocc/flashocc-r50.py"
CKPT="ckpts/flashocc-r50-256x704.pth"
PY=".venv/bin/python"
SCRIPT="tools/analysis_tools/benchmark_flashocc_precision.py"
GPU_ID="${GPU_ID:-7}"
OUT_DIR="work_dirs/precision_bench/full_eval_same_set"

mkdir -p "$OUT_DIR"

echo "[start] $(date '+%F %T %Z')"
echo "[info] config=$CONFIG"
echo "[info] ckpt=$CKPT"
echo "[info] gpu=$GPU_ID"
echo "[info] eval=full validation set, same setting for fp32/fp16/bf16"

for precision in fp32 fp16 bf16; do
  log_path="$OUT_DIR/${precision}_eval_full.log"
  echo "[run] $(date '+%F %T %Z') precision=$precision log=$log_path"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$PY" "$SCRIPT" "$CONFIG" "$CKPT" \
    --task eval \
    --precision "$precision" \
    --gpu-id 0 \
    --workers-per-gpu 0 \
    --samples-per-gpu 1 \
    > "$log_path" 2>&1
  echo "[done] $(date '+%F %T %Z') precision=$precision"
done

echo "[finish] $(date '+%F %T %Z')"
