#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "No python interpreter found in PATH or VIRTUAL_ENV." >&2
    exit 127
fi

PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
TORCHRUN_BIN=${TORCHRUN_BIN:-"$PYTHON_BIN_DIR/torchrun"}

NCCL_DEBUG=WARN \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
"$TORCHRUN_BIN" \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
