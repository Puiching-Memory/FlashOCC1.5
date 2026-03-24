#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-0}
if [ "$PORT" -eq 0 ]; then
    PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-$(dirname "$(command -v "$PYTHON_BIN")")/torchrun}

NCCL_DEBUG=WARN \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
"$TORCHRUN_BIN" \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
