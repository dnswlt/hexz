#!/bin/bash

# Validate two checkpoints against each other on a fixed, paired position set.

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <checkpoint_1> <checkpoint_2>"
    exit 1
fi

cd $(dirname $0)/..

model_name=res10
device=cuda
if [[ "$(uname)" = "Darwin" ]]; then
    device=mps
fi
iterations=3200
concurrency="${HEXZ_NBENCH_CONCURRENCY:-64}"
model_repo="${HEXZ_MODEL_REPO_BASE_DIR:-/home/dw/data/hexz-models}"
positions_file="${HEXZ_NBENCH_POSITIONS_FILE:-testdata/nbench/flagz_initial_v1.jsonl}"

go run ./cmd/nbench2 \
    -model-repo "$model_repo" \
    -key1 "$model_name:$1" \
    -key2 "$model_name:$2" \
    -positions-file "$positions_file" \
    -games 0 \
    -concurrency "$concurrency" \
    -both-sides \
    -iterations "$iterations" \
    -device "$device" \
    -stats-file "./stats/nbench.jsonl"
