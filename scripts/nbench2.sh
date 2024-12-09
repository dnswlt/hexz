#!/bin/bash

# Validate the two explicitly specified checkpoints against each other.

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <checkpoint_1> <checkpoint_2>"
    exit 1
fi

cd $(dirname $0)/..

model_name=res10
device=cuda
iterations=3200
num_games=20
model_repo="$HOME/tmp/hexz-models"

go run ./cmd/nbench2 \
    -model-repo "$model_repo" \
    -key1 "$model_name:$1" \
    -key2 "$model_name:$2" \
    -games $num_games \
    -both-sides \
    -iterations $iterations \
    -device "$device" \
    -stats-file "./stats/nbench.jsonl"
