#!/bin/bash
set -euo pipefail

# Validate two checkpoints against each other on a fixed, paired position set.
# Two CUDA cpuserver containers are started so the Go coordinator can run games
# concurrently without requiring a host C++ build.

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <model-or-checkpoint-1> <model-or-checkpoint-2>"
    echo "Examples: $0 62 63"
    echo "          $0 res10:63 res10-r4-cp63:10"
    exit 1
fi

cd "$(dirname "$0")/.."

model_name="${HEXZ_MODEL_NAME:-res10}"
iterations="${HEXZ_NBENCH_ITERATIONS:-3200}"
games="${HEXZ_NBENCH_GAMES:-0}"
concurrency="${HEXZ_NBENCH_CONCURRENCY:-64}"
model_repo="${HEXZ_MODEL_REPO_BASE_DIR:-/home/dw/data/hexz-models}"
positions_file="${HEXZ_NBENCH_POSITIONS_FILE:-testdata/nbench/flagz_initial_v1.jsonl}"
stats_file="${HEXZ_NBENCH_STATS_FILE:-./stats/nbench.jsonl}"
image="${HEXZ_CPUSERVER_IMAGE:-hexz-cpuserver-cuda:latest}"
p1_port="${HEXZ_NBENCH_P1_PORT:-50171}"
p2_port="${HEXZ_NBENCH_P2_PORT:-50172}"
p1_container="hexz-nbench2-p1"
p2_container="hexz-nbench2-p2"

parse_model_key() {
    local value="$1"
    if [[ "$value" == *:* ]]; then
        printf '%s %s\n' "${value%:*}" "${value##*:}"
    else
        printf '%s %s\n' "$model_name" "$value"
    fi
}

read -r p1_model p1_checkpoint < <(parse_model_key "$1")
read -r p2_model p2_checkpoint < <(parse_model_key "$2")
p1_key="$p1_model:$p1_checkpoint"
p2_key="$p2_model:$p2_checkpoint"

cleanup() {
    docker stop "$p1_container" "$p2_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

start_server() {
    local player="$1"
    local model="$2"
    local checkpoint="$3"
    local port="$4"
    local container="$5"
    local model_path="/models/models/flagz/$model/checkpoints/$checkpoint/scriptmodule.pt"
    echo "Starting $player server for $model:$checkpoint on port $port"
    docker run --rm -d \
        --name "$container" \
        -p "$port:$port" \
        -v "$model_repo:/models:ro" \
        --gpus all \
        "$image" \
        --device=cuda \
        --max_think_time_ms=0 \
        --model_path="$model_path" \
        --model_key="$model:$checkpoint" \
        --server_addr="0.0.0.0:$port" >/dev/null
}

start_server p1 "$p1_model" "$p1_checkpoint" "$p1_port" "$p1_container"
start_server p2 "$p2_model" "$p2_checkpoint" "$p2_port" "$p2_container"

sleep "${HEXZ_NBENCH_STARTUP_SECONDS:-3}"

go run ./cmd/nbench2 \
    -model-repo "$model_repo" \
    -key1 "$p1_key" \
    -key2 "$p2_key" \
    -p1-addr "localhost:$p1_port" \
    -p2-addr "localhost:$p2_port" \
    -positions-file "$positions_file" \
    -games "$games" \
    -concurrency "$concurrency" \
    -both-sides \
    -iterations "$iterations" \
    -stats-file "$stats_file"
