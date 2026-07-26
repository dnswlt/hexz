#!/bin/bash
set -euo pipefail

# Validate two checkpoints against each other on a fixed, paired position set.
# Two CUDA cpuserver containers are started so the Go coordinator can run games
# concurrently without requiring a host C++ build.

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <checkpoint_1> <checkpoint_2>"
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

cleanup() {
    docker stop "$p1_container" "$p2_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

for spec in "p1:$1:$p1_port:$p1_container" "p2:$2:$p2_port:$p2_container"; do
    IFS=: read -r player checkpoint port container <<< "$spec"
    model_path="/models/models/flagz/$model_name/checkpoints/$checkpoint/scriptmodule.pt"
    echo "Starting $player server for $model_name:$checkpoint on port $port"
    docker run --rm -d \
        --name "$container" \
        -p "$port:$port" \
        -v "$model_repo:/models:ro" \
        --gpus all \
        "$image" \
        --device=cuda \
        --max_think_time_ms=0 \
        --model_path="$model_path" \
        --model_key="$model_name:$checkpoint" \
        --server_addr="0.0.0.0:$port" >/dev/null
done

sleep "${HEXZ_NBENCH_STARTUP_SECONDS:-3}"

go run ./cmd/nbench2 \
    -model-repo "$model_repo" \
    -key1 "$model_name:$1" \
    -key2 "$model_name:$2" \
    -p1-addr "localhost:$p1_port" \
    -p2-addr "localhost:$p2_port" \
    -positions-file "$positions_file" \
    -games "$games" \
    -concurrency "$concurrency" \
    -both-sides \
    -iterations "$iterations" \
    -stats-file "$stats_file"
