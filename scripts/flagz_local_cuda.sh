#!/bin/bash
set -euo pipefail

# Starts a CUDA cpuserver Docker container and a local stateful Go Flagz server.
# This script is intended for quickly playing games locally.

cd "$(dirname "$0")/.."

url_path_prefix=/hexzml
server_port=8080
cpuserver_port=50071
cpuserver_addr="localhost:$cpuserver_port"

model_repo="${HEXZ_MODEL_REPO_BASE_DIR:-$HOME/data/hexz-models}"
model_name="${HEXZ_MODEL_NAME:-res10-r4-cp62}"
checkpoint="${HEXZ_CHECKPOINT:-50}"
image="${HEXZ_CPUSERVER_IMAGE:-hexz-cpuserver-cuda:latest}"
container_name="hexz-cpuserver-local"

model_path="/models/models/flagz/$model_name/checkpoints/$checkpoint/scriptmodule.pt"
model_key="$model_name:$checkpoint"

cleanup() {
    echo "Stopping cpuserver Docker container..."
    docker stop "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cleanup

echo "Starting cpuserver container with model $model_key on port $cpuserver_port..."
docker run --rm -d \
    --name "$container_name" \
    -p "$cpuserver_port:$cpuserver_port" \
    -v "$model_repo:/models:ro" \
    --gpus all \
    "$image" \
    --device=cuda \
    --max_think_time_ms=5000 \
    --model_path="$model_path" \
    --model_key="$model_key" \
    --server_addr="0.0.0.0:$cpuserver_port"

echo "Building Go server..."
go build ./cmd/server

echo "Starting Go server on port $server_port..."
./server \
  -cpu-player-mode remote \
  -url-path-prefix "$url_path_prefix" \
  -remote-cpu-url "$cpuserver_addr" \
  -cpu-think-time 5s \
  -port=$server_port \
  -debug

