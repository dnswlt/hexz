#!/bin/bash

# Validate the model-based CPU player against the Go MCTS-based reference player.

p1_iterations=${1:-32000}
p2_iterations=${2:-3200}
num_games=${3:-5}

server_port=50071
model_base_dir="${HEXZ_MODEL_REPO_BASE_DIR:-$HOME/data/hexz-models}"
model_name="${HEXZ_MODEL_NAME:-res10}"

latest_cp=$(ls "$model_base_dir/models/flagz/$model_name/checkpoints" 2>/dev/null | sort -nr | head -n 1)
latest_cp="${latest_cp:-63}"
scriptmodule="/models/models/flagz/$model_name/checkpoints/$latest_cp/scriptmodule.pt"

base_dir="$(realpath $(dirname $0)/..)"
log_dir="$base_dir/log"
mkdir -p "$log_dir"

# Clean up any leftover container
docker stop hexz-cpuserver > /dev/null 2>&1 || true

echo "Starting CUDA cpuserver container for $model_name checkpoint $latest_cp on port $server_port..."
docker run --rm -d \
  --name hexz-cpuserver \
  -p ${server_port}:${server_port} \
  -v "$model_base_dir:/models" \
  --gpus all \
  hexz-cpuserver-cuda:latest \
  --device=cuda \
  --max_think_time_ms=0 \
  --model_path="$scriptmodule" \
  --model_key="$model_name:$latest_cp" \
  --server_addr="0.0.0.0:${server_port}" > /dev/null

sleep 3  # let cpuserver start

echo "Running nbench: P1 (Go MCTS reference, $p1_iterations iters) vs P2 (ML Model $model_name:$latest_cp, $p2_iterations iters)..."
cd "$base_dir"
/usr/local/go/bin/go run ./cmd/nbench \
  -num-games $num_games \
  -p1-addr "" \
  -p2-addr "localhost:${server_port}" \
  -p1-max-iter $p1_iterations \
  -p2-max-iter $p2_iterations \
  -p2-eval 2>&1 | tee -a "$log_dir/nbench_mcts.log"

echo "Stopping cpuserver container..."
docker stop hexz-cpuserver > /dev/null 2>&1
