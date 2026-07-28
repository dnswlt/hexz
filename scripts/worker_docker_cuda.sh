#!/bin/bash
# Runs the CUDA worker Docker container connecting to the local training server.

base_dir="$(realpath $(dirname $0)/..)"
log_dir="$base_dir/log"
mkdir -p "$log_dir"

server_addr="${1:-${HEXZ_TRAINING_SERVER_ADDR:-host.docker.internal:50051}}"
runtime_seconds="${2:-${HEXZ_MAX_RUNTIME_SECONDS:-120}}"
worker_spec="${HEXZ_WORKER_SPEC:-cuda@4:128:256}"
suspend_while_training="${HEXZ_SUSPEND_WHILE_TRAINING:-true}"

echo "Starting CUDA worker container connecting to $server_addr for $runtime_seconds seconds..."
docker run \
  -e HEXZ_TRAINING_SERVER_ADDR="$server_addr" \
  -e HEXZ_MAX_RUNTIME_SECONDS="$runtime_seconds" \
  -e HEXZ_WORKER_SPEC="$worker_spec" \
  -e HEXZ_SUSPEND_WHILE_TRAINING="$suspend_while_training" \
  -e HEXZ_STARTUP_DELAY_SECONDS=0 \
  --add-host=host.docker.internal:host-gateway \
  --gpus all \
  hexz-worker-cuda:latest 2>&1 | tee -a "$log_dir/worker.log"
