#!/bin/bash
set -euo pipefail

# Starts the private Flagz ML endpoint on this workstation:
#
#   LAN/nginx -> Go game server (:8080) -> CUDA cpuserver (localhost:50071)
#
# The default model is a versioned, repository-backed runtime asset. Redis and
# PostgreSQL are optional because the NUC hosting the reverse proxy may not also
# host the old Hexz data services.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

url_path_prefix="${HEXZ_URL_PATH_PREFIX:-/hexzml}"
server_host="${HEXZ_SERVER_HOST:-0.0.0.0}"
server_port="${HEXZ_SERVER_PORT:-8080}"
cpuserver_port="${HEXZ_CPUSERVER_PORT:-50071}"
cpuserver_addr="localhost:$cpuserver_port"

model_path="${HEXZ_MODEL_PATH:-$repo_root/models/flagz/res10-rich-v1-r4-cp60/scriptmodule.pt}"
model_key="${HEXZ_MODEL_KEY:-res10-rich-v1-r4:60}"
image="${HEXZ_CPUSERVER_IMAGE:-hexz-cpuserver-cuda:latest}"
container_name="${HEXZ_CPUSERVER_CONTAINER:-hexz-cpuserver-public}"

redis_addr="${HEXZ_REDIS_ADDR:-}"
postgres_url="${HEXZ_POSTGRES_URL:-}"

if [[ ! -f "$model_path" ]]; then
    echo "Model does not exist: $model_path" >&2
    exit 1
fi
model_path="$(realpath "$model_path")"

cleanup() {
    echo "Stopping CUDA cpuserver..."
    docker stop "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cleanup

echo "Starting CUDA cpuserver with $model_key..."
docker run --rm -d \
    --name "$container_name" \
    -p "127.0.0.1:$cpuserver_port:$cpuserver_port" \
    -v "$model_path:/model/scriptmodule.pt:ro" \
    --gpus all \
    "$image" \
    --device=cuda \
    --max_think_time_ms=5000 \
    --uct_c=1.5 \
    --initial_root_q_value=0 \
    --initial_q_penalty=0 \
    --model_path=/model/scriptmodule.pt \
    --model_key="$model_key" \
    --server_addr="0.0.0.0:$cpuserver_port" >/dev/null

sleep "${HEXZ_CPUSERVER_STARTUP_SECONDS:-3}"
if ! docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -qx true; then
    echo "CUDA cpuserver failed to start:" >&2
    docker logs "$container_name" >&2 || true
    exit 1
fi

echo "Building Go server..."
go build ./cmd/server

server_args=(
    -cpu-player-mode remote
    -url-path-prefix "$url_path_prefix"
    -remote-cpu-url "$cpuserver_addr"
    -cpu-think-time 5s
    -host "$server_host"
    -port "$server_port"
    -debug
)
if [[ -n "$redis_addr" ]]; then
    server_args+=(-redis-addr "$redis_addr")
fi
if [[ -n "$postgres_url" ]]; then
    server_args+=(-postgres-url "$postgres_url")
fi

echo "Starting private ML game server at http://$server_host:$server_port$url_path_prefix"
if [[ -z "$redis_addr" ]]; then
    echo "Using in-memory game state (set HEXZ_REDIS_ADDR to use Redis)."
fi
./server "${server_args[@]}" 2>&1 | tee log/hexz.log
