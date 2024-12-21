#!/bin/bash

# Starts a local C++ CUDA cpuserver and a local stateful Go Flagz server.
# This script is intended for quickly playing games locally.

cd $(dirname $0)/..

# Run under different path from non-ML server.
url_path_prefix=/hexzml
server_port=8080
cpuserver_addr="localhost:50071"
model_base_dir="$HOME/tmp/hexz-models/models/flagz"
model_name=res10
latest=$(ls "$model_base_dir/$model_name/checkpoints" | awk -F'/' '{print $NF}' | sort -nr | head -n 1)
scriptmodule="$model_base_dir/$model_name/checkpoints/$latest/scriptmodule.pt"


pushd cpp/build > /dev/null
echo "Starting cpuserver with model $model_name:$latest"
./cpuserver --device=cuda --max_think_time_ms=5000 --model_path="$scriptmodule" --model_key="$model_name:$latest" --server_addr="$cpuserver_addr" &
cpu_pid=$!
echo "Started cpuserver with PID $cpu_pid"
popd > /dev/null

trap 'echo "Killing cpuserver process $cpu_pid"; kill $cpu_pid; exit' INT

go build ./cmd/server || { echo "Build failed. Exiting."; exit 1; }

env PGPASSWORD=$(cat psql.cred.txt) ./server \
  -cpu-player-mode remote \
  -url-path-prefix "$url_path_prefix" \
  -remote-cpu-url "$cpuserver_addr" \
  -cpu-think-time 5s \
  -port=$server_port \
  -postgres-url "postgres://hexz@nuc:5432/hexz" \
  -debug \
  -redis-addr nuc:6379 \
  2>&1 | tee log/hexz.log

echo "Terminating cpuserver process"
kill $cpu_pid

