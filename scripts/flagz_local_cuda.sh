#!/bin/bash

# Starts a local C++ CUDA cpuserver and a local stateful Go Flagz server.
# This script is intended for quickly playing games locally.

tls_flags=""

# Process flags
for arg in "$@"; do
    case "$arg" in
        tls|-tls|--tls)
        tls_flags='-tls-cert=./fullchain.pem -tls-key=./privkey.pem'
        ;;
    esac
done

cd $(dirname $0)/..

server_port=8071
cpuserver_addr="localhost:50071"
model_base_dir="$HOME/tmp/hexz-models/models/flagz"
model_name=res10
latest=$(ls "$model_base_dir/$model_name/checkpoints" | awk -F'/' '{print $NF}' | sort -nr | head -n 1)
scriptmodule="$model_base_dir/$model_name/checkpoints/$latest/scriptmodule.pt"


pushd cpp/build > /dev/null
echo "Starting cpuserver with model $model_name:$latest"
./cpuserver --device=cuda --max_think_time_ms=5000 --model_path="$scriptmodule" --server_addr="$cpuserver_addr" &
cpu_pid=$!
echo "Started cpuserver with PID $cpu_pid"
popd > /dev/null

go run ./cmd/server -remote-cpu-url "$cpuserver_addr" -cpu-think-time 5s -stateless=false -port=$server_port $tls_flags

echo "Terminating cpuserver process"
kill $cpu_pid
