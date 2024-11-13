#!/bin/bash

# Validate the model-based CPU player against the Go MCTS-based one.

p1_iterations=${1:-1000}
p2_iterations=${2:-1600}

server_addr=localhost:50071
model_base_dir="$HOME/tmp/hexz-models/models/flagz"
model_name=resus
latest=$(ls "$model_base_dir/$model_name/checkpoints" | awk -F'/' '{print $NF}' | sort -nr | head -n 1)
scriptmodule="$model_base_dir/$model_name/checkpoints/$latest/scriptmodule.pt"

cd $(dirname $0)/..

pushd cpp/build > /dev/null
echo ./cpuserver --device=cuda --max_think_time_ms=0 --model_path="$scriptmodule" --server_addr=$server_addr
./cpuserver --device=cuda --max_think_time_ms=0 --model_path="$scriptmodule" --server_addr=$server_addr &
cpu_pid=$!
sleep 2  # let cpuserver become available
popd > /dev/null

echo go run ./cmd/nbench -p2-addr $server_addr -p1-max-iter $p1_iterations -p2-max-iter $p2_iterations -svg-file ./nbench.html
go run ./cmd/nbench -num-games 3 -p2-addr $server_addr -p1-max-iter $p1_iterations -p2-max-iter $p2_iterations -svg-file ./nbench.html

echo "Terminating cpuserver process"
kill $cpu_pid
