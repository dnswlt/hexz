#!/bin/bash

# Validate the latest checkpoint against a previous one.
# The latest model will play as P2.

iterations=3200

server_addr1=localhost:50071
server_addr2=localhost:50072

model_base_dir="$HOME/tmp/hexz-models/models/flagz"
model_name=res10
latest_cp=$(ls "$model_base_dir/$model_name/checkpoints" | awk -F'/' '{print $NF}' | sort -nr | head -n 1)
benchmark_cp=$((latest_cp - 5))

latest_model="$model_base_dir/$model_name/checkpoints/$latest_cp/scriptmodule.pt"
benchmark_model="$model_base_dir/$model_name/checkpoints/$benchmark_cp/scriptmodule.pt"

play_as_blue=0

p1_model="$latest_model"
p2_model="$benchmark_model"
if [[ $play_as_blue -eq 1 ]]; then
    p1_model="$benchmark_model"
    p2_model="$latest_model"
fi

cd $(dirname $0)/..

pushd cpp/build > /dev/null
echo "Starting cpuserver for P2 using model $p1_model and address $server_addr1"
./cpuserver --device=cuda --max_think_time_ms=0 --model_path="$p1_model" --server_addr=$server_addr1  >> /tmp/cpuserver1.log 2>&1 &
cpu1_pid=$!

echo "Starting cpuserver for P2 using model $p2_model and address $server_addr2"
./cpuserver --device=cuda --max_think_time_ms=0 --model_path="$p2_model" --server_addr=$server_addr2  >> /tmp/cpuserver2.log 2>&1 &
cpu2_pid=$!

sleep 2  # let cpuservers become available
popd > /dev/null

echo 'go run ./cmd/nbench -num-games 5 -p1-addr $server_addr1 -p2-addr $server_addr2 -p1-max-iter $iterations -p2-max-iter $iterations'
go run ./cmd/nbench -num-games 5 -p1-addr $server_addr1 -p2-addr $server_addr2 -p1-max-iter $iterations -p2-max-iter $iterations -svg-file '' #./nbench_prev.html

echo "Terminating cpuserver processes"
kill $cpu1_pid
kill $cpu2_pid
