#!/bin/bash

# Validate the two explicitly specified checkpoints against each other.

iterations=3200
num_games=50

server_addr1=localhost:50171
server_addr2=localhost:50172

model_base_dir="$HOME/tmp/hexz-models/models/flagz"
model_name=res10

p1_cp="$1"
p2_cp="$2"

p1_model="$model_base_dir/$model_name/checkpoints/$p1_cp/scriptmodule.pt"
p2_model="$model_base_dir/$model_name/checkpoints/$p2_cp/scriptmodule.pt"

p1_model_key="$model_name:$p1_cp"
p2_model_key="$model_name:$p2_cp"

cd $(dirname $0)/..

pushd cpp/build > /dev/null
echo "Starting cpuserver for P1 using model $p1_model and address $server_addr1"
./cpuserver --device=cuda --max_think_time_ms=0 --model_path="$p1_model" --model_key="$p1_model_key" --server_addr=$server_addr1  >> /tmp/cpuserver1.log 2>&1 &
cpu1_pid=$!

echo "Starting cpuserver for P2 using model $p2_model and address $server_addr2"
./cpuserver --device=cuda --max_think_time_ms=0 --model_path="$p2_model" --model_key="$p2_model_key" --server_addr=$server_addr2  >> /tmp/cpuserver2.log 2>&1 &
cpu2_pid=$!

sleep 2  # let cpuservers become available
popd > /dev/null

go run ./cmd/nbench -logfile './nbench.jsonl' -num-games $num_games -p1-addr $server_addr1 -p2-addr $server_addr2 -p1-max-iter $iterations -p2-max-iter $iterations -svg-file '' #./nbench_prev.html

echo "Terminating cpuserver processes"
kill $cpu1_pid
kill $cpu2_pid
