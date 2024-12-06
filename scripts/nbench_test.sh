
cd $(dirname $0)/..

cpp/build/cpuserver \
    --device cuda \
   --max_think_time_ms 0 \
   --model_path ~/tmp/hexz-models/models/flagz/res10/checkpoints/58/scriptmodule.pt \
   --model_key res10:50 \
   --server_addr localhost:50091 &
cpu_pid=$!
echo "CPU server started (PID $cpu_pid)"

# Let CPU server come alive
sleep 3
go run ./cmd/nbench -num-games 10 -p1-addr localhost:50091 -p2-addr localhost:50091 -p1-max-iter 3200 -p2-max-iter 3200

echo "Terminating CPU server (PID $cpu_pid)"
kill $cpu_pid


# Separate RPCs (server-side batching):
# bash scripts/nbench_local.sh  0.66s user 0.77s system 1% cpu 1:27.53 total

# Batched RPCs (client-side batching):
# bash scripts/nbench_local.sh  0.66s user 1.17s system 2% cpu 1:26.13 total
