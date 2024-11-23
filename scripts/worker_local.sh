#!/bin/bash

base_dir="$(realpath $(dirname $0)/..)"
log_dir="$base_dir/log"

test -d "$log_dir" || mkdir "$log_dir"

cd "$base_dir/cpp/build"
# Run the worker for proper self-play training LOCALLY.
# Run for 1 hour.
# Best settings (threads:fibers:batch) for resnet/10 on RTX 4080: 4:128:256
env \
HEXZ_TRAINING_SERVER_ADDR=localhost:50051 \
HEXZ_MAX_RUNTIME_SECONDS=3600 \
HEXZ_WORKER_SPEC=cuda@4:128:256 \
HEXZ_SUSPEND_WHILE_TRAINING=true \
./worker 2>&1 | tee -a "$log_dir/worker.log"
