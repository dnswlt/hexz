#!/bin/bash

# Run the worker with gperftools CPU profiler enabled.
# The worker runs in dry_run mode, but retrieves training
# parameters from the training server, i.e. it runs with
# "real" settings.

base_dir="$(realpath $(dirname $0)/..)"
log_dir="$base_dir/log"

test -d "$log_dir" || mkdir "$log_dir"

cd "$base_dir/cpp/build"

env \
CPUPROFILE=./cpuprofile.prof \
HEXZ_TRAINING_SERVER_ADDR=localhost:50051 \
HEXZ_MAX_RUNTIME_SECONDS=10 \
HEXZ_WORKER_SPEC=cuda@4:128:256 \
HEXZ_SUSPEND_WHILE_TRAINING=true \
./worker --dry_run
