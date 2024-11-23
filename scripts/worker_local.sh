#!/bin/bash

cd $(dirname $0)/../cpp/build
# Run the worker for proper self-play training LOCALLY.
# Run for 1 hour.
# Best settings (T/F/B) for resnet/10 on RTX 4080: 4/256/256
env \
HEXZ_TRAINING_SERVER_ADDR=localhost:50051 \
HEXZ_MAX_RUNTIME_SECONDS=3600 \
HEXZ_WORKER_SPEC=cuda@4:128:256 \
HEXZ_SUSPEND_WHILE_TRAINING=true \
./worker
