#!/bin/bash

cd $(dirname $0)/../cpp/build
# Run the worker for proper self-play training LOCALLY.
# Run for 1 hour.
# Best settings (T/F/B) for resnet/10 on RTX 4080: 4/256/256
env \
HEXZ_TRAINING_SERVER_ADDR=localhost:50051 \
HEXZ_MAX_RUNTIME_SECONDS=3600 \
HEXZ_DEVICE=cuda \
HEXZ_WORKER_THREADS=4 \
HEXZ_FIBERS_PER_THREAD=256 \
HEXZ_PREDICTION_BATCH_SIZE=256 \
HEXZ_RUNS_PER_MOVE=800 \
HEXZ_RUNS_PER_FAST_MOVE=100 \
HEXZ_FAST_MOVE_PROB=0.5 \
HEXZ_UCT_C=1.5 \
HEXZ_DIRICHLET_CONCENTRATION=0.35 \
HEXZ_RANDOM_PLAYOUTS=0 \
HEXZ_STARTUP_DELAY_SECONDS=0 \
HEXZ_SUSPEND_WHILE_TRAINING=true \
./worker
