#!/bin/bash

local_model_path="$(dirname "$(realpath "$0")")/../cpp/testdata/scriptmodule.pt"

./worker \
--worker_threads=1 \
--training_server_addr='dryrun' \
--local_model_path="$local_model_path" \
--max_games=3 \
--device=cpu \
--fibers_per_thread=1 \
--prediction_batch_size=1 \
--runs_per_move=10 \
--dry_run