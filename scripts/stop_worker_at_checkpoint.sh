#!/bin/bash
#
# Stops one self-play worker after the training server has fully saved a target
# checkpoint. This lets a bounded training round run unattended without
# stopping the worker while a checkpoint is still being written.
#
# Usage:
#   scripts/stop_worker_at_checkpoint.sh CHECKPOINT CONTAINER
#
# Optional environment variables:
#   HEXZ_TRAINING_LOG  Training-server log to monitor (default: log/training.log)
#   HEXZ_POLL_SECONDS  Seconds between checks (default: 15)

set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || -z "$2" ]]; then
  echo "Usage: $0 CHECKPOINT CONTAINER" >&2
  exit 2
fi

target_checkpoint="$1"
worker_container="$2"
training_log="${HEXZ_TRAINING_LOG:-log/training.log}"
poll_seconds="${HEXZ_POLL_SECONDS:-15}"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "HEXZ_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$training_log" ]]; then
  echo "Training log does not exist: $training_log" >&2
  exit 1
fi
if ! docker inspect "$worker_container" >/dev/null 2>&1; then
  echo "Worker container does not exist: $worker_container" >&2
  exit 1
fi

completion_pattern="Training done:.*checkpoint=${target_checkpoint}\\)"
echo "Waiting for completed checkpoint ${target_checkpoint}; worker: ${worker_container}"
while ! rg -q "$completion_pattern" "$training_log"; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$worker_container")" != "true" ]]; then
    echo "Worker stopped before checkpoint ${target_checkpoint} completed" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done

echo "Checkpoint ${target_checkpoint} completed; stopping ${worker_container}"
docker stop "$worker_container"
