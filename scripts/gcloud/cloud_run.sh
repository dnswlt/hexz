#!/bin/bash
set -euo pipefail

training_server_addr="${HEXZ_TRAINING_SERVER_ADDR:?Set HEXZ_TRAINING_SERVER_ADDR to the training server host and port}"

gcloud run deploy worker-cuda-svc \
--image=europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/worker-cuda:latest \
--cpu=4 \
--memory=16Gi \
--max-instances=1 \
--set-env-vars=HEXZ_TRAINING_SERVER_ADDR="$training_server_addr",HEXZ_MAX_RUNTIME_SECONDS=3600,HEXZ_DEVICE=cuda,HEXZ_WORKER_THREADS=4,HEXZ_FIBERS_PER_THREAD=256,HEXZ_PREDICTION_BATCH_SIZE=256,HEXZ_RUNS_PER_MOVE=800,HEXZ_RUNS_PER_FAST_MOVE=100,HEXZ_FAST_MOVE_PROB=0.5,HEXZ_UCT_C=1.5,HEXZ_DIRICHLET_CONCENTRATION=0.35,HEXZ_ENABLE_HEALTH_SERVICE=true \
--use-http2 \
--no-cpu-throttling \
--no-cpu-boost \
--region=europe-west4 \
--project=hexz-cloud-run \
 && gcloud run services update-traffic worker-cuda-svc --to-latest
