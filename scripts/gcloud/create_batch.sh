gcloud beta batch jobs submit hexz-worker-batch-$(date +"%Y%m%d%H%M") --location europe-west4 --config - <<EOD
{
  "name": "projects/hexz-cloud-run/locations/europe-west4/jobs/hexz-worker-batch",
  "taskGroups": [
    {
      "taskCount": "1",
      "parallelism": "1",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "2000",
          "memoryMib": "8192"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/worker-cuda:latest",
              "entrypoint": "",
              "volumes": []
            },
            "environment": {
              "variables": {
                "HEXZ_TRAINING_SERVER_ADDR": "hexz.hopto.org:50051",
                "HEXZ_MAX_RUNTIME_SECONDS": "120",
                "HEXZ_DEVICE": "cuda",
                "HEXZ_WORKER_THREADS": "4",
                "HEXZ_FIBERS_PER_THREAD": "1",
                "HEXZ_PREDICTION_BATCH_SIZE": "1",
                "HEXZ_RUNS_PER_MOVE": "800",
                "HEXZ_RUNS_PER_FAST_MOVE": "100",
                "HEXZ_FAST_MOVE_PROB": "0.5",
                "HEXZ_UCT_C": "1.5",
                "HEXZ_DIRICHLET_CONCENTRATION": "0.35",
                "HEXZ_RANDOM_PLAYOUTS": "0",
                "HEXZ_STARTUP_DELAY_SECONDS": "0",
                "HEXZ_SUSPEND_WHILE_TRAINING": "false",
              },
            }
          }
        ],
        "volumes": []
      }
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "installGpuDrivers": true,
        "policy": {
          "provisioningModel": "STANDARD",
          "machineType": "n1-standard-4",
          "accelerators": [
              {
              "type": "nvidia-tesla-t4",
              "count": 1
              }
          ]
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOD
