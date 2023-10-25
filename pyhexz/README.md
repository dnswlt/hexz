# Hexz in Python

This directory contains the Python implementation of the Flagz game.

## Running it

Unless stated otherwise, all commands should be run from the current `pyhexz` directory.

### Anaconda

Follow https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html to install `miniconda`.

Then create the `pyhexz` conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate pyhexz
```

### Cython

Now compile the Cython modules:

```bash
python3 setup.py build_ext --build-lib=src
```

### pytest

Just run `pytest` without any arguments. The `pytest.ini` file tells pytest where to find stuff.

### Flask web server

WIP: to start the Flask web server and run a remote CPU player that can be called from the
Go server:

```bash
cd src
HEXZ_MODEL_PATH=$HOME/git/github.com/dnswlt/hexz-models/models/flagz/cain python3 -m flask --app pyhexz.server run --port 9094
```

The Flask app runs behind a `gunicorn` WSGI web server in the Docker container. To reproduce that
locally, run the app as

```bash
gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 'pyhexz.server:create_app()'
```

### Self-play

```bash
cd src
python3 -m pyhexz.hexz --mode=selfplay --model=../../hexz-models/models/flagz/cain --num-workers=6 --device=mps --max-games=10000000 --max-seconds=60 --runs-per-move=800 --output-dir=/tmp
```

### Docker and Cloud Run

WIP

```bash
# server
docker build . -f Dockerfile.server --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
# worker
docker build . -f Dockerfile.worker --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/worker:latest
docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/worker:latest
```

Running these images locally:

```bash
# server
PORT=8080 && docker run -p 8080:${PORT} -e PORT=${PORT} \
  -e HEXZ_MODEL_REPO_BASE_DIR=/tmp/hexz/models \
  -e HEXZ_BATCH_SIZE=512 \
  -e HEXZ_MODEL_NAME=test \
  europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
```

```bash
# worker
docker run \
  -e HEXZ_TRAINING_SERVER_URL=http://nuc:8080 \
  europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/worker:latest
```

## Training and self-play in the Cloud

The architecture is as follows:

* A single `server` running in a Docker container on a GCE VM is responsible for training and 
  distributing model updates.
  * It holds the latest checkpoint of the configured model in memory (specified by the `MODEL_NAME` and
    loaded from GCS).
  * It accepts `http POST` requests for `AddTrainingExamplesRequest` protobuf messages at `/examples`.
    These are sent from workers, see below.
  * Once enough (configurable via `HEXZ_BATCH_SIZE`) examples have been posted, it trains the
    current model with the collected examples, resulting in an updated model.
  * The new model is stored in memory and on local disk as the next checkpoint.
    Workers are informed about the new model in the `AddTrainingExamplesResponse`. They can also poll
    the training server (`/models/current`) to obtain the current model version.

* Multiple `worker` jobs running as batch jobs on Cloud Run generate training examples via self-play
  and send them to the server.
  * Workers expect the server to be present.
  * They query the training server for the latest model version (`/models/current`), download this
    model (`/models/{model_name}/checkpoints/{checkpoint}`), and use it for self-play. On delivering
    examples to the training server (`/examples`) they learn about model updates and download a new
    model as necessary.
  * Workers run for a configurable amount of time, before they quit.

Models and examples are stored by the `server` on local disk using the following folder structure:

* `$HEXZ_MODEL_REPO_BASE_DIR/models/flagz/{model_name}/checkpoints/{checkpoint_num}/`
  * `model.pt`
  * `examples/{example_batch}.zip`  (not implemented yet)

Models are stored in the standard PyTorch format using `torch.save`. Examples are stored as zip files
containing binary protobuf messages of type `github.com.dnswlt.hexz.NumpyExample`.
