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
docker build . -f Dockerfile.server --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest && \
  docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
docker build . -f Dockerfile.worker --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/worker:latest && \
  docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/worker:latest
```

## Training and self-play in the Cloud

The architecture is as follows:

* A single `server` running on a GCE VM is responsible for training and distributing model updates.
  * It holds the latest checkpoint of the configured model in memory (specified by the `MODEL_NAME` and
    loaded from GCS).
  * On loading the model, it also stores the model in Redis so that workers can read it.
  * It accepts `http POST` requests containing a `NumpyExample` protobuf message at `/training/examples`.
    These are sent from workers, see below.
  * Once enough (4096, configurable via `MINIBATCH_SIZE`) examples have been posted, it trains the
    current model with the collected examples, resulting in an updated model.
  * The new model is stored on GCS as the next checkpoint. The model stored in Redis is also updated.
    To inform workers about the new model, it publishes the latest model checkpoint number via Redis
    Pubsub to all workers.

* Multiple `worker` jobs running on Cloud Run generate training examples via self-play and send them
  to the server.
  * Workers expect the server to be present.
  * They read the latest model (identified by `MODEL_NAME`) from Redis (where the server put it)
    and use it for self-play until a new model checkpoint gets published via Redis Pubsub.
  * During self-play, workers POST the generated examples to the server, one example at a time.
    Workers do not store any data themselves.
  * Once workers receive a notification that a new model checkpoint is available, they load the
    the new checkpoint from Redis and continue their self-play using that model.
  * Workers run for a configurable amount of time, before they quit.

Models and examples are stored by the `server` in a GCS bucket using the following folder structure:

* `/models/flagz/{model_name}/checkpoints/{checkpoint_num}/`
  * `model.pt`
  * `examples/{example_batch}.zip`

Models are stored in the standard PyTorch format using `torch.save`. Examples are stored as zip files
containing binary protobuf messages of type `github.com.dnswlt.hexz.NumpyExample`.
