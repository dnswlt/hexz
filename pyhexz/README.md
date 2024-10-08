# Hexz in Python

This directory contains the Python implementation of the Flagz server.

## Running it

Unless stated otherwise, all commands should be run from the current `pyhexz` directory.

### Anaconda

Follow <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html> to install `miniconda`.

Then create the `pyhexz` conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate pyhexz
```

(You might need to run `~/miniconda/bin/activate pyhexz` instead,
depending on your conda setup.)

### Cython

Now compile the Cython modules:

```bash
python3 setup.py build_ext --build-lib=src
```

**NOTE**: Building the `pyhexz.ccapi` extension assumes that
the C++ libraries in ../cpp were already built. If you don't care about the
Python `cpu_server` module (e.g. if you only want to run the training server),
you can skip this step.

Test the C++ bindings for the MoveSuggester by running:

```bash
cd src
# macos version. For Linux, use LD_LIBRARY_PATH
DYLD_LIBRARY_PATH=$HOME/git/github.com/dnswlt/hexz/cpp/build python3 -c '
from pyhexz.ccapi import CppMoveSuggester
m = CppMoveSuggester("../../cpp/testdata/scriptmodule.pt")
print(m.suggest_move(b"__HELLO__"))'
```

It should print a friendly hello message.

### pytest

Just run `pytest` without any arguments. The `pytest.ini` file tells pytest where to find stuff.

### Flask web server

The Flask app runs behind a `gunicorn` WSGI web server, both locally and in a Docker container.
To run the `training_server` locally:

```bash
cd src
env HEXZ_BATCH_SIZE=1024 \
  HEXZ_MODEL_NAME=edgar \
  HEXZ_MODEL_REPO_BASE_DIR=/tmp/hexz-models \
  HEXZ_NUM_EPOCHS=1 \
  HEXZ_MAX_CHECKPOINT_DIFF=2 \
  gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 'pyhexz.training_server:create_app()'
```

To only run the CPU player engine (use $DYLD_LIBRARY_PATH on macos):

```bash
LD_LIBRARY_PATH=$HOME/git/github.com/dnswlt/hexz/cpp/build \
  HEXZ_LOCAL_MODEL_PATH=$HOME/git/github.com/dnswlt/hexz-models/models/flagz/seth/checkpoints/60/scriptmodule.pt \
  gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 'pyhexz.cpu_server:create_app()'
```

### Docker and Cloud Run

```bash
# server
docker build . -f Dockerfile.server --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
```

Running these images locally:

```bash
# server
PORT=8080 && docker run -p 8080:${PORT} -e PORT=${PORT} \
  -e PYTHONUNBUFFERED=1 \
  -e HEXZ_MODEL_REPO_BASE_DIR=/tmp/hexz/models \
  -e HEXZ_BATCH_SIZE=512 \
  -e HEXZ_MODEL_NAME=test \
  -e HEXZ_NUM_EPOCHS=1 \
  -e HEXZ_MAX_CHECKPOINT_DIFF=2 \
  europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
```

## Training and self-play in the Cloud

### OUTDATED SECTION

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
