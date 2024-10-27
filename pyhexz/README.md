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
  gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 'pyhexz.training_server:create_app()'
```

```bash
# macos version. For Linux, use LD_LIBRARY_PATH
DYLD_LIBRARY_PATH=$HOME/git/github.com/dnswlt/hexz/cpp/build \
  HEXZ_LOCAL_MODEL_PATH=$HOME/git/github.com/dnswlt/hexz-models/models/flagz/seth/checkpoints/60/scriptmodule.pt \
  gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 'pyhexz.cpu_server:create_app()'
```

### Docker and Cloud Run

```bash
# server
docker build . -f Dockerfile.server --tag europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
docker push europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
```

Running the image locally:

```bash
# server
PORT=8080 && docker run -p 8080:${PORT} -e PORT=${PORT} \
  -e PYTHONUNBUFFERED=1 \
  -e HEXZ_MODEL_REPO_BASE_DIR=/tmp/hexz/models \
  -e HEXZ_BATCH_SIZE=512 \
  -e HEXZ_MODEL_NAME=test \
  -e HEXZ_NUM_EPOCHS=1 \
  europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/server:latest
```

## Python game implementation

If you wonder where the Python game engine code is:
it was removed in <https://github.com/dnswlt/hexz/commit/cdc83f6e098d3ab7dfc99439b824b8857ac70abe>.
