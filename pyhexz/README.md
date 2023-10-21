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

## pytest

Just run `pytest` without any arguments. The `pytest.ini` file tells pytest where to find stuff.

### Flask web server

WIP: to start the Flask web server and run a remote CPU player that can be called from the
Go server:

```bash
cd src
HEXZ_MODEL_PATH=$HOME/git/github.com/dnswlt/hexz-models/models/flagz/cain python3 -m flask --app pyhexz.server run --port 9094
```

### Self-play

```bash
cd src
python3 -m pyhexz.hexz --mode=selfplay --model=../../hexz-models/models/flagz/cain --num-workers=6 --device=mps --max-games=10000000 --max-seconds=60 --runs-per-move=800 --output-dir=/tmp
```

### Docker and Cloud Run

WIP

```bash
docker build .
```
