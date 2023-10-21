# Hexz in Python

This directory contains the Python implementation of the Flagz game.

## Running it

Unless stated otherwise, all commands should be run from the current `pyhexz` directory.

### Cython

First, compile the Cython modules:

```bash
python3 setup.py build_ext --build-lib=src
```

## pytest

Just run `pytest` without any arguments. The `pytest.ini` file tells pytest where to find stuff.

### Flask web server

```bash
cd src
HEXZ_MODEL_PATH=$HOME/git/github.com/dnswlt/hexz-models/models/flagz/cain python3 -m flask --app pyhexz.server run --port 9094
```

### Self-play

```bash
cd src
python3 -m pyhexz.hexz --mode=selfplay --model=../../hexz-models/models/flagz/cain --num-workers=6 --device=mps --max-games=10000000 --max-seconds=60 --runs-per-move=800 --output-dir=/tmp
```

