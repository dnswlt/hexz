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

### Protocol buffers

Before running the training server, you need to generate the protocol buffer sources
for Python. Run the following command in the project root folder:

```bash
scripts/run_protoc.sh py
```

### Flask web server

The Flask app runs behind a `gunicorn` WSGI web server, both locally and in a Docker container.
To run the `training_server` locally:

```bash
env HEXZ_MODEL_BLOCKS=10 \
  HEXZ_MODEL_TYPE=resnet \
  HEXZ_BATCH_SIZE=4096 \
  HEXZ_TRAINING_TRIGGER_THRESHOLD=100000 \
  HEXZ_MODEL_NAME=resus \
  HEXZ_MODEL_REPO_BASE_DIR=$HOME/tmp/hexz-models \
  HEXZ_NUM_EPOCHS=7 \
  HEXZ_DEVICE=cuda \
  HEXZ_SHUFFLE=true \
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

## Training

It is surprisingly hard to find detailed information on how AlphaZero's training actually worked
in the [paper](https://arxiv.org/pdf/1712.01815.pdf):

We know that

* 700'000 "training steps" were run, using mini-batches of size 4096.
* 800 MCTS simulations were executed for each move.
* The learning rate was decreased over time: 0.2, 0.02, 0.002, 0.0002.
* 44 million training games were played for chess.
* During training, moves were selected proportional to their MCTS visit count.

But how did training actually work?

* What does a training step in the "700'000 training steps" mentioned in
  the AlphaZero paper actually involve?
* After how many new samples was a training step triggered?
* How many epochs were used per training step?

DeepMind's Open Spiel has an
[implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/alpha_zero.py#L425)
of AlphaZero that provides the following answers:

* Each training step (call to `learn`) uses a single epoch; in other words,
  the typical DL "epoch loop" is not used for training at all.
* Each training step selects examples from a `replay_buffer` that contains the latest samples from
  self-play games. Its size (`replay_buffer_size`) can be configured and defaults to 2^16, i.e. 65536.
* Once the replay buffer has `replay_buffer_size/replay_buffer_reuse` (default: `65536/3 = 21845`)
  new individual samples (not whole games! each game has N samples, typically one per move),
  a new training step is performed.
* N batches (of default size 1024) of samples are sampled
  *with replacement* from the whole buffer, where N is `replay_buffer_size/train_batch_size`;
  so on average, each sample in the buffer is used once, but due to the replacement some may
  be chosen more often and others not at all.
* The model gets checkpointed after each training step. Training is always done on the
  current model, i.e. training does not start from a fresh model each time.

I don't think that the default values are chosen specifically to be optimal for learning
chess. Open Spiel supports a variety of simpler games as well.
However, the structure of training is very similar to ours! We should probably drop the epoch loop,
and maybe instead train more frequently, i.e. reduce the trigger threshold.
As of Nov 2024, we train for 7 epochs once we have 100'000 new samples, sampling (without replacement)
batches of size 4096 from a window of the latest 1'000'000 samples.

In DeepMind's terms, we set `replay_buffer_reuse = 10` and `replay_buffer_size = 1_000_000`.
Due to the 7 epochs, our reuse factor per sample is actually 70. That's probably too much.

If we assume similar parameters in the actual AlphaZero implementation:

* 700'000 training steps have been run for 44'000'000 games.
* Assume 40 moves per game, so 1'760'000'000 samples.
* 2514 new samples per training step?? That's roughly the size of a mini-batch!
  So their average number of moves per game is probably 65.16: 700'000 * 4096 / 65.16 ~= 44'000'000.

Still it does not give us a clue about the training window used for each training step.
Each 4096 batch must trigger a training step: otherwise you couldn't train 700'000 steps
on the 44m games. But those mini-batches must not contain sequential samples from a single game.

