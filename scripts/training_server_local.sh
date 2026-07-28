base_dir="$(realpath "$(dirname "$0")/..")"

if [[ ! -x "$base_dir/pyhexz/.venv/bin/gunicorn" && -z "$CONDA_DEFAULT_ENV" ]]; then
    if [[ -e "$HOME/miniconda3/bin/activate" ]]; then
        . $HOME/miniconda3/bin/activate pyhexz
    else
        echo "No pyhexz virtualenv or active Conda environment found"
        exit 1
    fi
fi

log_dir="$base_dir/log"

mkdir -p "$log_dir"

GUNICORN_BIN="gunicorn"
if [[ -x "$base_dir/pyhexz/.venv/bin/gunicorn" ]]; then
    GUNICORN_BIN="$base_dir/pyhexz/.venv/bin/gunicorn"
fi

cd "$base_dir/pyhexz/src"

device="${HEXZ_DEVICE:-cuda}"
if [[ -z "${HEXZ_DEVICE:-}" && "$(uname)" = "Darwin" ]]; then
    device=mps
fi

repo_dir="${HEXZ_MODEL_REPO_BASE_DIR:-/home/dw/data/hexz-models}"
model_name="${HEXZ_MODEL_NAME:-res10}"

env \
HEXZ_MODEL_BLOCKS="${HEXZ_MODEL_BLOCKS:-10}" \
HEXZ_MODEL_TYPE="${HEXZ_MODEL_TYPE:-resnet}" \
HEXZ_BATCH_SIZE="${HEXZ_BATCH_SIZE:-4096}" \
HEXZ_TRAINING_TRIGGER_THRESHOLD="${HEXZ_TRAINING_TRIGGER_THRESHOLD:-25000}" \
HEXZ_TRAINING_EXAMPLES_WINDOW_SIZE="${HEXZ_TRAINING_EXAMPLES_WINDOW_SIZE:-1048576}" \
HEXZ_MODEL_NAME="$model_name" \
HEXZ_MODEL_REPO_BASE_DIR="$repo_dir" \
HEXZ_NUM_EPOCHS="${HEXZ_NUM_EPOCHS:-1}" \
HEXZ_TRAINING_BATCHES_PER_TRIGGER="${HEXZ_TRAINING_BATCHES_PER_TRIGGER:-25}" \
HEXZ_REPLAY_SAMPLING_CHUNK_SIZE="${HEXZ_REPLAY_SAMPLING_CHUNK_SIZE:-256}" \
HEXZ_TRAINING_SEED="${HEXZ_TRAINING_SEED:-1}" \
HEXZ_OPTIMIZER="${HEXZ_OPTIMIZER:-adam}" \
HEXZ_LEARNING_RATE="${HEXZ_LEARNING_RATE:-0.0003}" \
HEXZ_ADAM_WEIGHT_DECAY="${HEXZ_ADAM_WEIGHT_DECAY:-1e-4}" \
HEXZ_DEVICE="$device" \
HEXZ_SHUFFLE="${HEXZ_SHUFFLE:-true}" \
HEXZ_PIN_MEMORY="${HEXZ_PIN_MEMORY:-false}" \
PYTHONPATH="$base_dir/pyhexz/src" \
"$GUNICORN_BIN" --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 'pyhexz.training_server:create_app()' 2>&1 | tee -a "$log_dir/training.log"
