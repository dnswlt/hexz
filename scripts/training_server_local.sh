if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    if [[ -e "$HOME/miniconda3/bin/activate" ]]; then
        . $HOME/miniconda3/bin/activate pyhexz
    else
        echo "You must be in the 'pyhexz' conda env"
        exit 1
    fi
fi

base_dir="$(realpath $(dirname $0)/..)"
log_dir="$base_dir/log"

mkdir -p "$log_dir"

GUNICORN_BIN="gunicorn"
if [[ -x "$base_dir/pyhexz/.venv/bin/gunicorn" ]]; then
    GUNICORN_BIN="$base_dir/pyhexz/.venv/bin/gunicorn"
fi

cd "$base_dir/pyhexz/src"

device=cuda
if [[ "$(uname)" = "Darwin" ]]; then
    device=mps
fi

repo_dir="${HEXZ_MODEL_REPO_BASE_DIR:-/home/dw/data/hexz-models}"
model_name="${HEXZ_MODEL_NAME:-res10}"

env \
HEXZ_MODEL_BLOCKS=10 \
HEXZ_MODEL_TYPE=resnet \
HEXZ_BATCH_SIZE=4096 \
HEXZ_TRAINING_TRIGGER_THRESHOLD=100000 \
HEXZ_TRAINING_EXAMPLES_WINDOW_SIZE=1048576 \
HEXZ_MODEL_NAME="$model_name" \
HEXZ_MODEL_REPO_BASE_DIR="$repo_dir" \
HEXZ_NUM_EPOCHS=7 \
HEXZ_LEARNING_RATE=0.001 \
HEXZ_ADAM_WEIGHT_DECAY=1e-4 \
HEXZ_DEVICE="$device" \
HEXZ_SHUFFLE=true \
PYTHONPATH="$base_dir/pyhexz/src" \
"$GUNICORN_BIN" --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 'pyhexz.training_server:create_app()' 2>&1 | tee -a "$log_dir/training.log"
