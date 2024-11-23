if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    if [[ -e "$HOME/miniconda3/bin/activate" ]]; then
        . $HOME/miniconda3/bin/activate pyhexz
    else
        echo "You must be in the 'pyhexz' conda env"
        exit 1
    fi
fi

cd $(dirname $0)/../pyhexz/src

env \
HEXZ_MODEL_BLOCKS=10 \
HEXZ_MODEL_TYPE=resnet \
HEXZ_BATCH_SIZE=4096 \
HEXZ_TRAINING_TRIGGER_THRESHOLD=100000 \
HEXZ_TRAINING_EXAMPLES_WINDOW_SIZE=300000 \
HEXZ_MODEL_NAME=res10 \
HEXZ_MODEL_REPO_BASE_DIR=$HOME/tmp/hexz-models \
HEXZ_MIN_RUNS_PER_MOVE=800 \
HEXZ_NUM_EPOCHS=1 \
HEXZ_LEARNING_RATE=0.01 \
HEXZ_ADAM_WEIGHT_DECAY=1e-4 \
HEXZ_DEVICE=cuda \
HEXZ_SHUFFLE=true \
HEXZ_TRAINING_PARAMS_FILE="../../scripts/training_params_local.json" \
gunicorn --bind :8088 --workers 1 --threads 8 --timeout 0 'pyhexz.training_server:create_app()'
