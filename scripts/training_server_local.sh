test "$CONDA_DEFAULT_ENV" = "pyhexz" || { echo "Must be in the 'pyhexz' conda env"; exit 1; }

env \
HEXZ_MODEL_BLOCKS=10 \
HEXZ_MODEL_TYPE=resnet \
HEXZ_BATCH_SIZE=4096 \
HEXZ_TRAINING_TRIGGER_THRESHOLD=100000 \
HEXZ_MODEL_NAME=resus \
HEXZ_MODEL_REPO_BASE_DIR=$HOME/tmp/hexz-models \
HEXZ_NUM_EPOCHS=7 \
HEXZ_DEVICE=cuda \
HEXZ_SHUFFLE=true \
gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 'pyhexz.training_server:create_app()'
