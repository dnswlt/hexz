#!/bin/bash

# This bash script should be used as the ENTRYPOINT of the Dockerfile
# running pyhexz commands or servers.

# using conda activate in shell scripts seems to be a complete mess.
# found the following approach that works at
# https://github.com/conda/conda/issues/7980#issuecomment-492784093
eval "$(conda shell.bash hook)"
conda activate pyhexz

# exec the final command:
cd src
if [ "$HEXZ_ENTRYPOINT_MODE" = "gunicorn" ]; then
    # Run as a web server
    exec gunicorn --bind :$PORT "$@"
fi
# Run as a batch command.
exec python3 -m pyhexz.worker "$@"
