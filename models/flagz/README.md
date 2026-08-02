# Runtime models

This directory contains versioned neural models intended for running the game
server. Test fixtures remain under `cpp/testdata` and must not be replaced by
runtime models.

The private CUDA game-server launcher, `scripts/flagz_public_cuda.sh`, uses the
checkpoint-60 rich model by default. Set `HEXZ_MODEL_PATH` and
`HEXZ_MODEL_KEY` together to test another repository-backed artifact.
