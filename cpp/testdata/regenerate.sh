#!/bin/bash
# Regenerate the test data used by the C++ tests.
#
# Should be run in a conda (or other) environment that has
# torch and other deps installed.
set -eux

dir="$(dirname $0)"
pushd "$dir/../../pyhexz/src"
python3 -m pyhexz.run --mode=export --model="$dir/scriptmodule.pt" --force

popd
pushd "$dir"
python3 -c '
import torch
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
torch.save(t, "tensor_2x2.pt")'

echo "Successfully generated new testdata."
