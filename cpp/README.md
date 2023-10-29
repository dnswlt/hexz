# Hexz in C++

Because ... why not have it in three different languages?!

## Install dependencies

On macos:

```bash
brew install cmake
brew install protobuf
```

## Build the binary

All commands assume you are in the `cpp` subdirectory.

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$HOME/miniconda3/pkgs/pytorch-2.1.0-py3.11_0/lib/python3.11/site-packages/torch/share/cmake" .. && cmake --build .
```

### ARM-based Macs

As of Oct 2023 there are no official pre-built arm64 binaries for libtorch on pytorch.org,
so the small example from https://pytorch.org/cppdocs/installing.html does not work.

The easiest way to get libtorch for arm64 is by installing torch via Anaconda (or miniconda3),
and then adding the path to the torch `cmake` folder to `CMAKE_PREFIX_PATH` (see above).


### VS Code

See `.vscode/c_cpp_properties.json`. Also add `CMAKE_PREFIX_PATH` under _Cmake: Configure Environment_.

