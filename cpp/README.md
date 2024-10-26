# Hexz in C++

To generate training examples as efficiently as possible,
we need to implement the MCTS search using PyTorch in C++.
The Python version is too slow for this task.

## Install dependencies

On macos:

```bash
brew install cmake
```

## Build the binary

**NOTE:** Make sure you are NOT inside a conda environment when building the C++ libs.

All commands assume you are in the `cpp` subdirectory.

### Dependencies

#### libtorch

The following works for Linux, for macos check the PyTorch homepage
for the relevant archive (probably
<https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.1.zip>)

```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip
cd ~/opt
unzip /tmp/libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

MacOS security policy might prevent from executing that library. In that case, run sth like

```bash
sudo xattr -r -d com.apple.quarantine $HOME/opt/libtorch/lib/*.dylib
```

#### gRPC

As of Oct 2024 we migrated from http to gRPC for the communication between (C++) workers
and (Python) training server.

To build gRPC, follow <https://grpc.io/docs/languages/cpp/quickstart/>, which on Linux meant
first installing gRPC, protobuf and absl to `$HOME/.local`:

(the `tmp/grpc` directory consumed a whopping 3.5G after running `make`!)

```bash
cd $HOME/tmp
git clone --recurse-submodules -b v1.66.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc

export GRPC_INSTALL_DIR=$HOME/.local
cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$GRPC_INSTALL_DIR \
      ../..
make -j 4
make install
popd
```

#### Boost fibers

```bash
wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2
tar xjf boost_1_86_0.tar.bz2
cd boost_1_86_0/
./bootstrap.sh --prefix=$HOME/opt/boost
./b2 install
```

### Finally: Build it

Build the worker:

```bash
# Create new build directory to start from a clean slate.
rm -rf ./build
mkdir build
cd build
# Build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/.local/lib/cmake;$HOME/opt/libtorch/share/cmake;$HOME/opt/boost/lib/cmake" .
make -j4
```

### Docker

Build from the base directory, not from `cpp`, because docker needs access to the `.proto` files.

```bash
docker build . -f cpp/Dockerfile.ccworker --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/ccworker:latest
docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/ccworker:latest
```

Run the Docker image locally:

```bash
docker run \
  -e PYTHONUNBUFFERED=1 \
  -e HEXZ_TRAINING_SERVER_ADDR=http://$HOSTNAME:8080 \
  -e HEXZ_MAX_RUNTIME_SECONDS=60 \
  -e HEXZ_RUNS_PER_MOVE=800 \
  -e HEXZ_UCT_C=1.5 \
  -e HEXZ_RUNS_PER_FAST_MOVE=100 \
  -e HEXZ_DIRICHLET_CONCENTRATION=0.35 \
  -e HEXZ_FAST_MOVE_PROB=0.5 \
  -e HEXZ_STARTUP_DELAY_SECONDS=0 \
  europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/ccworker:latest
```

### VS Code

Install the CMake and C/C++ extensions.
Add the following to your `.vscode/settings.json` (which is not under version control):
Adjust the `CMAKE_PREFIX_PATH` directories depending on where you installed PyTorch and gRPC.

```json
    "cmake.sourceDirectory": "${workspaceFolder}/cpp",
    "cmake.buildDirectory": "${workspaceFolder}/cpp/build",
    "cmake.configureOnOpen": true,
    "cmake.configureSettings": {
        "CMAKE_PREFIX_PATH": [
            "${userHome}/opt/lib/cmake", 
            "${userHome}/opt/libtorch/share/cmake",
            "${userHome}/opt/boost/lib/cmake"
        ]
    },
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
```

## Run the binary

To run the worker, first make sure the training server is up and running. Then:

```bash
env HEXZ_TRAINING_SERVER_ADDR=localhost:50051 \
  HEXZ_MAX_RUNTIME_SECONDS=60 \
  HEXZ_RUNS_PER_MOVE=800 \
  HEXZ_UCT_C=1.5 \
  HEXZ_RUNS_PER_FAST_MOVE=100 \
  HEXZ_DIRICHLET_CONCENTRATION=0.35 \
  HEXZ_FAST_MOVE_PROB=0.5 \
  HEXZ_RESIGN_THRESHOLD=0.999 \
  HEXZ_WORKER_THREADS=1 \
  ./worker
```

### Heap profiling

Using gperftools:

```bash
env HEAPPROFILE=/tmp/worker.hprof LD_PRELOAD=$HOME/tmp/gperftools-2.13/.libs/libtcmalloc.so \
  HEXZ_TRAINING_SERVER_ADDR=http://localhost:8080 \
  HEXZ_MAX_RUNTIME_SECONDS=60 \
  HEXZ_RUNS_PER_MOVE=800 \
  HEXZ_UCT_C=1.5 \
  HEXZ_RUNS_PER_FAST_MOVE=100 \
  HEXZ_DIRICHLET_CONCENTRATION=0.35 \
  HEXZ_FAST_MOVE_PROB=0.5 \
  HEXZ_STARTUP_DELAY_SECONDS=0 \
  ./worker
```

## References

* The AlphaGo paper _Mastering the game of Go with deep neural networks and tree search_:
  <https://research.google/pubs/pub44806/>,
  <https://indico.hep.caltech.edu/event/56/contributions/774/attachments/385/475/nature16961.pdf>
* The AlphaGo Zero paper _Mastering the Game of Go without Human Knowledge_:
  <https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf>
* The AlphaZero paper _Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm_:
  <https://arxiv.org/pdf/1712.01815.pdf>
* Facebook's reimplementation of AZ: _ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero_:
  <https://research.facebook.com/publications/elf-opengo-an-analysis-and-open-reimplementation-of-alphazero/>
* KataGo: D. Wu, _Accelerating Self-Play Learning in Go_:
  <https://arxiv.org/pdf/1902.10565.pdf>
