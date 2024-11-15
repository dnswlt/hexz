# Hexz in C++

To generate training examples as efficiently as possible,
we need to implement the MCTS search using PyTorch (`libtorch`) in C++.
The Python version is too slow for this task.

## Build the binary

**NOTE**: Make sure you are NOT inside a conda environment when building the C++ libs.

**TIP**: See the Docker section below if you just want to run the workers, not build them.

All commands assume you are in the `cpp` subdirectory.

### Dependencies

#### cmake

On Linux:

```bash
sudo apt install cmake
```

On macos:

```bash
brew install cmake
```

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

#### Protocol buffers

gRPC comes with the `protoc` and `grpc_cpp_plugin` binaries that are needed to generate
the C++ protocol buffer source files. After installing gRPC, run

```bash
scripts/run_protoc.sh cpp
```

to generate the C++ source files.

#### Boost fibers

```bash
wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2
tar xjf boost_1_86_0.tar.bz2
cd boost_1_86_0/
./bootstrap.sh --prefix=$HOME/opt/boost
./b2 install
```

### Finally: Build the worker

```bash
# Create new build directory to start from a clean slate.
rm -rf ./build
mkdir build
cd build
# Build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/.local/lib/cmake;$HOME/opt/libtorch/share/cmake;$HOME/opt/boost/lib/cmake" .
make -j4
```

## Docker

For clients that just want to run the workers (and not develop them), Docker is probably the easiest way:

### NVIDIA Container Toolkit

To enable NVIDIA on docker, install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```

### Build the docker image

**NOTE**: Run `docker build` from the base directory, not from `cpp`,
because docker needs access to the `.proto` files.

```bash
docker build . -f cpp/Dockerfile.worker-cuda --tag europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/worker-cuda:latest
```

(Note to a future me updating the Dockerfile: relevant nvidia docker images can be found here:
<https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/supported-tags.md>)

Push to GCP (only possible for the project owner :) ):

```bash
docker push europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/worker-cuda:latest
```

### Run the docker image

To run the Docker image locally (remove `--gpus all` when running the CPU-only container):

Adjust `HEXZ_MAX_RUNTIME_SECONDS`, `HEXZ_WORKER_THREADS`, `HEXZ_FIBERS_PER_THREAD` as needed
to optimize GPU utilization. `HEXZ_PREDICTION_BATCH_SIZE` should be at most #threads * #fibers.

Set `HEXZ_TRAINING_SERVER_ADDR` to the address of a running training server (see
[../pyhexz/README.md](../pyhexz/README.md)).

```bash
docker run \
  -e HEXZ_TRAINING_SERVER_ADDR=$HOSTNAME:50051 \
  -e HEXZ_DEVICE=cuda \
  -e HEXZ_MAX_RUNTIME_SECONDS=120 \
  -e HEXZ_WORKER_THREADS=8 \
  -e HEXZ_FIBERS_PER_THREAD=16 \
  -e HEXZ_PREDICTION_BATCH_SIZE=128 \
  -e HEXZ_RUNS_PER_MOVE=800 \
  -e HEXZ_UCT_C=1.5 \
  -e HEXZ_RUNS_PER_FAST_MOVE=100 \
  -e HEXZ_DIRICHLET_CONCENTRATION=0.35 \
  -e HEXZ_FAST_MOVE_PROB=0.5 \
  -e HEXZ_STARTUP_DELAY_SECONDS=0 \
  --gpus all \
  europe-west4-docker.pkg.dev/hexz-cloud-run/hexz/worker-cuda:latest
```

## VS Code

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

For a test run to see if everything works (this will not send any training examples
to the training server):

```bash
./worker \
--worker_threads=1 \
--training_server_addr=localhost:50051 \
--max_runtime_seconds=60 \
--device=cuda \
--fibers_per_thread=1 \
--prediction_batch_size=256 \
--suspend_while_training=true \
--runs_per_move=8 \
--dry_run
```

For a proper run, use the following config (using docker style env vars instead of flags):

```bash
env HEXZ_TRAINING_SERVER_ADDR=localhost:50051 \
  HEXZ_MAX_RUNTIME_SECONDS=120 \
  HEXZ_DEVICE=cuda \
  HEXZ_WORKER_THREADS=16 \
  HEXZ_FIBERS_PER_THREAD=32 \
  HEXZ_PREDICTION_BATCH_SIZE=512 \
  HEXZ_RUNS_PER_MOVE=800 \
  HEXZ_RUNS_PER_FAST_MOVE=100 \
  HEXZ_FAST_MOVE_PROB=0.5 \
  HEXZ_UCT_C=1.5 \
  HEXZ_DIRICHLET_CONCENTRATION=0.35 \
  HEXZ_RANDOM_PLAYOUTS=0 \
  HEXZ_STARTUP_DELAY_SECONDS=5 \
  HEXZ_SUSPEND_WHILE_TRAINING=true \
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
* OpenSpiel: DeepMind's open source alpha zero implementation:
  <https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md>
* muZero: _Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model_
  <https://arxiv.org/pdf/1911.08265>
