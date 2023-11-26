# Hexz in C++

Because ... why not have it in three different languages?!

## Install dependencies

On macos:

```bash
brew install cmake
brew install protobuf
```

## Build the binary

**NOTE:** Make sure you are NOT inside a conda environment when building the C++ libs.

### ARM-based Macs

All commands assume you are in the `cpp` subdirectory.

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/miniconda3/pkgs/pytorch-2.1.0-py3.11_0/lib/python3.11/site-packages/torch/share/cmake" ..
cmake --build . --parallel 4
```

As of Oct 2023 there are no official pre-built arm64 binaries for libtorch on pytorch.org,
so the small example from <https://pytorch.org/cppdocs/installing.html> does not work.

The easiest way to get libtorch for arm64 is by installing torch via Anaconda (or miniconda3),
and then adding the path to the torch `cmake` folder to `CMAKE_PREFIX_PATH` (see above).

### Linux

#### Protocol buffers

Build protobuf as a shared library[^1]: (here we'll install it to the default `/usr/loca/` prefix)

```bash
cd /tmp
wget https://github.com/protocolbuffers/protobuf/releases/download/v24.4/protobuf-24.4.tar.gz
tar xzf protobuf-24.4.tar.gz
cd protobuf-24.4
cd third_party

cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_PROTOC_BINARIES=OFF .
cmake --build . --parallel 4 
sudo cmake --install .
ldconfig
```

[^1] NOTE: I had linker error when trying to use the statically linked protobuf library. It
    might be an option to build statically with `-DDCMAKE_POSITION_INDEPENDENT_CODE=ON` (I tried
    that, but then forgot to actually install the library before testing it... :facepalm:).
    But since PyTorch is dynamically linked and the much larger lib anyway, we can go all-in
    on dylib, why not.

```
/usr/bin/ld: libhexzpb.a(hexz.pb.cc.o): warning: relocation against `_ZN6hexzpb46_AddTrainingExamplesResponse_default_instance_E' in read-only section `.text.startup'                
/usr/bin/ld: libhexzpb.a(hexz.pb.cc.o): relocation R_X86_64_PC32 against symbol `descriptor_table_hexz_2eproto' can not be used when making a shared object; recompile with -fPIC     
/usr/bin/ld: final link failed: bad value                                                                                                                                             
collect2: error: ld returned 1 exit status    
```

#### libtorch

```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
cd ~/opt
unzip /tmp/libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

#### CMake

To build the worker on Linux, now finally run:

```bash
cd $HEXZ_REPO_DIR/cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$HOME/opt/libtorch/share/cmake;/usr/local/lib/cmake/protobuf" ..
cmake --build . --parallel 4
```

### Docker

```bash
docker build . -f Dockerfile.ccworker --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/ccworker:latest
docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/ccworker:latest
```

Run the Docker image locally:

```bash
docker run \
  -e PYTHONUNBUFFERED=1 \
  -e HEXZ_TRAINING_SERVER_URL=http://$HOSTNAME:8080 \
  -e HEXZ_MAX_RUNTIME_SECONDS=60 \
  -e HEXZ_RUNS_PER_MOVE=800 \
  -e HEXZ_UCT_C=5.0 \
  -e HEXZ_RUNS_PER_FAST_MOVE=100 \
  -e HEXZ_DIRICHLET_CONCENTRATION=0.35 \
  -e HEXZ_FAST_MOVE_PROB=0.5 \
  -e HEXZ_STARTUP_DELAY_SECONDS=0 \
  europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/ccworker:latest
```

### VS Code

See `.vscode/c_cpp_properties.json`. Also add `CMAKE_PREFIX_PATH` under _Cmake: Configure Environment_.

## Run the binary

To run the worker, first make sure the training server is up and running. Then:

```bash
env HEXZ_TRAINING_SERVER_URL=http://localhost:8080 \
  HEXZ_MAX_RUNTIME_SECONDS=60 \
  HEXZ_RUNS_PER_MOVE=800 \
  HEXZ_UCT_C=2.5 \
  HEXZ_RUNS_PER_FAST_MOVE=100 \
  HEXZ_DIRICHLET_CONCENTRATION=0.35 \
  HEXZ_FAST_MOVE_PROB=0.5 \
  ./worker
```

### Heap profiling

Using gperftools:

```bash
env HEAPPROFILE=/tmp/worker.hprof LD_PRELOAD=$HOME/tmp/gperftools-2.13/.libs/libtcmalloc.so \
  HEXZ_TRAINING_SERVER_URL=http://localhost:8080 \
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
