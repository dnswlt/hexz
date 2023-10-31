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
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/miniconda3/pkgs/pytorch-2.1.0-py3.11_0/lib/python3.11/site-packages/torch/share/cmake" .. && cmake --build .
```

### ARM-based Macs

As of Oct 2023 there are no official pre-built arm64 binaries for libtorch on pytorch.org,
so the small example from https://pytorch.org/cppdocs/installing.html does not work.

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
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -Dprotobuf_BUILD_TESTS=OFF .
cmake --build . --parallel 4 
sudo cmake --install .
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
cmake -DCMAKE_BUILD_TYPE=Release -DCPR_ENABLE_SSL=OFF -DCMAKE_PREFIX_PATH="$HOME/opt/libtorch/share/cmake;/usr/local/lib/cmake/protobuf" ..
cmake --build .
# Run the binary, AT LAST!
LD_LIBRARY_PATH=/usr/local/lib HEXZ_LOCAL_MODEL_PATH=/tmp/scriptmodule.pt ./worker
```


### VS Code

See `.vscode/c_cpp_properties.json`. Also add `CMAKE_PREFIX_PATH` under _Cmake: Configure Environment_.

