#!/bin/bash

if [[ $# == 0 ]]; then
    echo "Usage $0 <go|cpp|py>..."
    exit 1
fi

# protoc has to run from the project root directory.
cd $(dirname $0)/..

gen_go=0
gen_cpp=0
gen_py=0

for arg in "$@"; do
    case "$arg" in
        go)
            gen_go=1
            ;;
        cpp)
            gen_cpp=1
            ;;
        py)
            gen_py=1
            ;;
        *)
            echo "Unrecognized argument: $arg"
            exit 1
            ;;
    esac
done

if [[ $gen_go == 1 ]]; then
    echo "Generating proto and gRPC files for Go..."
    protoc --go_out=. --go_opt=module=github.com/dnswlt/hexz --go-grpc_out=. --go-grpc_opt=module=github.com/dnswlt/hexz \
      proto/hexz.proto proto/nbench.proto 
fi

if [[ $gen_cpp == 1 ]]; then
    echo "Generating proto and gRPC files for C++..."
    protoc -Iproto --cpp_out=cpp/ --grpc_out=cpp/ --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/hexz.proto
    protoc -Iproto --cpp_out=cpp/ --grpc_out=cpp/ --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/health.proto
fi

if [[ $gen_py == 1 ]]; then
    echo "Generating proto and gRPC files for Python..."
    PYTHON_BIN="python3"
    if [[ -x "$(pwd)/pyhexz/.venv/bin/python3" ]]; then
        PYTHON_BIN="$(pwd)/pyhexz/.venv/bin/python3"
    elif [[ -n "$CONDA_PREFIX" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python3"
    fi

    # grpc_tools.protoc uses the directory structure to determine the _pb2's package.
    # Make sure generated code uses proper imports like
    # from pyhexz import hexz_pb2 as ...
    # in the generated _grpc.py file.
    # https://stackoverflow.com/questions/62818183/protobuf-grpc-relative-import-path-discrepancy-in-python/76946302#76946302
    cd pyhexz/src
    cp ../../proto/hexz.proto ../../proto/nbench.proto pyhexz/
    # Replace proto/ prefix by pyhexz/ in imports ... what a mess.
    sed -i -e 's|^import "proto/\([^"]*\).proto"|import "pyhexz/\1.proto"|g' pyhexz/*.proto
    "$PYTHON_BIN" -m grpc_tools.protoc --proto_path=. --python_out=. --pyi_out=. --grpc_python_out=. pyhexz/hexz.proto pyhexz/nbench.proto
    rm -f pyhexz/hexz.proto pyhexz/nbench.proto
fi
