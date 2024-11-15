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
    protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative hexzpb/hexz.proto
fi

if [[ $gen_cpp == 1 ]]; then
    echo "Generating proto and gRPC files for C++..."
    protoc -Ihexzpb --cpp_out=cpp/ --grpc_out=cpp/ --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) hexzpb/hexz.proto
    protoc -Ihexzpb --cpp_out=cpp/ --grpc_out=cpp/ --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) hexzpb/health.proto
fi

if [[ $gen_py == 1 ]]; then
    echo "Generating proto and gRPC files for Python..."
    # protoc -Ihexzpb hexzpb/hexz.proto --python_out=pyhexz/src/pyhexz --pyi_out=pyhexz/src/pyhexz
    test -z "$CONDA_PYTHON_EXE" && { echo "CONDA_PYTHON_EXE is not set. You must run '$0 py' in a conda environment"; exit 1; }

    # grpc_tools.protoc uses the directory structure to determine the _pb2's package.
    # Make sure generated code uses proper imports like
    # from pyhexz import hexz_pb2 as ...
    # in the generated _grpc.py file.
    # https://stackoverflow.com/questions/62818183/protobuf-grpc-relative-import-path-discrepancy-in-python/76946302#76946302
    cd pyhexz/src
    cp ../../hexzpb/hexz.proto pyhexz/
    python3 -m grpc_tools.protoc --proto_path=. --python_out=. --pyi_out=. --grpc_python_out=. pyhexz/hexz.proto
    rm pyhexz/hexz.proto
fi
