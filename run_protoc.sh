#!/bin/bash

cd $(dirname $0)
echo "Generating proto and gRPC files for Go..."
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative hexzpb/hexz.proto
echo "Generating proto and gRPC files for C++..."
protoc -Ihexzpb --cpp_out=cpp/ --grpc_out=cpp/ --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) hexzpb/hexz.proto
# Python protoc is run separately, as it must be run from within the conda environment.
