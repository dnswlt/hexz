#!/bin/bash

cd $(dirname $0)
protoc hexzpb/hexz.proto --go_out=. --go_opt=paths=source_relative
protoc -Ihexzpb hexzpb/hexz.proto --cpp_out=cpp/ --grpc_out=cpp/ --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin)
# Python protoc is run separately, as it must be run from within the conda environment.