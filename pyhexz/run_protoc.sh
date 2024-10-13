# protoc -Ihexzpb hexzpb/hexz.proto --python_out=pyhexz/src/pyhexz --pyi_out=pyhexz/src/pyhexz
test -z "$CONDA_PYTHON_EXE" && { echo "Must run $0 in a conda environment"; exit 1; }

# grpc_tools.protoc uses the directory structure to determine the _pb2's package.
# Make sure generated code uses proper imports like
# from pyhexz import hexz_pb2 as ...
# in the generated _grpc.py file.
# https://stackoverflow.com/questions/62818183/protobuf-grpc-relative-import-path-discrepancy-in-python/76946302#76946302
cd $(dirname $0)/src
cp ../../hexzpb/hexz.proto pyhexz/
python3 -m grpc_tools.protoc --proto_path=. --python_out=. --pyi_out=. --grpc_python_out=. pyhexz/hexz.proto
rm pyhexz/hexz.proto
echo "Generated Python gRPC and protocol buffer files in $(pwd)/pyhexz."
