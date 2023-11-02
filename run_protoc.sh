#!/bin/bash

cd $(dirname $0)
protoc hexzpb/hexz.proto --go_out=. --go_opt=paths=source_relative
protoc -Ihexzpb hexzpb/hexz.proto --python_out=pyhexz/src/pyhexz --pyi_out=pyhexz/src/pyhexz
protoc -Ihexzpb hexzpb/hexz.proto --cpp_out=cpp/
