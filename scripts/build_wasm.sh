#!/bin/bash

wasm_file="./resources/wasm/hexz.wasm"
GOOS=js GOARCH=wasm go build -o "$wasm_file" cmd/wasm/main.go && gzip -f $wasm_file
size=$(test "$(uname)" = "Darwin" && stat -f%z "$wasm_file.gz" || stat -c%s "$wasm_file.gz")
echo "Built WASM module $wasm_file.gz ($size bytes)."
