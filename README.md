# Hexz

Hexz is a collection of web-based games that are played on a hexagon
board.

## Docker

This is totally a WIP. To build and run `hexz` as a Docker container:

```
docker build -t hexz .
docker run -p 8080:8080 hexz
```

## Cloud Run

Build and deploy:

```
docker build . --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/hexz:latest
docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/hexz:latest
```

Run the Artifact Registry image locally:

```
PORT=8080 && docker run -p 8080:${PORT} -e PORT=${PORT} europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/hexz:latest
```


## Protocol Buffers
The generated sources of all `.proto` files are checked in to this repository,
so users shouldn't need to regenerate them. 

To do so anyway, run the following command in the root directory of this
repository:

```
protoc hexzpb/hexz.proto --go_out=. --go_opt=paths=source_relative
```


## WASM

The vision is to run CPU players in the user's browser, not on the server.

Totally WIP: to build the WASM module, run:

```
GOOS=js GOARCH=wasm go build -o ./resources/wasm/hexz.wasm cmd/wasm/main.go && gzip -f ./resources/wasm/hexz.wasm
```
