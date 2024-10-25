# Hexz

Hexz is a collection of web-based games that are played on a hexagon board.

## Overview

This repository contains several related but independent components.
If you just want to play the games, the following components are all you need.

* A game server (written in Go) that can be used to play the full collection of hexz games (e.g. Flagz or Classic):
  [cmd/server/main.go](./cmd/server/main.go).

* A Web Assembly (WASM) module that can be used to play Flagz in single player mode (1P), where the CPU player
  will run in the user's browser: [cmd/wasm/main.go](./cmd/wasm/main.go). The CPU player uses Monte-Carlo tree search
  (MCTS) to evaluate its moves.

* A CPU player server that can alternatively be used for Flagz 1P, where the CPU player is running as a
  standalone server: [cmd/cpu/main.go](./cmd/cpu/main.go). The server-based CPU player uses the same algorithm
  as the WASM one, but may be stronger because it has access to more (CPU, memory) resources.
  It can also be useful when evaluating other (e.g. ML-based) game engines against this reference implementation.

As part of an ongoing experiment to obtain stronger CPU players, there are several components that allow you
to train an AlphaZero-style CPU player:

* A PyTorch model that is used in an AlphaZero-style MCTS guided by a neural network with a policy and a value
  head: [pyhexz/src/pyhexz/hexz.py](./pyhexz/src/pyhexz/hexz.py) (yes, it should be renamed to `model.py`).

* A C++ implementation of workers that generate training examples via self-play using the PyTorch model
  in a neural MCTS algorithm: [cpp/worker_main.cc](./cpp/worker_main.cc).
  The workers send the examples to a training server, from which they also obtain the latest model updates.

  * There is also a Python implementation of the workers, but it is currently not maintained.

* A Python/Flask training server that accepts training examples from the workers and continuously
  trains the model in minibatches using the provided training examples:
  [pyhexz/src/pyhexz/server.py](./pyhexz/src/pyhexz/server.py)

* An evaluation tool `nbench` ([cmd/nbench/main.go](./cmd/nbench/main.go)) that lets different CPU
  players play against each other and evaluates which one is stronger.

To train the ML model and use it as a CPU player, you need to build and run the C++ workers and the Python
training server. See the respective [cpp/README.md](./cpp/README.md)
and [pyhexz/README.md](./pyhexz/README.md) files for instructions.

There are also several visualisation tools (HTML+SVG export) to understand better what the CPU players
and ML models are doing.

## Building and running

The following instructions only concern the Go game server and WASM module. See [cpp/README.md](./cpp/README.md)
and [pyhexz/README.md](./pyhexz/README.md) for instructions on building and running the ML toolkit.

### Run the game server locally

The simplest way to start a local server to play the games is by running the stateful server with `go run`
(see <https://go.dev/doc/install> if you don't have Go installed yet):

```bash
go run cmd/server/main.go
```

The stateful server was the initial implementation for local use, but the stateless variant
is better suited for use as a public server. You can also run it locally, but it requires
a Redis server to maintain the game state, and optionally a PostgreSQL server for game and move
history. See below for details. Start the server in stateless mode thus:

```bash
go run cmd/server/main.go -stateless
```

Run `go run cmd/server/main.go -help` for an overview of command-line options.

The server prints the URL you should open to start playing (usually <http://localhost:8080/hexz>).

### Redis and PostgreSQL

To run the game server in stateless mode, you must have a Redis server running.
A PostgreSQL database is optional, but enables undo/redo functionality and game
history.

Redis does not require any configuration. Just install and run it. E.g., on macos:
<https://redis.io/docs/install/install-redis/install-redis-on-mac-os/>.

The PostgreSQL setup is of course a bit more involved. See [sql/schema.sql](./sql/schema.sql) for
the schema. TODO: create Dockerfile to simplify this.

### Docker and Cloud Run

Build and deploy Docker image:

```bash
docker build . --tag europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/hexz:latest
docker push europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/hexz:latest
```

Run the Artifact Registry image locally:

```bash
PORT=8080 && docker run -p 8080:${PORT} -e PORT=${PORT} europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/hexz:latest
```

Deploy to Cloud Run:

```bash
gcloud run deploy hexz --image=europe-west6-docker.pkg.dev/hexz-cloud-run/hexz/hexz:latest --region=europe-west6 --project=hexz-cloud-run  && \
  gcloud run services update-traffic hexz --to-latest
```

### WASM

CPU players typically run in the user's browser, not on the server.

To build the WASM module, run:

```bash
GOOS=js GOARCH=wasm go build -o ./resources/wasm/hexz.wasm cmd/wasm/main.go && gzip -f ./resources/wasm/hexz.wasm
```

### Protocol Buffers

The generated sources of all `.proto` files are **no longer** checked in to
this repository, so users need to regenerate them.

Run the following command in the root directory of this repository:

```bash
bash run_protoc.sh
```

If there are errors generating the protobuf sources for Go, you might need to install `protoc-gen-go`:

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
env PATH=$PATH:$HOME/go/bin bash run_protoc.sh
```

### Cloud Logging

List logs in ascending order, starting from a given timestamp:

```bash
t=$(TZ=UTC date -d'2 hours ago' +%Y-%m-%dT%H:%M:%SZ) && \
  gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=hexz AND textPayload:\"CPU stats\" AND timestamp>=\"$t\"" --project hexz-cloud-run --order=asc --limit=10
```

List recent logs, in descending order:

```bash
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=hexz AND textPayload:"CPU stats"' --freshness=2h --project hexz-cloud-run --limit=10
```
