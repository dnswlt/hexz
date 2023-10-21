# Hexz

Hexz is a collection of web-based games that are played on a hexagon
board.

## Docker and Cloud Run

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

## Cloud Logging

List logs in ascending order, starting from a given timestamp:

```bash
t=$(TZ=UTC date -d'2 hours ago' +%Y-%m-%dT%H:%M:%SZ) && \
  gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=hexz AND textPayload:\"CPU stats\" AND timestamp>=\"$t\"" --project hexz-cloud-run --order=asc --limit=10
```

List recent logs, in descending order:

```bash
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=hexz AND textPayload:"CPU stats"' --freshness=2h --project hexz-cloud-run --limit=10
```

## Protocol Buffers

The generated sources of all `.proto` files are checked in to this repository,
so users shouldn't need to regenerate them. 

To do so anyway, run the following command in the root directory of this
repository:

```bash
protoc hexzpb/hexz.proto --go_out=. --go_opt=paths=source_relative
protoc -Ihexzpb hexzpb/hexz.proto --python_out=python 
```

## WASM

CPU players run in the user's browser, not on the server.

To build the WASM module, run:

```
GOOS=js GOARCH=wasm go build -o ./resources/wasm/hexz.wasm cmd/wasm/main.go && gzip -f ./resources/wasm/hexz.wasm
```
