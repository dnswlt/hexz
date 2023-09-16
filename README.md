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
