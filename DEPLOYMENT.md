# Production deployment

The complete Hexz ML application runs on the CUDA workstation as one Docker
Compose project. The NUC is only a TLS reverse proxy. PostgreSQL, Redis and the
C++ gRPC service are private Compose services; only the Go server is published
on the LAN.

## First start

```bash
cp .env.example .env
# Replace HEXZ_POSTGRES_PASSWORD in .env with a long random value.
docker compose up -d
docker compose ps
```

Compose uses the repository-backed rich checkpoint 60. On a clean machine it
builds the Go and C++ images when they are absent. On MONSTA the existing CUDA
image is reused. PostgreSQL initializes `sql/schema.sql` only when its named
volume is empty.

Keep `.env` with the host deployment state and do not casually change its
database password after PostgreSQL has initialized. It is ignored by Git.

The application is then available at:

```text
http://MONSTA:8080/hexzml
```

## Everyday operation

```bash
# Start or reconcile everything after pulling new code.
docker compose up -d

# Inspect health and logs.
docker compose ps
docker compose logs -f game cpuserver

# Stop the application without deleting containers or data.
docker compose stop

# Remove containers and the private network, retaining named volumes.
docker compose down
```

Do not add `--volumes` to `docker compose down` unless the PostgreSQL and Redis
data are intentionally being destroyed. With `restart: unless-stopped`, the
application returns after a host or Docker restart unless it was manually
stopped.

Rebuild only the Go service after source changes:

```bash
docker compose build game
docker compose up -d game
```

Rebuilding `cpuserver` is much slower and is needed only after C++ changes:

```bash
docker compose build cpuserver
docker compose up -d cpuserver game
```

## Backups

The model is in Git. PostgreSQL holds durable game history and user data. Redis
holds live game/session state and persists it to its named volume using AOF.

Create a PostgreSQL dump:

```bash
mkdir -p backups
docker compose exec -T postgres pg_dump -U hexz -d hexz -Fc \
  > backups/hexz-$(date +%Y%m%d-%H%M%S).dump
```

Restore a dump into an initialized, otherwise disposable database:

```bash
docker compose stop game
docker compose exec -T postgres dropdb -U hexz --if-exists hexz
docker compose exec -T postgres createdb -U hexz hexz
docker compose exec -T postgres pg_restore -U hexz -d hexz --clean --if-exists \
  < backups/hexz.dump
docker compose start game
```

## NUC reverse proxy

The NUC needs only one proxy location in the chosen TLS virtual host. This is
the complete Hexz-specific part; Redis, PostgreSQL and CUDA are never exposed
there. Hexz uses Server-Sent Events, so buffering must remain disabled and the
read timeout must be long.

```nginx
location ^~ /hexzml {
    proxy_pass http://192.168.1.71:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 1h;
    proxy_send_timeout 1h;
}
```

Keep this in a separate Hexz virtual host rather than modifying the weather
site. Test with `nginx -t` before reloading nginx.
