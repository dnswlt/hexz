#!/bin/bash
set -euo pipefail

# Backward-compatible convenience entry point. The production deployment is
# the self-contained Compose application documented in DEPLOYMENT.md.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

docker compose up -d "$@"
docker compose ps
