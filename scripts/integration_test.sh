#!/bin/bash

# Runs all Go Hexz server tests, assuming local dependencies.

# Exit on first error.
set -e

cd $(dirname $0)/..

export PGPASSWORD=hexz_test
redis_addr="localhost:6379"
postgres_url="postgres://hexz_test@localhost:5432/hexz_test"

echo "Running unit tests..."
go test ./...

echo "Running Redis integration tests..."
go test ./internal/hexzmem -test-redis-addr $redis_addr

echo "Running PostgreSQL integration tests..."
go test ./internal/hexzsql -test-postgres-url $postgres_url

echo "Running server integration tests..."
go test ./pkg/hexz -test-redis-addr $redis_addr -test-postgres-url $postgres_url
