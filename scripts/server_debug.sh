#!/bin/bash

# Starts a local Hexz game server that connects to a local PostgreSQL and a local Redis server.
# This must only be used for development and testing.

export PGPASSWORD=hexz

postgres_url="postgres://hexz@localhost:5432/hexz"
redis_addr="localhost:6379"
from_address="nobody@example.com"

go run ./cmd/server -debug -postgres-url "$postgres_url" -redis-addr "$redis_addr" -from-address "$from_address"
