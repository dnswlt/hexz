#!/bin/bash

# Starts a local Hexz game server that connects to a local PostgreSQL and a local Redis server.
# This must only be used for development and testing.

export PGPASSWORD=hexz
go run ./cmd/server -debug -postgres-url postgres://hexz@localhost:5432/hexz -redis-addr localhost:6379 -from-address nobody@example.com
