package main

import "testing"

func TestRedactPassword(t *testing.T) {
	got := redactPGPassword("postgres://user:password@host:1234/dbname?sslmode=disable")
	if got != "postgres://user:<redacted>@host:1234/dbname?sslmode=disable" {
		t.Errorf("Wrong redacted URL: %s", got)
	}
	got = redactPGPassword("user=jack password=secret host=pg.example.com port=5432 dbname=mydb")
	if got != "user=jack password=<redacted> host=pg.example.com port=5432 dbname=mydb" {
		t.Errorf("Wrong redacted URL: %s", got)
	}
	got = redactPGPassword("unknown/config/something")
	if got != "unknown/config/something" {
		t.Errorf("Wrong redacted URL: %s", got)
	}
}
