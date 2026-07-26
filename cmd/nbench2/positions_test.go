package main

import (
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/proto"
)

func TestStartingPositionRoundTrip(t *testing.T) {
	positions, err := generateStartingPositions(4)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "positions.jsonl")
	if err := writeStartingPositions(path, positions); err != nil {
		t.Fatal(err)
	}
	got, err := readStartingPositions(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(positions) {
		t.Fatalf("got %d positions, want %d", len(got), len(positions))
	}
	for i := range positions {
		if got[i].ID != positions[i].ID {
			t.Errorf("position %d ID = %q, want %q", i, got[i].ID, positions[i].ID)
		}
		if !proto.Equal(got[i].State, positions[i].State) {
			t.Errorf("position %d changed after round trip", i)
		}
	}
}

func TestWriteStartingPositionsDoesNotOverwrite(t *testing.T) {
	positions, err := generateStartingPositions(1)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "positions.jsonl")
	if err := writeStartingPositions(path, positions); err != nil {
		t.Fatal(err)
	}
	if err := writeStartingPositions(path, positions); err == nil {
		t.Fatal("second write unexpectedly succeeded")
	}
}

func TestWilsonInterval(t *testing.T) {
	lo, hi := wilsonInterval(90, 100)
	if lo < 0.825 || lo > 0.827 {
		t.Errorf("lower bound = %.6f, want approximately 0.826", lo)
	}
	if hi < 0.943 || hi > 0.945 {
		t.Errorf("upper bound = %.6f, want approximately 0.944", hi)
	}
}
