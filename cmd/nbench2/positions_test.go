package main

import (
	"path/filepath"
	"testing"

	npb "github.com/dnswlt/hexz/pkg/nbenchpb"
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

func result(position string, winner int32) *npb.BenchmarkResult_GameResult {
	return &npb.BenchmarkResult_GameResult{
		PositionId: position,
		Winner:     winner,
	}
}

func stats(results ...*npb.BenchmarkResult_GameResult) *ResultStats {
	s := &ResultStats{}
	for _, result := range results {
		s.Add(result)
	}
	return s
}

func TestPairedSummaryUsesDrawsAndPositionPairs(t *testing.T) {
	// Model 1 is P1 in the first games and P2 in the reverse games.
	// Its per-position scores are 1.0, 0.25, and 0.25.
	first := stats(
		result("a", 1),
		result("b", 0),
		result("c", 2),
	)
	reverse := stats(
		result("c", 0),
		result("a", 2),
		result("b", 1),
	)

	got, err := pairedSummary(first, reverse)
	if err != nil {
		t.Fatal(err)
	}
	if got.Positions != 3 || got.Games != 6 {
		t.Fatalf("positions/games = %d/%d, want 3/6", got.Positions, got.Games)
	}
	if got.Model1Wins != 2 || got.Model2Wins != 2 || got.Draws != 2 {
		t.Errorf("W/D/L = %d/%d/%d, want 2/2/2",
			got.Model1Wins, got.Draws, got.Model2Wins)
	}
	if got.Model1Score != 0.5 {
		t.Errorf("score = %v, want 0.5", got.Model1Score)
	}
	// The interval must use three paired observations, not six games.
	if got.ScoreLo < 0.009 || got.ScoreLo > 0.011 {
		t.Errorf("lower bound = %.6f, want approximately 0.010", got.ScoreLo)
	}
	if got.ScoreHi < 0.989 || got.ScoreHi > 0.991 {
		t.Errorf("upper bound = %.6f, want approximately 0.990", got.ScoreHi)
	}
}

func TestPairedSummaryRejectsMismatchedPositions(t *testing.T) {
	_, err := pairedSummary(stats(result("a", 1)), stats(result("b", 2)))
	if err == nil {
		t.Fatal("pairedSummary unexpectedly accepted mismatched positions")
	}
}
