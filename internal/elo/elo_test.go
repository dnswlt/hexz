package elo

import (
	"math"
	"os"
	"path"
	"testing"
	"time"

	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	npb "github.com/dnswlt/hexz/pkg/nbenchpb"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func pRes(key *pb.ModelKey, wins int32) *npb.BenchmarkResult_PlayerResult {
	return &npb.BenchmarkResult_PlayerResult{
		ModelKey: key,
		Wins:     wins,
	}
}

func approxEq(a, b, thresh float64) bool {
	return math.Abs(a-b) < thresh
}

func TestExpected(t *testing.T) {
	tests := []struct {
		eloSelf  float64
		eloOther float64
		want     float64
	}{
		{1500, 1600, 0.359935},
		{1600, 1500, 1 - 0.359935},
		{1600, 2000, 1.0 / 11},
		{2000, 1600, 10.0 / 11},
		{2400, 1600, 100.0 / 101},
		// A player with a log10(2) * scaling factor higher Elo should win twice as often.
		{1000, 1000 + (scalingFactor * math.Log10(2)), 1.0 / 3},
	}
	for _, tc := range tests {
		if got := expected(tc.eloSelf, tc.eloOther); !approxEq(got, tc.want, 1e-6) {
			t.Errorf("expected(%v, %v) == %v, want %v", tc.eloSelf, tc.eloOther, got, tc.want)
		}
	}
}

func TestSequentialEloUpdates2500(t *testing.T) {
	// What is the difference between updating Elo for 10 games individually
	// vs doing it in one step?
	tests := []struct {
		eloStart1     float64
		eloStart2     float64
		scores        []float64
		wantElo1Incr  float64
		wantElo1Batch float64
		wantElo2Incr  float64
		wantElo2Batch float64
	}{
		{
			eloStart1: 1500,
			eloStart2: 2500,
			scores: []float64{
				1, 1, 1, 1, 0, 0, 0, 1, 1, 0.5,
			},
			wantElo1Incr:  1704.1,
			wantElo1Batch: 1707.0,
			wantElo2Incr:  2295.9,
			wantElo2Batch: 2293.0,
		},
		{
			eloStart1: 1500,
			eloStart2: 1500,
			scores: []float64{
				1, 1, 1, 1, 0, 0, 0, 1, 1, 0.5,
			},
			wantElo1Incr:  1526.4,
			wantElo1Batch: 1548.0,
			wantElo2Incr:  1473.6,
			wantElo2Batch: 1452.0,
		},
	}
	for _, tc := range tests {
		var elo1 float64 = tc.eloStart1
		var elo2 float64 = tc.eloStart2
		var scoreSum float64
		for _, score := range tc.scores {
			expect := expected(elo1, elo2)
			elo1 = elo1 + 32*(score-expect)
			elo2 = elo2 + 32*(expect-score) // (1 - score) - (1 - expect) == expect - score
			scoreSum += score
		}
		expectedSum := float64(len(tc.scores)) * expected(tc.eloStart1, tc.eloStart2)
		elo1Batch := tc.eloStart1 + 32*(scoreSum-expectedSum)
		elo2Batch := tc.eloStart2 + 32*(expectedSum-scoreSum)
		if !approxEq(elo1, tc.wantElo1Incr, 0.1) || !approxEq(elo1Batch, tc.wantElo1Batch, 0.1) {
			t.Errorf("Elo1 after %d updates: %.1f, Elo1 after batch update: %.1f", len(tc.scores), elo1, elo1Batch)
		}
		if !approxEq(elo2, tc.wantElo2Incr, 0.1) || !approxEq(elo2Batch, tc.wantElo2Batch, 0.1) {
			t.Errorf("Elo2 after %d updates: %.1f, Elo1 after batch update: %.1f", len(tc.scores), elo2, elo2Batch)
		}
	}
}

func TestEloScores(t *testing.T) {
	keys := []*pb.ModelKey{
		{Name: "test", Checkpoint: 1},
		{Name: "test", Checkpoint: 2},
		{Name: "test", Checkpoint: 3},
	}
	tests := []struct {
		name  string
		input []*npb.BenchmarkResult
		want  []*Rating
	}{
		{
			name:  "empty",
			input: nil,
			want:  nil,
		},
		{
			name: "single game",
			input: []*npb.BenchmarkResult{
				{
					Games:    1,
					P1Result: pRes(keys[0], 1),
					P2Result: pRes(keys[1], 0),
				},
			},
			want: []*Rating{
				{Key: keys[0], Games: 1, Wins: 1, Rating: 1516},
				{Key: keys[1], Games: 1, Rating: 1484},
			},
		},
		{
			name: "ten games",
			input: []*npb.BenchmarkResult{
				{
					Games:    10,
					P1Result: pRes(keys[0], 10),
					P2Result: pRes(keys[1], 0),
				},
			},
			want: []*Rating{
				{Key: keys[0], Games: 10, Wins: 10, Rating: 1660},
				{Key: keys[1], Games: 10, Rating: 1340},
			},
		},
		{
			name: "3 players, all games",
			input: []*npb.BenchmarkResult{
				{
					Games:    1,
					P1Result: pRes(keys[0], 1),
					P2Result: pRes(keys[1], 0),
				},
				{
					Games:    1,
					P1Result: pRes(keys[1], 1),
					P2Result: pRes(keys[2], 0),
				},
				{
					Games:    1,
					P1Result: pRes(keys[0], 1),
					P2Result: pRes(keys[2], 0),
				},
			},
			want: []*Rating{
				{Key: keys[0], Games: 2, Wins: 2, Rating: 1530.496882},
				{Key: keys[1], Games: 2, Wins: 1, Rating: 1500.736306},
				{Key: keys[2], Games: 2, Wins: 0, Rating: 1468.766810},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			elos := Ratings(tc.input)
			if diff := cmp.Diff(tc.want, elos, cmp.Comparer(proto.Equal), cmpopts.EquateApprox(1e-3, 1e-3)); diff != "" {
				t.Errorf("Unexpected Scores() result (-want +got): %s", diff)
			}
		})
	}
}

func TestAppendRead(t *testing.T) {
	// Append a few records, read them back, expect the same results.
	tempDir := t.TempDir()
	statsFile := path.Join(tempDir, "results.jsonl")
	results := []*npb.BenchmarkResult{
		{
			Started:         timestamppb.New(time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)),
			DurationSeconds: (3 * time.Minute).Seconds(),
			Games:           10,
			P1Result:        pRes(&pb.ModelKey{Name: "test", Checkpoint: 1}, 5),
			P2Result:        pRes(&pb.ModelKey{Name: "test", Checkpoint: 2}, 5),
			Args:            []string{"--foo"},
		},
		{
			Started:         timestamppb.New(time.Date(2024, 1, 2, 12, 0, 0, 0, time.UTC)),
			DurationSeconds: (5 * time.Minute).Seconds(),
			Games:           20,
			P1Result:        pRes(&pb.ModelKey{Name: "test", Checkpoint: 1}, 15),
			P2Result:        pRes(&pb.ModelKey{Name: "test", Checkpoint: 2}, 5),
			Args:            []string{"--bar"},
		},
	}
	for _, r := range results {
		if err := AppendStats(statsFile, r); err != nil {
			t.Fatalf("AppendStats failed: %v", err)
		}
		// For fun, add a comment line and a newline after each record
		f, err := os.Open(statsFile)
		if err != nil {
			t.Fatalf("Cannot open stats file: %v", err)
		}
		f.WriteString("# this is a comment\n")
		f.WriteString("\n")
		f.Close()
	}
	readResults, err := ReadResults(statsFile)
	if err != nil {
		t.Fatalf("ReadStats failed: %v", err)
	}
	if diff := cmp.Diff(results, readResults, protocmp.Transform()); diff != "" {
		t.Errorf("Read results differ from original (-want +got): %s", diff)
	}
}
