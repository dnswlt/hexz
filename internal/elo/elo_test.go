package elo

import (
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
func TestEloScores(t *testing.T) {
	keys := []*pb.ModelKey{
		{Name: "test", Checkpoint: 1},
		{Name: "test", Checkpoint: 2},
		{Name: "test", Checkpoint: 3},
	}
	tests := []struct {
		name  string
		input []*npb.BenchmarkResult
		want  []*EloScore
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
			want: []*EloScore{
				{Key: keys[0], Games: 1, Wins: 1, Score: 1516},
				{Key: keys[1], Games: 1, Score: 1484},
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
			want: []*EloScore{
				{Key: keys[0], Games: 10, Wins: 10, Score: 1660},
				{Key: keys[1], Games: 10, Score: 1340},
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
			want: []*EloScore{
				{Key: keys[0], Games: 2, Wins: 2, Score: 1530.496882},
				{Key: keys[1], Games: 2, Wins: 1, Score: 1500.736306},
				{Key: keys[2], Games: 2, Wins: 0, Score: 1468.766810},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			elos := Scores(tc.input)
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
