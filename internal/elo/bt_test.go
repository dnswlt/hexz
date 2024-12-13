package elo

import (
	"testing"

	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	npb "github.com/dnswlt/hexz/pkg/nbenchpb"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/testing/protocmp"
)

func mk(cp int) *pb.ModelKey {
	return &pb.ModelKey{Name: "test", Checkpoint: int32(cp)}
}

func TestBradleyTerry(t *testing.T) {
	tests := []struct {
		results []*npb.BenchmarkResult
		want    []*Rating
	}{
		{
			results: []*npb.BenchmarkResult{
				{
					Games:    3,
					P1Result: pRes(mk(1), 2),
					P2Result: pRes(mk(2), 1),
				},
			},
			want: []*Rating{
				{
					Key:    mk(1),
					Games:  3,
					Wins:   2,
					Draws:  0,
					Rating: 2.0 / 3,
				},
				{
					Key:    mk(2),
					Games:  3,
					Wins:   1,
					Draws:  0,
					Rating: 1.0 / 3,
				},
			},
		},
		{
			results: []*npb.BenchmarkResult{
				{
					Games:    3,
					P1Result: pRes(mk(1), 2),
					P2Result: pRes(mk(2), 1),
				},
				{
					Games:    3,
					P1Result: pRes(mk(2), 2),
					P2Result: pRes(mk(3), 1),
				},
			},
			want: []*Rating{
				{
					Key:    mk(1),
					Games:  3,
					Wins:   2,
					Draws:  0,
					Rating: 0.571428,
				},
				{
					Key:    mk(2),
					Games:  6,
					Wins:   3,
					Draws:  0,
					Rating: 0.285714,
				},
				{
					Key:    mk(3),
					Games:  3,
					Wins:   1,
					Draws:  0,
					Rating: 0.142857,
				},
			},
		},
	}
	for _, tc := range tests {
		ratings := BradleyTerry(tc.results)
		if len(ratings) != len(tc.want) {
			t.Fatalf("Wrong number of ratings: want %d, got %d", len(tc.want), len(ratings))
		}
		if diff := cmp.Diff(tc.want, ratings, cmpopts.EquateApprox(1e-3, 1e-3), protocmp.Transform()); diff != "" {
			t.Errorf("Wrong model in position 0 (-want +got): %v", diff)
		}
	}
}

func TestBradleyTerryRatings(t *testing.T) {
	tests := []struct {
		name    string
		results []*npb.BenchmarkResult
		want    map[string]float64
	}{
		{
			name: "3 players, circular",
			results: []*npb.BenchmarkResult{
				{Games: 1, P1Result: pRes(mk(1), 1), P2Result: pRes(mk(2), 0)},
				{Games: 1, P1Result: pRes(mk(2), 1), P2Result: pRes(mk(3), 0)},
				{Games: 1, P1Result: pRes(mk(3), 1), P2Result: pRes(mk(1), 0)},
			},
			want: map[string]float64{
				mkey(mk(1)): 1.0,
				mkey(mk(2)): 1.0,
				mkey(mk(3)): 1.0,
			},
		},
		{
			name: "4 players, one dominates, one always loses",
			results: []*npb.BenchmarkResult{
				{Games: 1, P1Result: pRes(mk(1), 1), P2Result: pRes(mk(2), 0)},
				{Games: 1, P1Result: pRes(mk(1), 1), P2Result: pRes(mk(3), 0)},
				{Games: 1, P1Result: pRes(mk(1), 1), P2Result: pRes(mk(4), 0)},
				{Games: 1, P1Result: pRes(mk(2), 1), P2Result: pRes(mk(3), 0)},
				{Games: 1, P1Result: pRes(mk(2), 1), P2Result: pRes(mk(4), 0)},
				{Games: 1, P1Result: pRes(mk(3), 1), P2Result: pRes(mk(4), 0)},
			},
			want: map[string]float64{
				mkey(mk(1)): 3.409,
				mkey(mk(2)): 1.460,
				mkey(mk(3)): 0.685,
				mkey(mk(4)): 0.293,
			},
		},
		{
			name: "2 players, one wins twice as often",
			results: []*npb.BenchmarkResult{
				{Games: 15, P1Result: pRes(mk(1), 10), P2Result: pRes(mk(2), 5)},
			},
			want: map[string]float64{
				mkey(mk(1)): 1.381,
				mkey(mk(2)): 0.724,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ratings := BradleyTerry(tc.results)
			if len(ratings) != len(tc.want) {
				t.Fatalf("Wrong number of ratings: want %d, got %d", len(tc.want), len(ratings))
			}
			for i := 0; i < len(ratings); i++ {
				want := tc.want[mkey(ratings[i].Key)]
				got := ratings[i].Rating
				if !cmp.Equal(want, got, cmpopts.EquateApprox(1e-3, 1e-3)) {
					t.Errorf("Rating for %v not equal: want: %.3f, got: %.3f", ratings[i].Key, want, got)
				}
			}
		})
	}
}
