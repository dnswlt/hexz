package hexz

import (
	"math"
	"testing"
	"time"
)

func BenchmarkMCTSPlayRandomGame(b *testing.B) {
	ge := NewGameEngineFlagz()
	mcts := NewMCTS()
	for i := 0; i < b.N; i++ {
		r := 0
		c := 0
	Outer:
		for ; r < len(ge.B.Fields); r++ {
			for ; c < len(ge.B.Fields[r]); c++ {
				if !ge.B.Fields[r][c].occupied() {
					break Outer
				}
			}
		}
		mcts.playRandomGame(ge.Clone(), &mcNode{
			r:        r,
			c:        c,
			cellType: cellFlag,
			turn:     1,
		})
	}
}

func TestMCTSFull(t *testing.T) {
	if testing.Short() {
		t.Skip("Don't run full MCTS simulation in -short mode.")
	}
	// Play one full game without crashing
	thinkTime := time.Duration(10) * time.Millisecond

	ge := NewGameEngineFlagz()

	mcts := []*MCTS{
		NewMCTS(),
		NewMCTS(),
	}
	for !ge.IsDone() {
		ti := ge.Board().Turn - 1
		m, _ := mcts[ti].SuggestMove(ge, thinkTime)
		if !ge.MakeMove(m) {
			t.Fatal("Cannot make move")
		}
	}
}

func TestMCTSNoThinkTime(t *testing.T) {
	// Even with zero and negative think time, MCTS should yield a valid move suggestion.
	ge := NewGameEngineFlagz()

	mcts := NewMCTS()
	thinkTime := time.Duration(0)
	m, _ := mcts.SuggestMove(ge, thinkTime)
	if !ge.MakeMove(m) {
		t.Errorf("Cannot make move %v", m)
	}
	thinkTime = -3 * time.Second
	m, _ = mcts.SuggestMove(ge, thinkTime)
	if !ge.MakeMove(m) {
		t.Errorf("Cannot make move %v", m)
	}
}

// Quick check that sqrt and log are just as fast on float32 as they are on
// float64, despite the casting nuisances.

/*
On an M1 Pro, float64 and float32 results are identical. But on my Intel NUC float32 is way faster:

benchstat sqrt32.txt sqrt64.txt
goos: linux
goarch: amd64
pkg: github.com/dnswlt/hackz/hexz
cpu: Intel(R) Core(TM) i5-6260U CPU @ 1.80GHz
       │ sqrt32.txt  │             sqrt64.txt              │
       │   sec/op    │   sec/op     vs base                │
Sqrt-4   417.6µ ± 0%   626.7µ ± 0%  +50.06% (p=0.000 n=10)

*/

func BenchmarkSqrt32(b *testing.B) {
	const iterations = 1000
	for i := 0; i < b.N; i++ {
		var s float32
		for x := float32(1); x < iterations; x += 1 {
			s += float32(math.Sqrt(float64(x)))
		}
		if s < 0 {
			b.Errorf("Wrong sum: %f", s)
		}
	}
}

func BenchmarkSqrt64(b *testing.B) {
	const iterations = 1000
	for i := 0; i < b.N; i++ {
		var s float64
		for x := float64(1); x < iterations; x += 1 {
			s += math.Sqrt(x)
		}
		if s < 0 {
			b.Errorf("Wrong sum: %f", s)
		}
	}
}

/*
Log is slightly slower for float32 on arm64 (M1 Pro):

go test -bench "^BenchmarkLog$" -run ^$ -count=10  -use-float32 | tee log32.txt
go test -bench "^BenchmarkLog$" -run ^$ -count=10  | tee log64.txt
benchstat log32.txt log64.txt
goos: darwin
goarch: arm64
pkg: github.com/dnswlt/hackz/hexz
       │  log32.txt  │             log64.txt              │
       │   sec/op    │   sec/op     vs base               │
Log-10   527.4µ ± 0%   509.3µ ± 0%  -3.44% (p=0.000 n=10)

But on my Intel NUC, log32 is MUCH slower:

benchstat log32.txt log64.txt
goos: linux
goarch: amd64
pkg: github.com/dnswlt/hackz/hexz
cpu: Intel(R) Core(TM) i5-6260U CPU @ 1.80GHz
      │  log32.txt  │              log64.txt              │
      │   sec/op    │   sec/op     vs base                │
Log-4   4.348m ± 0%   1.388m ± 0%  -68.08% (p=0.000 n=10)

How does this make any sense?!

*/

func BenchmarkLog32(b *testing.B) {
	const iterations = 1000
	for i := 0; i < b.N; i++ {
		var s float32
		for x := float32(1); x < iterations; x += 1 {
			s += float32(math.Log(float64(x)))
		}
		if s < 0 {
			b.Errorf("Wrong sum: %f", s)
		}
	}
}

func BenchmarkLog64(b *testing.B) {
	const iterations = 1000
	for i := 0; i < b.N; i++ {
		var s float64
		for x := float64(1); x < iterations; x += 1 {
			s += math.Log(x)
		}
		if s < 0 {
			b.Errorf("Wrong sum: %f", s)
		}
	}
}
