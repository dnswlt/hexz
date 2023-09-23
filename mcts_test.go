package hexz

import (
	"math"
	"runtime"
	"strings"
	"testing"
	"time"

	unsafe "unsafe"
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
		root := newMcNode(r, c)
		root.setFlag()
		mcts.playRandomGame(ge.Clone(), &root)
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

func TestMCTSBitOps(t *testing.T) {
	m := newMcNode(3, 4)
	if m.r() != 3 {
		t.Errorf("Wrong row: %d", m.r())
	}
	if m.c() != 4 {
		t.Errorf("Wrong col: %d", m.c())
	}
	m.setTurn(2)
	if m.turn() != 2 {
		t.Errorf("Wrong turn: %d", m.turn())
	}
	m.setFlag()
	if m.cellType() != cellFlag {
		t.Errorf("Flag not set")
	}
	m.setLiveChildren(17)
	if m.liveChildren() != 17 {
		t.Errorf("Wrong live children: %d", m.liveChildren())
	}
	m.decrLiveChildren()
	if m.liveChildren() != 16 {
		t.Errorf("Wrong live children: %d", m.liveChildren())
	}
}

func TestSizeofMcNode(t *testing.T) {
	// Changing the mcNode
	if !strings.Contains(runtime.GOARCH, "64") {
		t.Skip("Only run this test on 64bit architectures")
	}
	want := uintptr(40)
	if got := unsafe.Sizeof(mcNode{}); got != want {
		t.Error("Wrong size: ", got, ", want: ", want)
	}
}

func TestMCTSSingleMove(t *testing.T) {
	// This test makes a single move with a lot of think time. Mostly useful for memory profiling.
	if testing.Short() {
		t.Skip("Don't run MCTS simulation in -short mode.")
	}
	thinkTime := time.Duration(5000) * time.Millisecond

	ge := NewGameEngineFlagz()
	mcts := NewMCTS()
	m, stats := mcts.SuggestMove(ge, thinkTime)
	t.Log(stats)
	if !ge.MakeMove(m) {
		t.Errorf("Cannot make move: %v", m)
	}
}

func TestMCTSSingleMoveMidGame(t *testing.T) {
	// This test makes a single move in the middle of a random game with a lot of think time.
	// Mostly useful for memory profiling.
	if testing.Short() {
		t.Skip("Don't run MCTS simulation in -short mode.")
	}
	thinkTime := time.Duration(5000) * time.Millisecond

	ge := NewGameEngineFlagz()
	// Advance the game a bit.
	for i := 0; i < 30; i++ {
		mv, err := ge.RandomMove()
		if err != nil {
			t.Fatal("No next move: ", err)
		}
		if !ge.MakeMove(mv) {
			t.Fatalf("Cannot make move: %v", mv)
		}
	}
	// Now suggest a move.
	mcts := NewMCTS()
	m, stats := mcts.SuggestMove(ge, thinkTime)
	t.Log(stats)
	if !ge.MakeMove(m) {
		t.Errorf("Cannot make move: %v", m)
	}
}

func BenchmarkMCTSRun(b *testing.B) {
	gameEngine := NewGameEngineFlagz()
	ge := gameEngine.Clone()
	mcts := NewMCTS()
	for i := 0; i < b.N; i++ {
		ge.copyFrom(gameEngine)
		mcts.run(ge, &mcNode{}, 0)
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
