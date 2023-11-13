package hexz

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"testing"
	"time"

	unsafe "unsafe"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func BenchmarkMCTSPlayRandomGame(b *testing.B) {
	ge := NewGameEngineFlagz()
	orig := ge.Clone()
	mcts := NewMCTS()
	for i := 0; i < b.N; i++ {
		ge.copyFrom(orig)
		mcts.playRandomGame(ge)
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

func TestMCTSAverageNumberOfMoves(t *testing.T) {
	// Calculate the average number of moves in a game.
	// Results:
	// moves:7532 sumnx:136363 avg:18.1
	t.Skip("Only used for experimentation")
	sumNextMoves := 0
	moves := 0
	for i := 0; i < 100; i++ {
		ge := NewGameEngineFlagz()
		mcts := NewMCTS()
		for !ge.IsDone() {
			m, stats := mcts.SuggestMoveLimit(ge, 800)
			if !ge.MakeMove(m) {
				t.Fatal("Cannot make move")
			}
			sumNextMoves += stats.BranchNodes[1] + stats.LeafNodes[1]
			moves++
		}
		fmt.Printf("Finished game %d\n", i)

	}
	fmt.Printf("moves:%d sumnx:%d avg:%.1f\n", moves, sumNextMoves, float64(sumNextMoves)/float64(moves))
}

func TestMCTSLosingBoardWithHighQ(t *testing.T) {
	// This test writes examples of winning and losing boards to a file.
	// The idea is to see if hexz can really still result in a loss
	// even if the MCTS is very sure it will win.
	//
	// Comment out the next line to actually run this (long) test.
	t.Skip("To be activated manually for experiments.")

	if testing.Short() {
		t.Skip("Don't run full MCTS simulation in -short mode.")
	}
	thinkTime := time.Duration(1000) * time.Millisecond

	ge := NewGameEngineFlagz()

	mcts := NewMCTS()
	wroteBoard := false
	for !ge.IsDone() {
		m, stats := mcts.SuggestMove(ge, thinkTime)
		if stats.MinQ() > 0.98 && mcts.LosingBoard != nil && mcts.WinningBoard != nil {
			// We are very sure we'll win. Let's see what the boards looks like.
			ExportSVG("losing_board.html",
				[]*Board{
					ge.Board(),
					mcts.WinningBoard,
					mcts.LosingBoard,
				},
				[]string{
					"Current board",
					"Winning board",
					"Losing board",
				})
			wroteBoard = true
			break
		}
		if !ge.MakeMove(m) {
			t.Fatal("Cannot make move")
		}
	}
	if !wroteBoard {
		t.Error("Did not write losing board")
	}
}

func TestMCTSBitOps(t *testing.T) {
	var m mcNode
	m.set(3, 4 /*turn*/, 2, cellFlag)
	if m.r() != 3 {
		t.Errorf("Wrong row: %d", m.r())
	}
	if m.c() != 4 {
		t.Errorf("Wrong col: %d", m.c())
	}
	if m.turn() != 2 {
		t.Errorf("Wrong turn: %d", m.turn())
	}
	if m.cellType() != cellFlag {
		t.Errorf("Flag not set")
	}
	m.set(9, 10, 1, cellNormal)
	if m.r() != 9 {
		t.Errorf("Wrong row: %d", m.r())
	}
	if m.c() != 10 {
		t.Errorf("Wrong col: %d", m.c())
	}
	if m.turn() != 1 {
		t.Errorf("Wrong turn: %d", m.turn())
	}
	if m.cellType() != cellNormal {
		t.Errorf("Wrong cell type: %d", m.cellType())
	}
}

func TestSizeofMcNode(t *testing.T) {
	// Detect if we change the mcNode size, which might lead to increased GC pressure.
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
	root := &mcNode{}
	for i := 0; i < b.N; i++ {
		ge.copyFrom(gameEngine)
		mcts.run(ge, root)
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

func TestMcNodeIncr(t *testing.T) {
	opt := cmpopts.EquateApprox(1e6, 0)
	var m mcNode
	m.set(0, 0, 1, cellNormal)
	// P1 wins.
	m.incr(1)
	if m.wins != 1 {
		t.Errorf("Wrong wins: %d", m.wins)
	}
	if m.count != 1 {
		t.Errorf("Wrong count: %d", m.count)
	}
	if !cmp.Equal(m.Q(), 1.0, opt) {
		t.Errorf("Wrong Q: %f", m.Q())
	}
	// P2 wins.
	m.incr(2)
	if m.wins != 0 {
		t.Errorf("Wrong wins: %d", m.wins)
	}
	if m.count != 2 {
		t.Errorf("Wrong count: %d", m.count)
	}
	if !cmp.Equal(m.Q(), 0.5, opt) {
		t.Errorf("Wrong Q: %f", m.Q())
	}
	// Draw
	m.incr(0)
	if m.wins != 0 {
		t.Errorf("Wrong wins: %d", m.wins)
	}
	if m.count != 3 {
		t.Errorf("Wrong count: %d", m.count)
	}
	if !cmp.Equal(m.Q(), 0.5, opt) {
		t.Errorf("Wrong Q: %f", m.Q())
	}
	// P1 wins again.
	m.incr(1)
	if m.wins != 1 {
		t.Errorf("Wrong wins: %d", m.wins)
	}
	if m.count != 4 {
		t.Errorf("Wrong count: %d", m.count)
	}
	if !cmp.Equal(m.Q(), 0.75, opt) {
		t.Errorf("Wrong Q: %f", m.Q())
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

func BenchmarkTabulatedLog(b *testing.B) {
	const cacheSize = 100_000
	c := make([]float64, cacheSize)
	for i := range c {
		c[i] = math.Log(float64(i))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var s float64
		for j := 1; j < len(c); j++ {
			s += c[j]
		}
		if s < 0 {
			b.Errorf("Wrong sum: %f", s)
		}
	}
}

func BenchmarkRegularLog64(b *testing.B) {
	const cacheSize = 100_000
	for i := 0; i < b.N; i++ {
		var s float64
		for j := 1; j < cacheSize; j++ {
			s += math.Log(float64(j))
		}
		if s < 0 {
			b.Errorf("Wrong sum: %f", s)
		}
	}
}
