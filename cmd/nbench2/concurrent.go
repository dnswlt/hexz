package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/dnswlt/hexz/internal/elo"
	"github.com/dnswlt/hexz/pkg/hexz"
	"github.com/dnswlt/hexz/pkg/hexzpb"
	npb "github.com/dnswlt/hexz/pkg/nbenchpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

type ResultStats struct {
	games int
	wins  [2]int
	mut   sync.Mutex
}

func (r *ResultStats) Games() int {
	r.mut.Lock()
	defer r.mut.Unlock()
	return r.games
}

func (r *ResultStats) Wins() [2]int {
	r.mut.Lock()
	defer r.mut.Unlock()
	return r.wins
}

// winner == 0 means draw.
func (r *ResultStats) Add(winner int) {
	r.mut.Lock()
	defer r.mut.Unlock()
	r.games++
	if winner > 0 {
		r.wins[winner-1]++
	}
}
func (r *ResultStats) String() string {
	r.mut.Lock()
	defer r.mut.Unlock()
	return fmt.Sprintf("ResultStats(games=%d wins=%v)", r.games, r.wins)
}

type ConcurrentNBench struct {
	p1 *hexz.RemoteCPUPlayer
	p2 *hexz.RemoteCPUPlayer
	// These are populated when the game is started
	p1Key     *hexzpb.ModelKey
	p2Key     *hexzpb.ModelKey
	numGames  int
	stats     *ResultStats
	statsFile string
}

func NewConcurrentNBench(p1, p2 *hexz.RemoteCPUPlayer, numGames int, statsFile string) (*ConcurrentNBench, error) {
	// Get model keys for logging
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	p1Key, err := p1.ModelKey(ctx)
	if err != nil {
		return nil, fmt.Errorf("cannot get model key for P1: %v", err)
	}
	p2Key, err := p2.ModelKey(ctx)
	if err != nil {
		return nil, fmt.Errorf("cannot get model key for P2: %v", err)
	}

	return &ConcurrentNBench{
		p1:        p1,
		p2:        p2,
		p1Key:     p1Key,
		p2Key:     p2Key,
		numGames:  numGames,
		stats:     &ResultStats{},
		statsFile: statsFile,
	}, nil
}

func (nb *ConcurrentNBench) appendStats(started, done time.Time) error {
	res := &npb.BenchmarkResult{
		Started:         tpb.New(started),
		DurationSeconds: done.Sub(started).Seconds(),
		Games:           int32(nb.stats.games),
		P1Result: &npb.BenchmarkResult_PlayerResult{
			ModelKey:   nb.p1Key,
			Wins:       int32(nb.stats.wins[0]),
			Iterations: int32(nb.p1.MaxIterations()),
		},
		P2Result: &npb.BenchmarkResult_PlayerResult{
			ModelKey:   nb.p2Key,
			Wins:       int32(nb.stats.wins[1]),
			Iterations: int32(nb.p2.MaxIterations()),
		},
		Args: os.Args[1:],
	}
	return elo.AppendStats(nb.statsFile, res)
}

func (nb *ConcurrentNBench) P1Name() string {
	return fmt.Sprintf("P1(%s:%d)", nb.p1Key.Name, nb.p1Key.Checkpoint)
}
func (nb *ConcurrentNBench) P2Name() string {
	return fmt.Sprintf("P2(%s:%d)", nb.p2Key.Name, nb.p2Key.Checkpoint)
}

func (nb *ConcurrentNBench) Play(ctx context.Context) error {
	log.Printf("Playing %s vs. %s\n", nb.P1Name(), nb.P2Name())

	var wg sync.WaitGroup

	started := time.Now()
	// Start "worker" goroutines playing games.
	workerCtx, workerCancel := context.WithCancel(ctx)
	defer workerCancel()
	for i := 0; i < nb.numGames; i++ {
		wg.Add(1)
		go func(ctx context.Context, i int) {
			defer wg.Done()
			ge := hexz.NewGameEngineFlagz()
			for !ge.IsDone() {
				var mv *hexz.GameEngineMove
				// Make a direct RPC, relying on batching on the server side.
				p := nb.p1
				if ge.Board().Turn == 2 {
					p = nb.p2
				}
				var err error
				mv, _, err = p.SuggestMove(ctx, ge)
				if err != nil {
					log.Printf("Worker %d: RPC failed: %v", i, err)
					workerCancel() // Kill all workers, we want all or nothing.
					return
				}
				if err := ge.MakeMoveError(*mv); err != nil {
					log.Printf("Failed to make a move: %v\n", err)
					workerCancel()
					return
				}
				log.Printf("[worker %d] score at move %d: %v\n", i, ge.Board().Move, ge.Board().Score)
			}
			log.Printf("[worker %d] Game over after %d moves. Final score: %v\n", i, ge.Board().Move, ge.Board().Score)
			nb.stats.Add(ge.Winner())
		}(workerCtx, i)
	}

	// Wait until all workers are done.
	wg.Wait()
	done := time.Now()
	// Print results.
	if nb.stats.Games() != nb.numGames {
		return fmt.Errorf("benchmark was aborted; intermediate result: %v", nb.stats)
	}
	var result string
	if nb.stats.wins[0] > nb.stats.wins[1] {
		result = "P1 wins"
	} else if nb.stats.wins[0] == nb.stats.wins[1] {
		result = "draw"
	} else {
		result = "P2 wins"
	}

	log.Printf("Final result: best of %d: %s vs %s: %s %d-%d\n", nb.numGames, nb.P1Name(), nb.P2Name(), result, nb.stats.wins[0], nb.stats.wins[1])
	if nb.statsFile != "" {
		err := nb.appendStats(started, done)
		if err != nil {
			return fmt.Errorf("failed to append results to %q: %v", nb.statsFile, err)
		}
	}
	return nil
}
