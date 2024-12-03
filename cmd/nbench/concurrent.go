package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/dnswlt/hexz/pkg/hexz"
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

func playConcurrent(p1, p2 *hexz.RemoteCPUPlayer, numGames int) {
	// Get model keys for logging
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	p1Key, err := p1.ModelKey(ctx)
	if err != nil {
		log.Fatalf("Cannot get model key for P1: %v", err)
	}
	p2Key, err := p2.ModelKey(ctx)
	if err != nil {
		log.Fatalf("Cannot get model key for P2: %v", err)
	}
	p1Name := fmt.Sprintf("P1(%s:%d)", p1Key.Name, p1Key.Checkpoint)
	p2Name := fmt.Sprintf("P2(%s:%d)", p2Key.Name, p2Key.Checkpoint)
	log.Printf("Playing %s vs. %s\n", p1Name, p2Name)

	var wg sync.WaitGroup

	stats := &ResultStats{}

	// Start "worker" goroutines playing games.
	workerCtx, workerCancel := context.WithCancel(context.Background())
	defer workerCancel()
	for i := 0; i < numGames; i++ {
		wg.Add(1)
		go func(ctx context.Context, i int) {
			defer wg.Done()
			ge := hexz.NewGameEngineFlagz()
			for !ge.IsDone() {
				var mv *hexz.GameEngineMove
				// Make a direct RPC, relying on batching on the server side.
				p := p1
				if ge.Board().Turn == 2 {
					p = p2
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
			stats.Add(ge.Winner())
		}(workerCtx, i)
	}

	// Wait until all workers are done.
	wg.Wait()

	// Print results.
	if stats.Games() != numGames {
		log.Printf("Benchmark was aborted. Intermediate result: %v", stats)
		return
	}
	var result string
	if stats.wins[0] > stats.wins[1] {
		result = "P1 wins"
	} else if stats.wins[0] == stats.wins[1] {
		result = "draw"
	} else {
		result = "P2 wins"
	}

	log.Printf("Final result: best of %d: %s vs %s: %s %d-%d\n", numGames, p1Name, p2Name, result, stats.wins[0], stats.wins[1])
}
