package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/dnswlt/hexz/pkg/hexz"
	"golang.org/x/sync/errgroup"
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
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
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
	cancel()
	log.Printf("Playing %s vs. %s\n", p1Name, p2Name)

	var wg sync.WaitGroup
	type request struct {
		ge *hexz.GameEngineFlagz
		ch chan *hexz.GameEngineMove
	}
	registerChan := make(chan int)
	unregisterChan := make(chan int)
	requestChan := make(chan request)

	stats := &ResultStats{}
	// Start "worker" goroutines playing games.
	for i := 0; i < numGames; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			registerChan <- i
			defer func() {
				unregisterChan <- i
			}()
			ge := hexz.NewGameEngineFlagz()
			for !ge.IsDone() {
				replyChan := make(chan *hexz.GameEngineMove)
				requestChan <- request{
					ge: ge,
					ch: replyChan,
				}
				mv, ok := <-replyChan
				if !ok {
					// Channel was closed, abort
					return
				}
				if err := ge.MakeMoveError(*mv); err != nil {
					log.Printf("Failed to make a move: %v\n", err)
					return
				}
				log.Printf("[worker %d] score at move %d: %v\n", i, ge.Board().Move, ge.Board().Score)
			}
			log.Printf("[worker %d] Game over after %d moves. Final score: %v\n", i, ge.Board().Move, ge.Board().Score)
			stats.Add(ge.Winner())
		}(i)
	}

	// Channel to signal termination to the batching RPC goroutine.
	bgCtx, cancel := context.WithCancel(context.Background())
	var bgWg sync.WaitGroup
	bgWg.Add(1)
	// Batching RPC goroutine:
	go func(ctx context.Context, numWorkers int) {
		defer bgWg.Done()
		// Helper function to process a full batch.
		processBatch := func(batch []request) {
			// Requests can be for P1 or P2 moves. We need to split them
			// and merge the results.
			p1Indices := make(map[int]bool)
			var ges1, ges2 []*hexz.GameEngineFlagz
			for i, r := range batch {
				if r.ge.Board().Turn == 1 {
					ges1 = append(ges1, r.ge)
					p1Indices[i] = true
				} else {
					ges2 = append(ges2, r.ge)
				}
			}
			started := time.Now()
			moves := [][]*hexz.GameEngineMove{nil, nil}
			g, ctx := errgroup.WithContext(ctx)
			if len(ges1) > 0 {
				g.Go(func() error {
					ms, err := p1.SuggestMoves(ctx, ges1)
					if err != nil {
						log.Printf("SuggestMoves failed (P1): %v\n", err)
						return err
					}
					moves[0] = ms
					return nil
				})
			}
			if len(ges2) > 0 {
				g.Go(func() error {
					ms, err := p2.SuggestMoves(ctx, ges2)
					if err != nil {
						log.Printf("SuggestMoves failed (P2): %v\n", err)
						return err
					}
					moves[1] = ms
					return nil
				})
			}
			err := g.Wait()
			if err != nil {
				// Fatal: RPC failed. Tell all worker that we're done.
				for _, b := range batch {
					close(b.ch)
				}
				return
			}
			log.Printf("Received %d (%d+%d) move suggestions after %.3f\n", len(batch), len(moves[0]), len(moves[1]), time.Since(started).Seconds())
			var j1, j2 int
			for i := 0; i < len(batch); i++ {
				if p1Indices[i] {
					batch[i].ch <- moves[0][j1]
					j1++
				} else {
					batch[i].ch <- moves[1][j2]
					j2++
				}
			}
		}
		activeWorkers := 0
		// Let all workers register before making any RPC requests.
		// Otherwise, the first worker might already trigger a request
		// for its first move and get out of sync (w.r.t. its active turn) with the others,
		// resulting in suboptimal batching.
		for activeWorkers < numWorkers {
			<-registerChan
			activeWorkers++
		}
		// Now receive requests, batch them and make a SuggestMoves RPC.
		var batch []request
		for {
			select {
			case <-ctx.Done():
				return
			case r := <-requestChan:
				batch = append(batch, r)
				if len(batch) >= activeWorkers {
					processBatch(batch)
					batch = nil
				}
			case <-unregisterChan:
				activeWorkers--
				if len(batch) > 0 && len(batch) >= activeWorkers {
					processBatch(batch)
					batch = nil
				}
			}
		}
	}(bgCtx, numGames)
	// Wait until all workers are done.
	wg.Wait()
	// Tell the GPU pipeline thread we're done.
	cancel()
	bgWg.Wait()

	// Print results.
	if stats.Games() != numGames {
		log.Printf("Benchmark was aborted. Intermediate result: %v", stats)
		return
	}
	result := "P1 wins"
	if stats.wins[0] == stats.wins[1] {
		result = "draw"
	} else if stats.wins[1] > stats.wins[0] {
		result = "P2 wins"
	}

	log.Printf("Final result: best of %d: %s vs %s: %s %d-%d\n", numGames, p1Name, p2Name, result, stats.wins[0], stats.wins[1])
}
