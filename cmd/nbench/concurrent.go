package main

import (
	"context"
	"fmt"
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
	ctx := context.Background()
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
					fmt.Printf("Failed to make a move: %v\n", err)
					return
				}
				fmt.Printf("[worker %d] score at move %d: %v\n", i, ge.Board().Move, ge.Board().Score)
			}
			fmt.Printf("[worker %d] Game over after %d moves. Final score: %v\n", i, ge.Board().Move, ge.Board().Score)
			stats.Add(ge.Winner())
		}(i)
	}

	// Channel to signal termination to the batching RPC goroutine.
	done := make(chan bool)
	// Batching RPC goroutine:
	go func(numWorkers int) {
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
			var moves1, moves2 []*hexz.GameEngineMove
			var err error
			if len(ges1) > 0 {
				fmt.Printf("Making %d requests for P1\n", len(ges1))
				moves1, err = p1.SuggestMoves(ctx, ges1)
				if err != nil {
					fmt.Printf("SuggestMoves failed (P1): %v\n", err)
					// Tell all clients waiting for a response we failed.
					for _, b := range batch {
						close(b.ch)
					}
					return
				}
			}
			if len(ges2) > 0 {
				fmt.Printf("Making %d requests for P2\n", len(ges2))
				moves2, err = p2.SuggestMoves(ctx, ges2)
				if err != nil {
					fmt.Printf("SuggestMoves failed (P2): %v\n", err)
					// Tell all clients waiting for a response we failed.
					for _, b := range batch {
						close(b.ch)
					}
					return
				}
			}
			fmt.Printf("Received %d (%d+%d) move suggestions after %.3f\n", len(batch), len(moves1), len(moves2), time.Since(started).Seconds())
			var j1, j2 int
			for i := 0; i < len(batch); i++ {
				if p1Indices[i] {
					batch[i].ch <- moves1[j1]
					j1++
				} else {
					batch[i].ch <- moves2[j2]
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
			case <-done:
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
	}(numGames)
	// Wait until all workers are done.
	wg.Wait()
	result := "P1 wins"
	if stats.wins[0] == stats.wins[1] {
		result = "draw"
	} else if stats.wins[1] > stats.wins[0] {
		result = "P2 wins"
	}
	fmt.Printf("Best of %d final result after %d games: %s: %v\n", numGames, stats.games, result, stats.wins)
	// Tell the GPU pipeline thread we're done.
	done <- true
}
