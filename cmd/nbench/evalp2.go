package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/pkg/hexz"
)

type EvalResult struct {
	p1Iterations int
	p2Iterations int
	games        int
	wins         [2]int
	done         bool
}

func startHttpServer(ch <-chan EvalResult) *http.Server {
	reqCh := make(chan chan []EvalResult)
	go func() {
		results := []EvalResult{}
		for {
			select {
			case r, ok := <-ch:
				if !ok {
					return // we're done here
				}
				l := len(results)
				if l == 0 || results[l-1].done {
					results = append(results, r)
				} else {
					results[l-1] = r
				}
			case respCh := <-reqCh:
				res := make([]EvalResult, len(results))
				copy(res, results)
				respCh <- res
			}
		}
	}()
	httpServer := &http.Server{
		Addr: ":8088",
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			respCh := make(chan []EvalResult)
			reqCh <- respCh
			results := <-respCh
			fmt.Fprintf(w, "All results:\n")
			for _, r := range results {
				suffix := ""
				if !r.done {
					suffix = fmt.Sprintf(" *%d", r.games)
				}
				fmt.Fprintf(w, "  @(%d:%d): %d-%d%s\n", r.p1Iterations, r.p2Iterations, r.wins[0], r.wins[1], suffix)
			}
		}),
	}
	go func() {
		fmt.Printf("Listening on %s\n", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()
	return httpServer
}

func evalP2() {
	resultCh := make(chan EvalResult)
	defer close(resultCh)
	httpServer := startHttpServer(resultCh)

	var p1, p2 hexz.CPUPlayer
	var err error
	thinkTime := time.Duration(0)
	p1Iterations := p1Options.MaxIterations
	p2Iterations := p2Options.MaxIterations
	results := []EvalResult{}
	p2Lost := false
	remoteCPUClient, err := hexz.NewCPUPlayerServiceClient(p2Options.URL)
	if err != nil {
		log.Fatalf("Failed to create P2 as remote player: %v", err)
	}
	for rounds := 0; rounds < 7; rounds++ {
		p1 = hexz.NewLocalCPUPlayer(api.PlayerId("P1"), thinkTime, p1Iterations)
		p2 = hexz.NewRemoteCPUPlayer(remoteCPUClient, api.PlayerId("P2"), thinkTime, p2Iterations)
		result := EvalResult{
			p1Iterations: p1Iterations,
			p2Iterations: p2Iterations,
		}
		resultCh <- result
		for i := 0; i < *numGames; i++ {
			winner, err := playGame(i, result.wins, p1, p2)
			if err != nil {
				log.Fatalf("playing game failed: %v\n", err)
			}
			if winner > 0 {
				result.wins[winner-1]++
			}
			result.games++
			resultCh <- result
		}
		result.done = true
		fmt.Printf("Final result after %d games: %d-%d\n", *numGames, result.wins[0], result.wins[1])
		resultCh <- result
		results = append(results, result)
		if result.wins[1] == 0 {
			fmt.Printf("P2 did not win a single game with iterations limits %d : %d\n", p1Iterations, p2Iterations)
			p2Lost = true
			break
		}
		p1Iterations *= 2
		fmt.Printf("Doubling iterations for P1 to %d to make it stronger.\n", p1Iterations)
	}
	if !p2Lost {
		fmt.Printf("P2 never lost!\n")
	}
	fmt.Printf("All results:\n")
	for _, r := range results {
		fmt.Printf("  @(%d:%d): %d-%d\n", r.p1Iterations, r.p2Iterations, r.wins[0], r.wins[1])
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		fmt.Printf("Failed to shut down http server: %v\n", err)
	}
}
