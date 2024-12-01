// nbench lets a Go MCTS player play against a remote CPU player
// which will usually be a Neural MCTS player. (nbench itself does
// not care about which kind of opponent it is, it just makes RPC
// calls.)
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/pkg/hexz"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

var (
	player1URL       = flag.String("p1-addr", "", "Address for player 1 (empty for the built-in CPU player).")
	player2URL       = flag.String("p2-addr", "localhost:50051", "Address for player 2 (empty for the built-in CPU player).")
	p1ThinkTime      = flag.Duration("p1-think-time", 1*time.Second, "Maximum thinking time per move for P1.")
	p2ThinkTime      = flag.Duration("p2-think-time", 1*time.Second, "Maximum thinking time per move for P2.")
	p1MaxIterations  = flag.Int("p1-max-iter", 0, "Maximum MCTS iterations per move for P1 (overrides p1-think-time if >0).")
	p2MaxIterations  = flag.Int("p2-max-iter", 0, "Maximum MCTS iterations per move for P2 (overrides p2-think-time if >0).")
	svgMoveScoreKind = flag.String("score-kind", "FINAL", "Kind of move scores to add to the SVG output.")
	svgOutputFile    = flag.String("svg-file", "/tmp/nbench.html", "File to which SVG output is written.")
	skipMoves        = flag.Int("skip-moves", 0, "Number of initial moves to make randomly before using the suggestions")
	numGames         = flag.Int("num-games", 1, "Number of games to play")
	p2Eval           = flag.Bool("p2-eval", false, "If true, P1's max iterations are doubled until P2 loses")
)

func playGame(gameNum int, wins [2]int, p1, p2 hexz.CPUPlayer) (winner int, err error) {
	ge := hexz.NewGameEngineFlagz()
	cpuPlayers := []hexz.CPUPlayer{p1, p2}
	scoreKind, found := pb.SuggestMoveStats_ScoreKind_value[*svgMoveScoreKind]
	if !found {
		return 0, fmt.Errorf("invalid score kind: %s", *svgMoveScoreKind)
	}
	numMoves := 0
	// Skip moves, if requested. The idea is that the neural network should learn
	// good end game moves first, because in some sense the feedback obtained from
	// the final outcome of the game is more closely connected to the final moves
	// than with the initial ones, especially for networks that haven't seen many
	// games yet.
	for i := 0; i < *skipMoves; i++ {
		mv, err := ge.RandomMove()
		if err != nil {
			return 0, fmt.Errorf("get random move: %w", err)
		}
		if err := ge.MakeMoveError(mv); err != nil {
			return 0, fmt.Errorf("make random move: %w", err)
		}
		numMoves++
	}
	boards := []*hexz.Board{}
	stats := []*pb.SuggestMoveStats{}
	moves := []*hexz.GameEngineMove{}
	for !ge.IsDone() {
		turn := ge.B.Turn
		started := time.Now()
		mv, mvStats, err := cpuPlayers[turn-1].SuggestMove(context.Background(), ge)
		duration := time.Since(started)
		if err != nil {
			return 0, fmt.Errorf("remote SuggestMove failed: %v", err)
		}
		log.Printf("P%d suggested move %v in %dms\n", turn, mv.String(), duration.Milliseconds())
		boards = append(boards, ge.B.Copy())
		stats = append(stats, mvStats)
		moves = append(moves, mv)
		if *svgOutputFile != "" {
			// Update SVG after every move, so we can follow along as the game proceeds.
			hexz.ExportSVGWithStats(*svgOutputFile, boards, moves, stats, pb.SuggestMoveStats_ScoreKind(scoreKind), nil)
		}
		if err := ge.MakeMoveError(*mv); err != nil {
			return 0, fmt.Errorf("make move for P%d: %s %w", turn, mv.String(), err)
		}
		numMoves++
		log.Printf("Wins: %v. Game %d: score after %d moves: %v", wins, gameNum, numMoves, ge.B.Score)
	}
	// TODO: one more ExportSVGWithStats for the final board with no added stats or moves.
	log.Printf("Game %d ended after %d moves. Winner: %d. Final result: %v\n", gameNum, numMoves, ge.Winner(), ge.B.Score)
	return ge.Winner(), nil
}

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
	p1Iterations := *p1MaxIterations
	p2Iterations := *p2MaxIterations
	results := []EvalResult{}
	p2Lost := false
	remoteCPUClient, err := hexz.NewCPUPlayerServiceClient(*player2URL)
	if err != nil {
		fmt.Printf("Failed to create P2 as remote player: %v", err)
		os.Exit(1)
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
				fmt.Printf("playing game failed: %v\n", err)
				os.Exit(1)
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
			fmt.Printf("[worker %d] Game over after %d moves. Final score: %v", i, ge.Board().Move, ge.Board().Score)
		}(i)
	}

	done := make(chan bool)

	go func(numWorkers int) {
		processBatch := func(batch []request) {
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
			moves1, err := p1.SuggestMoves(ctx, ges1)
			if err != nil {
				fmt.Printf("SuggestMoves failed (P1): %v\n", err)
				// Tell all clients waiting for a response we failed.
				for _, b := range batch {
					close(b.ch)
				}
				return
			}
			moves2, err := p2.SuggestMoves(ctx, ges2)
			if err != nil {
				fmt.Printf("SuggestMoves failed (P2): %v\n", err)
				// Tell all clients waiting for a response we failed.
				for _, b := range batch {
					close(b.ch)
				}
				return
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
		// Let all workers register before sending any requests.
		// Otherwise, the first worker might already trigger a request
		// for its first move and become out of sync (w.r.t. its active turn) with the others.
		for activeWorkers < numWorkers {
			<-registerChan
			activeWorkers++
		}
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
	wg.Wait()
	done <- true
}

func main() {
	flag.Parse()
	if len(flag.Args()) > 0 {
		fmt.Printf("Unexpected extra args: %v\n", flag.Args())
		os.Exit(1)
	}
	if *p2Eval {
		if *p1MaxIterations == 0 || *p2MaxIterations == 0 {
			fmt.Printf("For --p2-eval mode you have to specify max iterations")
			os.Exit(2)
		}
		evalP2()
		return
	}
	var p1, p2 hexz.CPUPlayer
	if *p1MaxIterations > 0 {
		*p1ThinkTime = 0
	}
	if *p2MaxIterations > 0 {
		*p2ThinkTime = 0
	}
	if *player1URL == "" {
		p1 = hexz.NewLocalCPUPlayer(api.PlayerId("P1"), *p1ThinkTime, *p1MaxIterations)
	} else {
		remoteCPUClient, err := hexz.NewCPUPlayerServiceClient(*player1URL)
		if err != nil {
			fmt.Printf("Failed to create P1 as remote player: %v", err)
			os.Exit(1)
		}
		p1 = hexz.NewRemoteCPUPlayer(remoteCPUClient, api.PlayerId("P1"), *p1ThinkTime, *p1MaxIterations)
	}
	if *player2URL == "" {
		p2 = hexz.NewLocalCPUPlayer(api.PlayerId("P2"), *p2ThinkTime, *p2MaxIterations)
	} else {
		remoteCPUClient, err := hexz.NewCPUPlayerServiceClient(*player1URL)
		if err != nil {
			fmt.Printf("Failed to create P2 as remove player: %v", err)
			os.Exit(1)
		}
		p2 = hexz.NewRemoteCPUPlayer(remoteCPUClient, api.PlayerId("P2"), *p2ThinkTime, *p2MaxIterations)
	}
	if *player1URL != "" && *player2URL != "" && *numGames > 1 {
		// This is ML model evaluation mode. Play concurrently to benefit from concurrent requests to the GPU.
		playConcurrent(p1.(*hexz.RemoteCPUPlayer), p2.(*hexz.RemoteCPUPlayer), *numGames)
		return
	}
	var wins [2]int
	for i := 0; i < *numGames; i++ {
		winner, err := playGame(i, wins, p1, p2)
		if err != nil {
			fmt.Printf("playing game failed: %v\n", err)
			os.Exit(1)
		}
		if winner > 0 {
			wins[winner-1]++
		}
	}
	fmt.Printf("Final result after %d games: %d-%d\n", *numGames, wins[0], wins[1])
}
