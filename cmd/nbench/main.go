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
	"time"

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
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

func playGame(gameNum int, p1, p2 hexz.CPUPlayer) (winner int, err error) {
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
		log.Printf("Game %d: score after %d moves: %v", gameNum, numMoves, ge.B.Score)
	}
	// TODO: one more ExportSVGWithStats for the final board with no added stats or moves.
	log.Printf("Game %d ended after %d moves. Winner: %d. Final result: %v\n", gameNum, numMoves, ge.Winner(), ge.B.Score)
	return ge.Winner(), nil
}

type EvalResult struct {
	p1Iterations int
	p2Iterations int
	games        int
	score        [2]int
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
				fmt.Fprintf(w, "  @(%d:%d): %d-%d%s\n", r.p1Iterations, r.p2Iterations, r.score[0], r.score[1], suffix)
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
	for rounds := 0; rounds < 7; rounds++ {
		p1 = hexz.NewLocalCPUPlayer(hexz.PlayerId("P1"), thinkTime, p1Iterations)
		p2, err = hexz.NewRemoteCPUPlayer(hexz.PlayerId("P2"), *player2URL, thinkTime, p2Iterations)
		if err != nil {
			fmt.Printf("Failed to create P2 as remote player: %v", err)
			os.Exit(1)
		}
		result := EvalResult{
			p1Iterations: p1Iterations,
			p2Iterations: p2Iterations,
		}
		resultCh <- result
		for i := 0; i < *numGames; i++ {
			winner, err := playGame(p1, p2)
			if err != nil {
				fmt.Printf("playing game failed: %v\n", err)
				os.Exit(1)
			}
			if winner > 0 {
				result.score[winner-1]++
			}
			result.games++
			resultCh <- result
		}
		result.done = true
		fmt.Printf("Final result after %d games: %d-%d\n", *numGames, result.score[0], result.score[1])
		resultCh <- result
		results = append(results, result)
		if result.score[1] == 0 {
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
		fmt.Printf("  @(%d:%d): %d-%d\n", r.p1Iterations, r.p2Iterations, r.score[0], r.score[1])
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		fmt.Printf("Failed to shut down http server: %v\n", err)
	}
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
	var err error
	if *p1MaxIterations > 0 {
		*p1ThinkTime = 0
	}
	if *p2MaxIterations > 0 {
		*p2ThinkTime = 0
	}
	p1Iterations := *p1MaxIterations
	if *player1URL == "" {
		p1 = hexz.NewLocalCPUPlayer(hexz.PlayerId("P1"), *p1ThinkTime, p1Iterations)
	} else {
		p1, err = hexz.NewRemoteCPUPlayer(hexz.PlayerId("P1"), *player1URL, *p1ThinkTime, p1Iterations)
		if err != nil {
			fmt.Printf("Failed to create P1 as remove player: %v", err)
			os.Exit(1)
		}
	}
	if *player2URL == "" {
		p2 = hexz.NewLocalCPUPlayer(hexz.PlayerId("P2"), *p2ThinkTime, *p2MaxIterations)
	} else {
		p2, err = hexz.NewRemoteCPUPlayer(hexz.PlayerId("P2"), *player2URL, *p2ThinkTime, *p2MaxIterations)
		if err != nil {
			fmt.Printf("Failed to create P2 as remove player: %v", err)
			os.Exit(1)
		}
	}
	var wins [2]int
	for i := 0; i < *numGames; i++ {
		winner, err := playGame(i, p1, p2)
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
