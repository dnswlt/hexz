// nbench lets Go MCTS players and remote CPU players play against each other.
// Remote players will usually be Neural MCTS players.
// This tool is also used to evaluate progress in ML training,
// by letting the latest model checkpoint play against previous ones.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/pkg/hexz"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

type PlayerOptions struct {
	URL           string
	thinkTime     time.Duration // Use ThinkTime() to access this field.
	MaxIterations int
}

func (o *PlayerOptions) ThinkTime() time.Duration {
	if o.MaxIterations > 0 {
		return 0
	}
	return o.thinkTime
}

var (
	p1Options        PlayerOptions
	p2Options        PlayerOptions
	svgMoveScoreKind = flag.String("score-kind", "FINAL", "Kind of move scores to add to the SVG output.")
	svgOutputFile    = flag.String("svg-file", "/tmp/nbench.html", "File to which SVG output is written.")
	skipMoves        = flag.Int("skip-moves", 0, "Number of initial moves to make randomly before using the suggestions")
	numGames         = flag.Int("num-games", 1, "Number of games to play")
	p2Eval           = flag.Bool("p2-eval", false, "If true, P1's max iterations are doubled until P2 loses")
	resultLogFile    = flag.String("logfile", "./nbench.jsonl", "JSONlines file to which results are append")
)

func init() {
	flag.StringVar(&p1Options.URL, "p1-addr", "", "Address for player 1 (empty for the built-in CPU player).")
	flag.StringVar(&p2Options.URL, "p2-addr", "localhost:50051", "Address for player 2 (empty for the built-in CPU player).")
	flag.DurationVar(&p1Options.thinkTime, "p1-think-time", 0, "Maximum thinking time per move for P1.")
	flag.DurationVar(&p2Options.thinkTime, "p2-think-time", 0, "Maximum thinking time per move for P2.")
	flag.IntVar(&p1Options.MaxIterations, "p1-max-iter", 800, "Maximum MCTS iterations per move for P1 (overrides p1-think-time if >0).")
	flag.IntVar(&p2Options.MaxIterations, "p2-max-iter", 800, "Maximum MCTS iterations per move for P2 (overrides p2-think-time if >0).")
}

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

func createPlayer(playerId api.PlayerId, opts *PlayerOptions) (hexz.CPUPlayer, error) {
	if opts.URL == "" {
		return hexz.NewLocalCPUPlayer(playerId, opts.ThinkTime(), opts.MaxIterations), nil
	} else {
		remoteCPUClient, err := hexz.NewCPUPlayerServiceClient(opts.URL)
		if err != nil {
			return nil, fmt.Errorf("failed to create remote player: %v", err)
		}
		return hexz.NewRemoteCPUPlayer(remoteCPUClient, playerId, opts.ThinkTime(), opts.MaxIterations), nil
	}
}

func playSequential(p1, p2 hexz.CPUPlayer, numGames int) {
	var wins [2]int
	for i := 0; i < numGames; i++ {
		winner, err := playGame(i, wins, p1, p2)
		if err != nil {
			log.Fatalf("playing game failed: %v\n", err)
		}
		if winner > 0 {
			wins[winner-1]++
		}
	}
	fmt.Printf("Final result after %d games: %d-%d\n", numGames, wins[0], wins[1])
}

func main() {
	flag.Parse()
	if len(flag.Args()) > 0 {
		log.Fatalf("Unexpected extra args: %v\n", flag.Args())
	}
	if *p2Eval {
		if p1Options.MaxIterations == 0 || p2Options.MaxIterations == 0 {
			log.Fatalf("For --p2-eval mode you have to specify max iterations")
		}
		evalP2()
		return
	}
	p1, err := createPlayer(api.PlayerId("P1"), &p1Options)
	if err != nil {
		log.Fatal(err)
	}
	p2, err := createPlayer(api.PlayerId("P2"), &p2Options)
	if err != nil {
		log.Fatal(err)
	}
	if p1Options.URL != "" && p2Options.URL != "" && *numGames > 1 {
		// This is ML model evaluation mode. Play concurrently to benefit from concurrent requests to the GPU.
		nb, err := NewConcurrentNBench(p1.(*hexz.RemoteCPUPlayer), p2.(*hexz.RemoteCPUPlayer), *numGames, *resultLogFile)
		if err != nil {
			log.Fatal(err)
		}
		err = nb.Play()
		if err != nil {
			log.Fatal(err)
		}
		return
	}
	// Play games sequentially
	playSequential(p1, p2, *numGames)
}
