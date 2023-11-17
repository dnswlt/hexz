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
	"os"
	"time"

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
)

var (
	player1URL       = flag.String("p1-url", "", "URL for player 1 (empty for the built-in CPU layer).")
	player2URL       = flag.String("p2-url", "http://localhost:8080", "URL for player 2 (empty for the built-in CPU layer).")
	p1ThinkTime      = flag.Duration("p1-think-time", 1*time.Second, "Maximum thinking time per move.")
	p2ThinkTime      = flag.Duration("p2-think-time", 1*time.Second, "Maximum thinking time per move for remote player.")
	svgMoveScoreKind = flag.String("score-kind", "FINAL", "Kind of move scores to add to the SVG output.")
	svgOutputFile    = flag.String("svg-file", "/tmp/nbench.html", "File to which SVG output is written.")
	skipMoves        = flag.Int("skip-moves", 0, "Number of initial moves to make randomly before using the suggestions")
)

func makePlayer(id string, url string, thinkTime time.Duration) hexz.CPUPlayer {
	if url == "" {
		return hexz.NewLocalCPUPlayer(hexz.PlayerId(id), thinkTime)
	} else {
		return hexz.NewRemoteCPUPlayer(hexz.PlayerId(id), url, thinkTime)
	}
}

func playGame() error {
	ge := hexz.NewGameEngineFlagz()
	cpuPlayers := []hexz.CPUPlayer{
		makePlayer("P1", *player1URL, *p1ThinkTime),
		makePlayer("P2", *player2URL, *p2ThinkTime),
	}
	scoreKind, found := pb.SuggestMoveStats_ScoreKind_value[*svgMoveScoreKind]
	if !found {
		return fmt.Errorf("invalid score kind: %s", *svgMoveScoreKind)
	}
	nMoves := 0
	// Skip moves, if requested. The idea is that the neural network should learn
	// good end game moves first, because in some sense the feedback obtained from
	// the final outcome of the game is move closely connected to the final moves
	// than with the initial ones, especially for networks that haven't seen many
	// games yet.
	for i := 0; i < *skipMoves; i++ {
		mv, err := ge.RandomMove()
		if err != nil {
			return fmt.Errorf("get random move: %w", err)
		}
		if err := ge.MakeMoveError(mv); err != nil {
			return fmt.Errorf("make random move: %w", err)
		}
		nMoves++
	}
	boards := []*hexz.Board{}
	stats := []*pb.SuggestMoveStats{}
	moves := []*hexz.GameEngineMove{}
	for !ge.IsDone() {
		turn := ge.B.Turn
		mv, mvStats, err := cpuPlayers[turn-1].SuggestMove(context.Background(), ge)
		if err != nil {
			return fmt.Errorf("remote SuggestMove: %v", err)
		}
		boards = append(boards, ge.B.Copy())
		stats = append(stats, mvStats)
		moves = append(moves, mv)
		if *svgOutputFile != "" {
			// Update SVG after every move, so we can follow along as the game proceeds.
			hexz.ExportSVGWithStats(*svgOutputFile, boards, moves, stats, pb.SuggestMoveStats_ScoreKind(scoreKind), nil)
		}
		fmt.Printf("P%d suggested move %v\n", turn, mv.String())
		if err := ge.MakeMoveError(*mv); err != nil {
			return fmt.Errorf("make move for P%d: %s %w", turn, mv.String(), err)
		}
		nMoves++
		log.Printf("Score after %d moves: %v", nMoves, ge.B.Score)
	}
	fmt.Printf("Game ended after %d moves. Winner: %d. Final result: %v", nMoves, ge.Winner(), ge.B.Score)
	return nil
}

func main() {
	flag.Parse()
	if len(flag.Args()) > 0 {
		fmt.Printf("Unexpected extra args: %v\n", flag.Args())
		os.Exit(1)
	}
	if err := playGame(); err != nil {
		fmt.Printf("playing game failed: %v\n", err)
		os.Exit(1)
	}
}
