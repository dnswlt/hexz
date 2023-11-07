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
	remoteURL = flag.String("remote-url", "http://localhost:9094", "URL of the opponent player.")
	// maxIterations = flag.Int("max-iterations", -1, "Number of iterations each player is allowed to make per move. -1 for unlimited.")
	thinkTime       = flag.Duration("think-time", 1*time.Second, "Maximum thinking time per move.")
	remoteThinkTime = flag.Duration("remote-think-time", 3*time.Second, "Maximum thinking time per move for remote player.")
	svgOutputFile   = flag.String("svg-file", "/tmp/nbench.html", "File to which SVG output is written.")
)

func playGame() error {
	ge := hexz.NewGameEngineFlagz()
	cpuPlayers := []hexz.CPUPlayer{
		hexz.NewLocalCPUPlayer("P1", *thinkTime),
		hexz.NewRemoteCPUPlayer(hexz.PlayerId("P2"), *remoteURL, *remoteThinkTime),
	}
	nMoves := 0
	boards := []*hexz.Board{}
	stats := []*pb.SuggestMoveStats{}
	for !ge.IsDone() {
		turn := ge.B.Turn
		mv, mvStats, err := cpuPlayers[turn-1].SuggestMove(context.Background(), ge)
		if err != nil {
			return fmt.Errorf("remote SuggestMove: %v", err)
		}
		fmt.Printf("P%d suggested move %v\n", turn, mv.String())
		if err := ge.MakeMoveError(*mv); err != nil {
			return fmt.Errorf("make move for P%d: %s %w", turn, mv.String(), err)
		}
		log.Printf("Score after %d moves: %v", nMoves, ge.B.Score)
		nMoves++
		boards = append(boards, ge.B.Copy())
		stats = append(stats, mvStats)
		if *svgOutputFile != "" {
			hexz.ExportSVG(*svgOutputFile, boards, nil)
		}
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
