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
	mcts := hexz.NewMCTS()
	remote := hexz.NewRemoteCPUPlayer(hexz.PlayerId("P2"), *remoteURL, *remoteThinkTime)
	nMoves := 0
	boards := []*hexz.Board{}
	for !ge.IsDone() {
		if ge.B.Turn == 1 {
			mv, _ := mcts.SuggestMove(ge, *thinkTime)
			fmt.Printf("P1 suggested move %s\n", mv.String())
			if !ge.MakeMove(mv) {
				return fmt.Errorf("make move for P1: %v", mv)
			}
		} else {
			// P2's turn
			mv, err := remote.SuggestMove(context.Background(), ge)
			if err != nil {
				return fmt.Errorf("remote SuggestMove: %v", err)
			}
			req := mv.(hexz.ControlEventMove).MoveRequest
			fmt.Printf("P2 suggested move %v\n", req)
			if err := ge.MakeMoveError(hexz.GameEngineMove{
				PlayerNum: 2,
				Move:      req.Move,
				Row:       req.Row,
				Col:       req.Col,
				CellType:  req.Type,
			}); err != nil {
				return fmt.Errorf("make move for P2: %v %w", req, err)
			}
		}
		log.Printf("Score after %d moves: %v", nMoves, ge.B.Score)
		nMoves++
		boards = append(boards, ge.B.Copy())
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
