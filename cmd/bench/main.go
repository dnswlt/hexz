package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/dnswlt/hexz"
)

var cpuProfile = flag.String("cpuprofile", "", "write cpu profile to file")
var maxRuntime = flag.Duration("maxruntime", time.Duration(30)*time.Second, "maximum time to run the benchmark")
var numGames = flag.Int("numgames", 1, "Number of games to play")
var uctFactor = flag.Float64("uctfactor", 1.0, "weight of the exploration component in the UCT")
var thinkTime = flag.Duration("thinktime", time.Duration(2)*time.Second, "Think time per player and move")
var oppThinkTime = flag.Duration("oppthinktime", time.Duration(2)*time.Second, "Think time per player and move")
var gameHistoryDir = flag.String("gamehistorydir", "", "Directory to which game history of each played game is written")

// Compute the think time we'll give to the player.
// The more confident a player is that they'll win/lose, the less time we give them
// for the next move.
func getThinkTime(stats []*hexz.MCTSStats, isBenchPlayer bool) time.Duration {
	moveThinkTime := *oppThinkTime
	if isBenchPlayer {
		moveThinkTime = *thinkTime
	}
	l := len(stats)
	if l == 0 {
		return moveThinkTime
	}
	minQ := stats[l-1].MinQ()
	maxQ := stats[l-1].MaxQ()
	if math.Min(1-minQ, maxQ) < 0.02 {
		moveThinkTime = time.Duration(100) * time.Millisecond
	}
	return moveThinkTime
}

func main() {
	flag.Parse()
	// Optional profiling
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// Use Ctrl-C to interrupt early, but still print results.
	interrupted := make(chan bool)
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	go func() {
		<-c
		signal.Reset(os.Interrupt)
		fmt.Fprint(os.Stderr, "Interrupted, ending simulation\n")
		interrupted <- true
	}()

	started := time.Now()
	fmt.Println("Started", started)

	type wstat struct {
		wins   int
		losses int
		draws  int
	}
	var wstats [2]wstat
	nRuns := 0
	cancelled := false
	for time.Since(started) < *maxRuntime && nRuns < *numGames && !cancelled {
		nMoves := 0
		ge := hexz.NewGameEngineFlagz()

		var moveStats [2][]*hexz.MCTSStats
		mcts := []*hexz.MCTS{
			hexz.NewMCTS(),
			hexz.NewMCTS(),
		}
		// Evaluate parameters both on P1 and P2
		benchPlayer := nRuns%2 + 1
		mcts[benchPlayer-1].UctFactor = float32(*uctFactor)
		collectBoardHistory := *gameHistoryDir != ""
		gameId := hexz.GenerateGameId()
		var historyWriter *hexz.HistoryWriter
		if collectBoardHistory {
			var err error
			historyWriter, err = hexz.NewHistoryWriter(*gameHistoryDir, gameId)
			if err != nil {
				log.Fatal("cannot create history writer: ", err)
			}
			defer historyWriter.Close()
		}
	Gameloop:
		for !ge.IsDone() && time.Since(started) < *maxRuntime {
			select {
			case <-interrupted:
				cancelled = true
				break Gameloop
			default:
			}
			t := ge.Board().Turn - 1
			moveThinkTime := getThinkTime(moveStats[t], benchPlayer == ge.Board().Turn)
			m, stats := mcts[t].SuggestMove(ge, moveThinkTime)
			if collectBoardHistory {
				boardView := ge.Board().ViewFor(0)
				historyWriter.Write(&hexz.GameHistoryEntry{
					EntryType:  "move",
					Board:      boardView,
					MoveScores: stats.MoveScores(),
				})
			}
			moveStats[t] = append(moveStats[t], stats)
			// fmt.Print(stats)
			if !ge.MakeMove(m) {
				log.Fatal("Cannot make move")
			}
			fmt.Printf("Leaf nodes per depth: %v\n", stats.LeafNodes)
			var lPct strings.Builder
			for i := range stats.LeafNodes {
				fmt.Fprintf(&lPct, " %.1f", (1-float64(stats.LeafNodes[i])/(float64(stats.LeafNodes[i]+stats.BranchNodes[i])))*100)
			}
			fmt.Printf("%% Explored nodes per depth: [%s]\n", lPct.String())
			fmt.Printf("game:%d move:%s S:%d Q:%.3f q:%.3f score:%v t:%v\n",
				nRuns, m.String(), stats.TreeSize, stats.MaxQ(), stats.MinQ(), ge.Board().Score, stats.Elapsed)
			nMoves++
		}
		if ge.IsDone() {
			if collectBoardHistory {
				historyWriter.Write(&hexz.GameHistoryEntry{
					EntryType: "move",
					Board:     ge.Board().ViewFor(0),
					// No MoveScores for terminal board.
				})
				fmt.Printf("Wrote game history with gameId %s\n", gameId)
			}
			winner := ge.Winner()
			if winner == benchPlayer {
				wstats[benchPlayer-1].wins++
			} else if winner > 0 {
				wstats[benchPlayer-1].losses++
			} else {
				wstats[benchPlayer-1].draws++
			}
		}
		// Print aggregate stats of this game
		agg := []*hexz.MCTSStats{{}, {}}
		for i := 0; i < 2; i++ {
			for j := 0; j < len(moveStats[i]); j++ {
				s := moveStats[i][j]
				agg[i].Iterations += s.Iterations
				agg[i].Elapsed += s.Elapsed
				if agg[i].MaxDepth < s.MaxDepth {
					agg[i].MaxDepth = s.MaxDepth
				}
				if agg[i].TreeSize < s.TreeSize {
					agg[i].TreeSize = s.TreeSize
				}
			}
		}
		fmt.Println("Game finished", time.Now())
		fmt.Println("Aggregated stats P1:\n", agg[0].String())
		fmt.Println("Aggregated stats P2:\n", agg[1].String())
		fmt.Printf("=== Current results:\n  As P1: %+v\n  As P2: %+v\n", wstats[0], wstats[1])
		nRuns++
	}
	fmt.Println("Finished", time.Now())
	var sb strings.Builder
	flag.Visit(func(f *flag.Flag) {
		fmt.Fprintf(&sb, " -%s=%v", f.Name, f.Value)
	})
	fmt.Printf("Flags:%s\n", sb.String())
	fmt.Printf("=== Final results:\n  As P1: %+v\n  As P2: %+v\n", wstats[0], wstats[1])
}
