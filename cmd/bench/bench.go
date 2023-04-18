package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/dnswlt/hexz"
)

var cpuProfile = flag.String("cpuprofile", "", "write cpu profile to file")
var maxRuntime = flag.Duration("maxruntime", time.Duration(30)*time.Second, "maximum time to run the benchmark")
var maxFlagPositions = flag.Int("maxflagpos", 5, "maximum number of positions to randomly try placing a flag on")
var uctFactor = flag.Float64("uctfactor", 1.0, "weight of the exploration component in the UCT")
var thinkTime = flag.Duration("thinktime", time.Duration(2)*time.Second, "Think time per player and move")
var oppThinkTime = flag.Duration("oppthinktime", time.Duration(2)*time.Second, "Think time per player and move")
var flagsFirst = flag.Bool("flagsfirst", false, "If true, flags will be played first")

// Compute the think time we'll give to the player.
// If the player was 98% confident to win with any move on the last move,
// we only grant it a very small amount of think time, to speed up the
// process. Same if it thinks it will lose.
func getThinkTime(stats []*hexz.MCTSStats, isBenchPlayer bool) time.Duration {
	moveThinkTime := *oppThinkTime
	if isBenchPlayer {
		moveThinkTime = *thinkTime
	}
	l := len(stats)
	if l > 0 && (stats[l-1].MinQ() >= 0.98 || stats[l-1].MinQ() <= 0.02) {
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
	src := rand.NewSource(time.Now().UnixNano())
	for time.Since(started) < *maxRuntime && !cancelled {
		nMoves := 0
		ge := hexz.NewGameEngineFlagz(src)

		var moveStats [2][]*hexz.MCTSStats
		mcts := []*hexz.MCTS{
			hexz.NewMCTS(),
			hexz.NewMCTS(),
		}
		// Evaluate parameters both on P1 and P2
		benchPlayer := nRuns%2 + 1
		mcts[benchPlayer-1].MaxFlagPositions = *maxFlagPositions
		mcts[benchPlayer-1].UctFactor = *uctFactor
		mcts[benchPlayer-1].FlagsFirst = *flagsFirst

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
			moveStats[t] = append(moveStats[t], stats)
			fmt.Print(stats)
			if !ge.MakeMove(m) {
				log.Fatal("Cannot make move")
			}
			fmt.Printf("game:%d move:%d score:%v turn:%d\n", nRuns, nMoves, ge.Board().Score, t+1)
			nMoves++
		}
		if ge.IsDone() {
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
