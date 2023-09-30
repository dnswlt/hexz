package main

import (
	"flag"
	"fmt"
	"log"
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
	if maxQ < 0.05 || minQ > 0.95 {
		moveThinkTime = time.Duration(100) * time.Millisecond
	}
	return moveThinkTime
}

func printVisitCountHistograms(visitCounts []map[int]int) {
	for i, m := range visitCounts {
		h := make([]int, 12)
		max := 0
		for vc := range m {
			if vc > max {
				max = vc
			}
		}
		for vcount, nnodes := range m {
			if vcount == 0 {
				h[0] += nnodes
				continue
			}
			h[vcount*10/max+1] += nnodes
		}
		fmt.Printf("Depth %d:\n", i)
		for j := range h {
			if j == 0 {
				fmt.Printf("  0: %d\n", h[j])
				continue
			}
			fmt.Printf("  %d - %d: %d\n", max*(j-1)/len(h), max*j/len(h), h[j])
		}
	}
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
			// hexz.NewMCTSWithMem(8_000_000),
			// hexz.NewMCTSWithMem(8_000_000),
		}
		// Evaluate parameters both when playing as P1 and P2.
		benchPlayer := nRuns%2 + 1
		mcts[benchPlayer-1].UctFactor = *uctFactor
		mcts[benchPlayer-1].ReturnMostFrequentlyVisited = true
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
			hexz.EnableInitialDrawAssumption = (benchPlayer == ge.Board().Turn)
			m, stats := mcts[t].SuggestMove(ge, moveThinkTime)
			printVisitCountHistograms(stats.VisitCounts)
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
