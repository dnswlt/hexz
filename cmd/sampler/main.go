// The sampler command generates examples that can be used as training data
// for the HexZero neural network.
package main

import (
	"archive/zip"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path"
	"sync"
	"time"

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
	"google.golang.org/protobuf/proto"
)

func uniformLikelihoods(stats *hexz.MCTSStats) bool {
	minVisits := math.MaxInt64
	maxVisits := 0
	total := 0
	for _, m := range stats.Moves {
		if minVisits > m.Iterations {
			minVisits = m.Iterations
		}
		if maxVisits < m.Iterations {
			maxVisits = m.Iterations
		}
		total = m.Iterations
	}
	return float64(maxVisits)/float64(total)-float64(minVisits)/float64(total) < 0.001
}

func main() {
	runTime := flag.Duration("run-time", 10*time.Second, "Time to run before exiting.")
	iterations := flag.Int("iterations", 800, "Number of MCTS iterations per move.")
	outputDir := flag.String("output-dir", ".", "Directory to which examples are written.")
	maxUniformMoves := flag.Int("max-uniform-moves", 3, "Maximum number of uniform moves before stopping.")
	parallelTasks := flag.Int("parallelism", 1, "Number of goroutines to use for generating examples.")
	flag.Parse()
	if len(flag.Args()) > 0 {
		log.Fatalf("Unexpected args: %v", flag.Args())
	}
	var wg sync.WaitGroup
	countCh := make(chan int)
	go func() {
		started := time.Now()
		totalExamples := 0
		ticker := time.NewTicker(1 * time.Second)
		for {
			select {
			case <-ticker.C:
				elapsed := time.Since(started).Seconds()
				fmt.Printf("Generated %d examples after %.1fs (%.1f/s)\n", totalExamples, elapsed, float64(totalExamples)/elapsed)
				ticker.Reset(1 * time.Second)
			case v, ok := <-countCh:
				if !ok {
					return
				}
				totalExamples += v
			}
		}
	}()
	fmt.Printf("Generating samples for a duration of %v with %v iterations per move. Examples will get written to %q.\n",
		*runTime, *iterations, *outputDir)
	for i := 0; i < *parallelTasks; i++ {
		examplesFile := path.Join(*outputDir, fmt.Sprintf("examples-%s-%03d.zip", time.Now().Format("20060102-150405"), i))
		wg.Add(1)
		go func() {
			defer wg.Done()
			zf, err := os.Create(examplesFile)
			if err != nil {
				log.Fatalf("create file: %v", err)
			}
			defer zf.Close()
			zw := zip.NewWriter(zf)
			defer zw.Close()
			started := time.Now()
			nExamples := 0
			for time.Since(started) < *runTime {
				ge := hexz.NewGameEngineFlagz()
				gameId := hexz.GenerateGameId()
				mcts := hexz.NewMCTS()
				gameExamples := []*pb.MCTSExample{}
				// Counter for number of moves that yielded very uniform move probabilities.
				// The idea is that examples in which every move is equally good and likely
				// don't bring much value. So we stop generating further examples for such games.
				subseqUniformMoves := 0
				nMoves := 0
				for !ge.IsDone() && time.Since(started) < *runTime {
					limit := *iterations
					if nMoves < 6 {
						// Double the "think time" for the first 6 iterations, i.e. where flags are usually placed.
						limit *= 2
					}
					m, stats := mcts.SuggestMove(ge, 0, limit)
					moveStats := make([]*pb.MCTSExample_MoveStats, len(stats.Moves))
					for i, mv := range stats.Moves {
						moveStats[i] = &pb.MCTSExample_MoveStats{
							Move: &pb.GameEngineMove{
								PlayerNum: int32(m.PlayerNum),
								Move:      int32(m.Move),
								Row:       int32(mv.Row),
								Col:       int32(mv.Col),
								CellType:  pb.Field_CellType(mv.CellType),
							},
							Visits:  int32(mv.Iterations),
							WinRate: float32(mv.Q),
						}
					}
					if uniformLikelihoods(stats) {
						subseqUniformMoves++
					}
					if subseqUniformMoves == *maxUniformMoves {
						break
					}
					gameExamples = append(gameExamples, &pb.MCTSExample{
						Board:     ge.Board().Proto(),
						MoveStats: moveStats,
						GameId:    gameId,
					})
					countCh <- 1
					nMoves++
					if !ge.MakeMove(m) {
						log.Fatalf("Could not make a move")
						return
					}
				}
				if subseqUniformMoves < *maxUniformMoves && !ge.IsDone() {
					// Timeout
					break
				}
				// Update examples with result and write them to the zip archive.
				for _, e := range gameExamples {
					e.Result = []int32{int32(ge.Board().Score[0]), int32(ge.Board().Score[1])}
					data, err := proto.Marshal(e)
					if err != nil {
						log.Fatalf("marshal proto: %v", err)
					}
					f, err := zw.Create(fmt.Sprintf("example-%06d.pb", nExamples))
					if err != nil {
						log.Fatalf("add zip entry: %v", err)
					}
					if _, err := f.Write(data); err != nil {
						log.Fatalf("write zip entry: %v", err)
					}
				}
			}
		}()
	}
	wg.Wait()
	close(countCh)
}
