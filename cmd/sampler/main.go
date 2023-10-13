// The sampler command generates examples that can be used as training data
// for the HexZero neural network.
package main

import (
	"archive/zip"
	"flag"
	"fmt"
	"log"
	"os"
	"path"
	"time"

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
	"google.golang.org/protobuf/proto"
)

func main() {
	runTime := flag.Duration("run-time", 10*time.Second, "Time to run before exiting.")
	thinkTime := flag.Duration("think-time", 50*time.Millisecond, "Think time per move in seconds.")
	outputDir := flag.String("output-dir", ".", "Directory to which examples are written.")
	flag.Parse()
	if len(flag.Args()) > 0 {
		log.Fatalf("Unexpected args: %v", flag.Args())
	}
	examplesFile := path.Join(*outputDir, fmt.Sprintf("examples-%s.zip", time.Now().Format("20060102-150405")))
	zf, err := os.Create(examplesFile)
	if err != nil {
		log.Fatalf("create file: %v", err)
	}
	defer zf.Close()
	zw := zip.NewWriter(zf)
	defer zw.Close()
	started := time.Now()
	nGames := 0
	nExamples := 0
	fmt.Printf("Generating samples for a duration of %v with thinking time %v. Examples will get written to %q.\n",
		*runTime, *thinkTime, examplesFile)
	for time.Since(started) < *runTime {
		ge := hexz.NewGameEngineFlagz()
		mcts := hexz.NewMCTS()
		gameExamples := []*pb.MCTSExample{}
		fmt.Printf("Game %d at %d examples\n", nGames, nExamples)
		for !ge.IsDone() && time.Since(started) < *runTime {
			m, stats := mcts.SuggestMove(ge, *thinkTime)
			if !ge.MakeMove(m) {
				log.Fatalf("Could not make a move")
				return
			}
			moveStats := make([]*pb.MCTSExample_MoveStats, len(stats.Moves))
			for i, mv := range stats.Moves {
				moveStats[i] = &pb.MCTSExample_MoveStats{
					Move:    m.Proto(),
					Visits:  int32(mv.Iterations),
					WinRate: float32(mv.Q),
				}
			}
			gameExamples = append(gameExamples, &pb.MCTSExample{
				Board:     ge.Board().Proto(),
				MoveStats: moveStats,
			})
			nExamples++
		}
		if !ge.IsDone() {
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
		nGames++
	}
	fmt.Printf("Generated %d examples in %s\n", nExamples, examplesFile)
}
