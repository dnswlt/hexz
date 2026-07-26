// nbench2 evaluates hexz ML players against each other.
// It starts remote CPU players with different model checkpoints
// and plays games.
// Results are stored as BenchmarkResult in a JSONlines files.
// (See nbench.proto for the schema.)
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/internal/elo"
	"github.com/dnswlt/hexz/internal/xrand"
	"github.com/dnswlt/hexz/pkg/hexz"
	"google.golang.org/protobuf/proto"
)

type CLIArgs struct {
	CPUServerFile  string
	NumGames       int
	Concurrency    int
	Iterations     int
	StatsFile      string
	ModelRepo      string
	ModelKey1      string
	ModelKey2      string
	ModelKeys      []string
	Device         string
	PlayBothSides  bool
	PositionsFile  string
	WritePositions int

	// Elo related flags

	// Print Elo scores and exit.
	PrintElo bool
}

func createRemotePlayer(playerId api.PlayerId, url string, iterations int) (*hexz.RemoteCPUPlayer, error) {
	remoteCPUClient, err := hexz.NewCPUPlayerServiceClient(url)
	if err != nil {
		return nil, fmt.Errorf("failed to create remote player: %v", err)
	}
	return hexz.NewRemoteCPUPlayer(remoteCPUClient, playerId, 0, iterations), nil
}

func parseArgs() (*CLIArgs, error) {
	args := &CLIArgs{}

	flag.StringVar(&args.CPUServerFile, "cpuserver", "./cpp/build/cpuserver", "cpuserver binary to execute")
	flag.IntVar(&args.NumGames, "games", 1, "Number of games to play")
	flag.IntVar(&args.Concurrency, "concurrency", 0, "Maximum concurrent games (0 means one worker per game)")
	flag.IntVar(&args.Iterations, "iterations", 800, "Maximum MCTS iterations per move for both players")
	flag.StringVar(&args.StatsFile, "stats-file", "./stats/nbench.jsonl", "JSONlines file to which results are append")
	flag.StringVar(&args.ModelRepo, "model-repo", ".", "Base folder of hexz model repository")
	flag.StringVar(&args.ModelKey1, "key1", "", "Model key (e.g. \"res10:58\") of the first model to evaluate")
	flag.StringVar(&args.ModelKey2, "key2", "", "Model key (e.g. \"res10:23\") of the second model to evaluate")
	modelKeys := flag.String("keys", "", "Model keys (e.g. \"res10:23,res10:24,res10:25\") ")
	flag.StringVar(&args.Device, "device", "cuda", "PyTorch device type (cuda, cpu, mps)")
	flag.BoolVar(&args.PlayBothSides, "both-sides", false,
		"Whether to play both as P1 and P2. If true, twice as many games are played as specified in --games.")
	flag.StringVar(&args.PositionsFile, "positions-file", "",
		"JSONL corpus of initial positions. Reverse-seat games reuse the same positions.")
	flag.IntVar(&args.WritePositions, "write-positions", 0,
		"Generate this many initial positions in --positions-file and exit; refuses to overwrite.")
	flag.BoolVar(&args.PrintElo, "print-elo", false,
		"If true, print Elo scores resulting from all results in --stats-file, and exit.")

	flag.Parse()
	if *modelKeys != "" {
		args.ModelKeys = strings.Split(*modelKeys, ",")
	}
	if len(flag.Args()) > 0 {
		return nil, fmt.Errorf("unexpected extra args: %v", flag.Args())
	}
	return args, nil
}

func splitKey(key string) (string, int, error) {
	parts := strings.Split(key, ":")
	if len(parts) != 2 {
		return "", 0, fmt.Errorf("invalid model key format %q, expected \"name:checkpoint\"", key)
	}

	name := parts[0]
	number, err := strconv.Atoi(parts[1])
	if err != nil {
		return "", 0, fmt.Errorf("invalid integer value: %v", err)
	}
	return name, number, nil
}

func modelPath(base, modelKey string) (string, error) {
	name, key, err := splitKey(modelKey)
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "models", "flagz", name, "checkpoints", strconv.Itoa(key), "scriptmodule.pt"), nil
}

func launchCPUPlayer(ctx context.Context, args *CLIArgs, modelKey string, addr string) (*exec.Cmd, error) {
	mp, err := modelPath(args.ModelRepo, modelKey)
	if err != nil {
		return nil, err
	}
	_, err = os.Stat(mp)
	if err != nil {
		return nil, fmt.Errorf("model file for %s does not exist or is not accessible: %v", modelKey, err)
	}
	cmdArgs := []string{
		"--device=" + args.Device,
		"--max_think_time_ms=0",
		"--model_path=" + mp,
		"--model_key=" + modelKey,
		"--server_addr=" + addr,
	}
	log.Printf("Starting %s %s", args.CPUServerFile, strings.Join(cmdArgs, " "))
	cmd := exec.CommandContext(ctx, args.CPUServerFile, cmdArgs...)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start cpuserver: %v", err)
	}
	return cmd, nil
}

func waitForCompletion(cmd *exec.Cmd, ch chan bool) {
	if err := cmd.Wait(); err != nil {
		if sysErr, ok := err.(*exec.ExitError); ok {
			if sysErr.ExitCode() != -1 {
				// We expect exit code -1, which indicates the subprocess was killed by a (our) signal.
				log.Printf("Unexpected exit code while waiting for cpuserver[1]: %v", err)
			} else {
				log.Printf("Terminated subprocess %s %s: %v", cmd.Path, strings.Join(cmd.Args, " "), err)
			}
		} else {
			log.Printf("Error while waiting for cpuserver[1]: %v", err)
		}
	} else {
		log.Printf("Subprocess %s %s is done", cmd.Path, strings.Join(cmd.Args, " "))
	}
	ch <- true
}

func terminateSubprocesses(doneCh chan bool, cancel context.CancelFunc, cmds ...*exec.Cmd) {
	// Try to kill the subprocesses gracefully.
	for _, cmd := range cmds {
		if cmd.ProcessState != nil {
			log.Printf("Process has already terminated (exit code: %v)", cmd.ProcessState.ExitCode())
			continue
		}
		if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
			log.Printf("Sending SIGTERM failed: %v", err)
		}
	}

	timer := time.NewTimer(2 * time.Second)
	for i := 0; i < len(cmds); i++ {
		select {
		case <-doneCh:
		case <-timer.C:
			// Time's up, send KILL signal.
			log.Printf("Sending KILL signal to child processes (via cancel)")
			cancel()
			return
		}
	}
}

func playConcurrent(
	ctx context.Context,
	p1, p2 *hexz.RemoteCPUPlayer,
	positions []StartingPosition,
	concurrency int,
	statsFile string,
	positionSet string,
) (*ResultStats, error) {
	// This is ML model evaluation mode. Play concurrently to benefit from concurrent requests to the GPU.
	nb, err := NewConcurrentNBench(p1, p2, positions, concurrency, statsFile, positionSet)
	if err != nil {
		return nil, fmt.Errorf("failed to create ConcurrentNBench: %v", err)
	}
	if err := nb.Play(ctx); err != nil {
		return nil, err
	}
	return nb.stats, nil
}

func printEloRatings(statsFile string) {
	results, err := elo.ReadResults(statsFile)
	if err != nil {
		log.Fatalf("Failed to read results from %q: %v", statsFile, err)
	}
	eloRatings := elo.Ratings(results)
	fmt.Printf("Elo Ratings from %d records in %s:\n", len(results), statsFile)
	for i, e := range eloRatings {
		fmt.Printf("#%d %s:%d %.1f (%d/%d/%d)\n", i+1, e.Key.Name, e.Key.Checkpoint, e.Rating, e.Wins, e.Draws, e.Games-e.Wins-e.Draws)
	}
	fmt.Println()
	btRatings, err := elo.BradleyTerry(results)
	if err != nil {
		fmt.Printf("Bradley-Terry did not converge: %v\n", err)
		return
	}
	fmt.Printf("Bradley-Terry ratings from %d records in %s:\n", len(results), statsFile)
	for i, e := range btRatings {
		fmt.Printf("#%d %s:%d %.1f (%d/%d/%d)\n", i+1, e.Key.Name, e.Key.Checkpoint, e.Rating, e.Wins, e.Draws, e.Games-e.Wins-e.Draws)
	}
}

func countsFromStatsFile(statsFile string) (map[string]int, error) {
	// Choose model keys randomly, proportional to the inverse number of times they were used previously.
	counts := make(map[string]int)
	// Increase weights based on number of games the model was used in.
	results, err := elo.ReadResults(statsFile)
	if err != nil {
		return nil, fmt.Errorf("invalid stats file: %v", err)
	}
	for _, r := range results {
		k1 := r.P1Result.ModelKey
		k2 := r.P2Result.ModelKey
		counts[fmt.Sprintf("%s:%d", k1.Name, k1.Checkpoint)] += int(r.Games)
		if !proto.Equal(k1, k2) {
			counts[fmt.Sprintf("%s:%d", k2.Name, k2.Checkpoint)] += int(r.Games)
		}
	}
	return counts, nil
}

func chooseModels(args *CLIArgs) (string, string, error) {
	if args.ModelKey1 != "" {
		// Explicitly selected models.
		return args.ModelKey1, args.ModelKey2, nil
	}
	if len(args.ModelKeys) == 0 {
		return "", "", fmt.Errorf("no models to choose from")
	}
	if len(args.ModelKeys) == 1 {
		if args.ModelKeys[0] != "any" {
			return args.ModelKeys[0], args.ModelKeys[0], nil
		}
		if args.StatsFile == "" {
			return "", "", fmt.Errorf("must specify --stats-file when using --keys=any")
		}
	}
	counts := make(map[string]int)
	if args.StatsFile != "" {
		var err error
		counts, err = countsFromStatsFile(args.StatsFile)
		if err != nil {
			return "", "", err
		}
		if args.ModelKeys[0] == "any" {
			args.ModelKeys = make([]string, 0, len(counts))
			for k := range counts {
				args.ModelKeys = append(args.ModelKeys, k)
			}
		}
	}
	weights := make([]float64, len(args.ModelKeys))
	for i, m := range args.ModelKeys {
		weights[i] = 1 / (1 + float64(counts[m]))
	}
	ks := xrand.SampleWeighted(args.ModelKeys, weights, 2)
	return ks[0], ks[1], nil
}

func main() {
	args, err := parseArgs()
	if err != nil {
		log.Printf("Error parsing command line arguments: %v", err)
		return
	}

	if args.WritePositions > 0 {
		if args.PositionsFile == "" {
			log.Fatal("--positions-file is required with --write-positions")
		}
		positions, err := generateStartingPositions(args.WritePositions)
		if err != nil {
			log.Fatalf("Could not generate positions: %v", err)
		}
		if err := writeStartingPositions(args.PositionsFile, positions); err != nil {
			log.Fatalf("Could not write positions: %v", err)
		}
		log.Printf("Wrote %d positions to %s", len(positions), args.PositionsFile)
		return
	}

	if args.PrintElo {
		if args.StatsFile == "" {
			log.Fatalf("Must specifiy --stats-file for --print-elo")
		}
		printEloRatings(args.StatsFile)
		return
	}

	var positions []StartingPosition
	if args.PositionsFile != "" {
		positions, err = readStartingPositions(args.PositionsFile)
		if err != nil {
			log.Fatalf("Could not read positions: %v", err)
		}
		if args.NumGames > len(positions) {
			log.Fatalf("--games=%d exceeds the %d positions in %s", args.NumGames, len(positions), args.PositionsFile)
		}
		if args.NumGames > 0 {
			positions = positions[:args.NumGames]
		}
	} else {
		if args.NumGames <= 0 {
			log.Fatal("--games must be positive without --positions-file")
		}
		positions, err = generateStartingPositions(args.NumGames)
		if err != nil {
			log.Fatalf("Could not generate in-memory positions: %v", err)
		}
	}
	log.Printf("Using %d initial positions", len(positions))

	modelKey1, modelKey2, err := chooseModels(args)
	if err != nil {
		log.Fatalf("Could not choose model keys: %v", err)
	}
	log.Printf("Using model keys %v - %v", modelKey1, modelKey2)

	ctx, cancel := context.WithCancel(context.Background())
	defer func() {
		cancel()
	}()

	// Terminate subprocesses on SIGINT
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	go func() {
		<-c
		log.Printf("Received interrupt, canceling context...")
		cancel()
	}()

	subDoneCh := make(chan bool, 2)

	p1URL := "localhost:50171"
	cmd1, err := launchCPUPlayer(ctx, args, modelKey1, p1URL)
	if err != nil {
		log.Printf("Failed to start cpuserver[1]: %v", err)
		return
	}
	go waitForCompletion(cmd1, subDoneCh)

	p2URL := "localhost:50172"
	cmd2, err := launchCPUPlayer(ctx, args, modelKey2, p2URL)
	if err != nil {
		log.Printf("Failed to start cpuserver[2]: %v", err)
		return
	}
	go waitForCompletion(cmd2, subDoneCh)

	defer terminateSubprocesses(subDoneCh, cancel, cmd1, cmd2)

	log.Printf("Waiting for servers to come alive:")
	for i := 4; i > 0; i-- {
		log.Printf("%d ...", i)
		time.Sleep(1 * time.Second)
	}
	log.Printf("Let's go!")

	p1, err := createRemotePlayer(api.PlayerId("P1"), p1URL, args.Iterations)
	if err != nil {
		log.Printf("Failed to create remote P1: %v", err)
		return
	}
	p2, err := createRemotePlayer(api.PlayerId("P2"), p2URL, args.Iterations)
	if err != nil {
		log.Printf("Failed to create remote P2: %v", err)
		return
	}

	first, err := playConcurrent(
		ctx, p1, p2, positions, args.Concurrency, args.StatsFile, args.PositionsFile,
	)
	if err != nil {
		log.Printf("Play failed: %v", err)
		return
	}
	if args.PlayBothSides {
		reverse, err := playConcurrent(
			ctx, p2, p1, positions, args.Concurrency, args.StatsFile, args.PositionsFile,
		)
		if err != nil {
			log.Printf("Reverse play failed: %v", err)
			return
		}
		if err := logPairedSummary(modelKey1, modelKey2, first, reverse); err != nil {
			log.Printf("Cannot summarize paired results: %v", err)
			return
		}
	}
}
