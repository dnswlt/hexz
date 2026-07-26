package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sync"
	"time"

	"github.com/dnswlt/hexz/internal/elo"
	"github.com/dnswlt/hexz/pkg/hexz"
	"github.com/dnswlt/hexz/pkg/hexzpb"
	npb "github.com/dnswlt/hexz/pkg/nbenchpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

type ResultStats struct {
	games       int
	wins        [2]int
	gameResults []*npb.BenchmarkResult_GameResult
	mut         sync.Mutex
}

func (r *ResultStats) Games() int {
	r.mut.Lock()
	defer r.mut.Unlock()
	return r.games
}

func (r *ResultStats) Wins() [2]int {
	r.mut.Lock()
	defer r.mut.Unlock()
	return r.wins
}

// winner == 0 means draw.
func (r *ResultStats) Add(result *npb.BenchmarkResult_GameResult) {
	r.mut.Lock()
	defer r.mut.Unlock()
	r.games++
	winner := int(result.Winner)
	if winner > 0 {
		r.wins[winner-1]++
	}
	r.gameResults = append(r.gameResults, result)
}
func (r *ResultStats) String() string {
	r.mut.Lock()
	defer r.mut.Unlock()
	return fmt.Sprintf("ResultStats(games=%d wins=%v)", r.games, r.wins)
}

type ConcurrentNBench struct {
	p1 *hexz.RemoteCPUPlayer
	p2 *hexz.RemoteCPUPlayer
	// These are populated when the game is started
	p1Key       *hexzpb.ModelKey
	p2Key       *hexzpb.ModelKey
	positions   []StartingPosition
	concurrency int
	stats       *ResultStats
	statsFile   string
	positionSet string
}

func NewConcurrentNBench(
	p1, p2 *hexz.RemoteCPUPlayer,
	positions []StartingPosition,
	concurrency int,
	statsFile string,
	positionSet string,
) (*ConcurrentNBench, error) {
	// Get model keys for logging
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	p1Key, err := p1.ModelKey(ctx)
	if err != nil {
		return nil, fmt.Errorf("cannot get model key for P1: %v", err)
	}
	p2Key, err := p2.ModelKey(ctx)
	if err != nil {
		return nil, fmt.Errorf("cannot get model key for P2: %v", err)
	}

	if len(positions) == 0 {
		return nil, fmt.Errorf("no starting positions")
	}
	if concurrency <= 0 || concurrency > len(positions) {
		concurrency = len(positions)
	}
	return &ConcurrentNBench{
		p1:          p1,
		p2:          p2,
		p1Key:       p1Key,
		p2Key:       p2Key,
		positions:   positions,
		concurrency: concurrency,
		stats:       &ResultStats{},
		statsFile:   statsFile,
		positionSet: positionSet,
	}, nil
}

func (nb *ConcurrentNBench) appendStats(started, done time.Time) error {
	res := &npb.BenchmarkResult{
		Started:         tpb.New(started),
		DurationSeconds: done.Sub(started).Seconds(),
		Games:           int32(nb.stats.games),
		P1Result: &npb.BenchmarkResult_PlayerResult{
			ModelKey:   nb.p1Key,
			Wins:       int32(nb.stats.wins[0]),
			Iterations: int32(nb.p1.MaxIterations()),
		},
		P2Result: &npb.BenchmarkResult_PlayerResult{
			ModelKey:   nb.p2Key,
			Wins:       int32(nb.stats.wins[1]),
			Iterations: int32(nb.p2.MaxIterations()),
		},
		Args:        os.Args[1:],
		GameResults: nb.stats.gameResults,
		PositionSet: nb.positionSet,
	}
	return elo.AppendStats(nb.statsFile, res)
}

func (nb *ConcurrentNBench) P1Name() string {
	return fmt.Sprintf("P1(%s:%d)", nb.p1Key.Name, nb.p1Key.Checkpoint)
}
func (nb *ConcurrentNBench) P2Name() string {
	return fmt.Sprintf("P2(%s:%d)", nb.p2Key.Name, nb.p2Key.Checkpoint)
}

func (nb *ConcurrentNBench) Play(ctx context.Context) error {
	log.Printf("Playing %s vs. %s on %d positions with concurrency %d\n",
		nb.P1Name(), nb.P2Name(), len(nb.positions), nb.concurrency)

	var wg sync.WaitGroup

	started := time.Now()
	workerCtx, workerCancel := context.WithCancel(ctx)
	defer workerCancel()
	jobs := make(chan int)
	for worker := 0; worker < nb.concurrency; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := range jobs {
				position := nb.positions[i]
				ge := hexz.NewGameEngineFlagz()
				if err := ge.FromProto(position.State); err != nil {
					log.Printf("Worker %d: invalid position %s: %v", worker, position.ID, err)
					workerCancel()
					return
				}
				gameStarted := time.Now()
				for !ge.IsDone() {
					p := nb.p1
					if ge.Board().Turn == 2 {
						p = nb.p2
					}
					mv, _, err := p.SuggestMove(workerCtx, ge)
					if err != nil {
						log.Printf("Worker %d position %s: RPC failed: %v", worker, position.ID, err)
						workerCancel()
						return
					}
					if err := ge.MakeMoveError(*mv); err != nil {
						log.Printf("Worker %d position %s: failed move: %v", worker, position.ID, err)
						workerCancel()
						return
					}
				}
				score := ge.Board().Score
				result := &npb.BenchmarkResult_GameResult{
					PositionId:      position.ID,
					Winner:          int32(ge.Winner()),
					P1Score:         int32(score[0]),
					P2Score:         int32(score[1]),
					Moves:           int32(ge.Board().Move),
					DurationSeconds: time.Since(gameStarted).Seconds(),
				}
				log.Printf("[worker %d] Position %s over after %d moves: winner=%d score=%v",
					worker, position.ID, ge.Board().Move, ge.Winner(), score)
				nb.stats.Add(result)
			}
		}(worker)
	}
sendJobs:
	for i := range nb.positions {
		select {
		case jobs <- i:
		case <-workerCtx.Done():
			break sendJobs
		}
	}
	close(jobs)

	wg.Wait()
	done := time.Now()
	if nb.stats.Games() != len(nb.positions) {
		return fmt.Errorf("benchmark was aborted; intermediate result: %v", nb.stats)
	}
	var result string
	if nb.stats.wins[0] > nb.stats.wins[1] {
		result = "P1 wins"
	} else if nb.stats.wins[0] == nb.stats.wins[1] {
		result = "draw"
	} else {
		result = "P2 wins"
	}

	log.Printf("Final result: best of %d: %s vs %s: %s %d-%d\n",
		len(nb.positions), nb.P1Name(), nb.P2Name(), result, nb.stats.wins[0], nb.stats.wins[1])
	if nb.statsFile != "" {
		err := nb.appendStats(started, done)
		if err != nil {
			return fmt.Errorf("failed to append results to %q: %v", nb.statsFile, err)
		}
	}
	return nil
}

type PairedSummary struct {
	Positions  int
	Games      int
	Model1Wins int
	Model2Wins int
	Draws      int
	// Model1Score counts a win as 1, a draw as 0.5, and a loss as 0.
	Model1Score float64
	// The confidence interval treats each starting position, rather than each
	// game, as one independent observation. The two seat-swapped games from a
	// position are deliberately kept in the same observation.
	ScoreLo float64
	ScoreHi float64
}

func model1GameScore(result *npb.BenchmarkResult_GameResult, model1IsP1 bool) float64 {
	if result.Winner == 0 {
		return 0.5
	}
	if (result.Winner == 1) == model1IsP1 {
		return 1
	}
	return 0
}

func pairedSummary(first, reverse *ResultStats) (PairedSummary, error) {
	firstByPosition := make(map[string]*npb.BenchmarkResult_GameResult, len(first.gameResults))
	for _, result := range first.gameResults {
		if _, exists := firstByPosition[result.PositionId]; exists {
			return PairedSummary{}, fmt.Errorf("duplicate first-game position %q", result.PositionId)
		}
		firstByPosition[result.PositionId] = result
	}
	reverseByPosition := make(map[string]*npb.BenchmarkResult_GameResult, len(reverse.gameResults))
	for _, result := range reverse.gameResults {
		if _, exists := reverseByPosition[result.PositionId]; exists {
			return PairedSummary{}, fmt.Errorf("duplicate reverse-game position %q", result.PositionId)
		}
		reverseByPosition[result.PositionId] = result
	}
	if len(firstByPosition) != len(reverseByPosition) {
		return PairedSummary{}, fmt.Errorf(
			"position count mismatch: first=%d reverse=%d",
			len(firstByPosition), len(reverseByPosition),
		)
	}

	pairScores := make([]float64, 0, len(firstByPosition))
	for positionID, firstResult := range firstByPosition {
		reverseResult, ok := reverseByPosition[positionID]
		if !ok {
			return PairedSummary{}, fmt.Errorf("position %q missing from reverse games", positionID)
		}
		pairScores = append(pairScores,
			(model1GameScore(firstResult, true)+model1GameScore(reverseResult, false))/2,
		)
	}
	if len(pairScores) == 0 {
		return PairedSummary{}, fmt.Errorf("cannot summarize zero paired positions")
	}

	model1Wins := first.wins[0] + reverse.wins[1]
	model2Wins := first.wins[1] + reverse.wins[0]
	games := first.games + reverse.games
	draws := games - model1Wins - model2Wins
	var score float64
	for _, pairScore := range pairScores {
		score += pairScore
	}
	score /= float64(len(pairScores))

	lo, hi := score, score
	if len(pairScores) > 1 {
		var sumSquaredDeviations float64
		for _, pairScore := range pairScores {
			d := pairScore - score
			sumSquaredDeviations += d * d
		}
		sampleVariance := sumSquaredDeviations / float64(len(pairScores)-1)
		const z95 = 1.959963984540054
		margin := z95 * math.Sqrt(sampleVariance/float64(len(pairScores)))
		lo = math.Max(0, score-margin)
		hi = math.Min(1, score+margin)
	}
	return PairedSummary{
		Positions:   len(pairScores),
		Games:       games,
		Model1Wins:  model1Wins,
		Model2Wins:  model2Wins,
		Draws:       draws,
		Model1Score: score,
		ScoreLo:     lo,
		ScoreHi:     hi,
	}, nil
}

func logPairedSummary(model1, model2 string, first, reverse *ResultStats) error {
	summary, err := pairedSummary(first, reverse)
	if err != nil {
		return err
	}
	log.Printf(
		"Paired summary over %d positions (%d games): %s=%d, %s=%d, draws=%d; "+
			"%s score %.1f%% (paired-position 95%% CI %.1f%%..%.1f%%)",
		summary.Positions, summary.Games,
		model1, summary.Model1Wins, model2, summary.Model2Wins, summary.Draws,
		model1, 100*summary.Model1Score, 100*summary.ScoreLo, 100*summary.ScoreHi,
	)
	return nil
}
