package elo

import (
	"bufio"
	"cmp"
	"fmt"
	"math"
	"os"
	"path"
	"slices"
	"strings"

	"github.com/dnswlt/hexz/pkg/hexzpb"
	npb "github.com/dnswlt/hexz/pkg/nbenchpb"
	"google.golang.org/protobuf/encoding/protojson"
)

type EloScore struct {
	Key   *hexzpb.ModelKey
	Games int
	Wins  int
	Draws int
	Score float64
}

func mkey(key *hexzpb.ModelKey) string {
	return fmt.Sprintf("%s:%d", key.Name, key.Checkpoint)
}

func Scores(results []*npb.BenchmarkResult) []*EloScore {
	var initialScore float64 = 1500
	var scalingFactor float64 = 400
	var k float64 = 32

	var models []*hexzpb.ModelKey
	seenModels := make(map[string]bool)
	for _, r := range results {
		m1 := mkey(r.P1Result.ModelKey)
		if !seenModels[m1] {
			seenModels[m1] = true
			models = append(models, r.P1Result.ModelKey)
		}
		m2 := mkey(r.P2Result.ModelKey)
		if !seenModels[m2] {
			seenModels[m2] = true
			models = append(models, r.P2Result.ModelKey)
		}
	}

	elos := make(map[string]*EloScore)
	for _, m := range models {
		elos[mkey(m)] = &EloScore{
			Key:   m,
			Score: initialScore,
		}
	}

	for _, r := range results {
		m1 := mkey(r.P1Result.ModelKey)
		m2 := mkey(r.P2Result.ModelKey)
		// Expected scores
		expectedM1 := float64(r.Games) / (1 + math.Pow(10, (elos[m2].Score-elos[m1].Score)/scalingFactor))
		expectedM2 := float64(r.Games) - expectedM1
		// Actual scores
		draws := float64(r.Games - (r.P1Result.Wins + r.P2Result.Wins))
		scoreM1 := float64(r.P1Result.Wins) + draws/2
		scoreM2 := float64(r.P2Result.Wins) + draws/2
		// Update Elo scores
		elos[m1].Score += k * (scoreM1 - expectedM1)
		elos[m2].Score += k * (scoreM2 - expectedM2)
		// Update other stats
		elos[m1].Games += int(r.Games)
		elos[m2].Games += int(r.Games)
		elos[m1].Wins += int(r.P1Result.Wins)
		elos[m2].Wins += int(r.P2Result.Wins)
		elos[m1].Draws += int(draws)
		elos[m2].Draws += int(draws)
	}

	var result []*EloScore
	for _, e := range elos {
		result = append(result, e)
	}
	slices.SortFunc(result, func(a, b *EloScore) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		if c := cmp.Compare(b.Wins, a.Wins); c != 0 {
			return c
		}
		if c := cmp.Compare(b.Draws, a.Draws); c != 0 {
			return c
		}
		// As a tie-breaker, compare the age of the model.
		return cmp.Compare(b.Key.Checkpoint, a.Key.Checkpoint)
	})
	return result
}

func ReadResults(statsFile string) ([]*npb.BenchmarkResult, error) {
	f, err := os.Open(statsFile)
	if err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(f)
	lineNum := 0
	var results []*npb.BenchmarkResult
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		// Skip emtpy lines and comment lines starting with '#'.
		if line == "" || line[0] == '#' {
			continue
		}
		r := &npb.BenchmarkResult{}
		if err := protojson.Unmarshal([]byte(line), r); err != nil {
			return nil, fmt.Errorf("invalid JSON on line %d: %v", lineNum, err)
		}
		results = append(results, r)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}
	return results, nil
}

// AppendStats appends result to the JSONlines statsFile.
func AppendStats(statsFile string, result *npb.BenchmarkResult) error {
	if err := os.MkdirAll(path.Base(statsFile), 0755); err != nil {
		return fmt.Errorf("cannot create directory for logfile: %v", err)
	}
	f, err := os.OpenFile(statsFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	m := protojson.MarshalOptions{
		Multiline: false,
	}
	data, err := m.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal BenchmarkResult: %v", err)
	}
	data = append(data, '\n')
	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("failed to append BenchmarkResult: %v", err)
	}
	return nil
}
