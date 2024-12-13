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

type Rating struct {
	Key    *hexzpb.ModelKey
	Games  int
	Wins   int
	Draws  int
	Rating float64
}

func mkey(key *hexzpb.ModelKey) string {
	return fmt.Sprintf("%s:%d", key.Name, key.Checkpoint)
}

const (
	scalingFactor float64 = 400
)

func expected(eloPlayer, eloOpponent float64) float64 {
	return 1 / (1 + math.Pow(10, (eloOpponent-eloPlayer)/scalingFactor))
}

// initialRatings returns initial ratings for all models found in results.
func initialRatings(results []*npb.BenchmarkResult, initialRating float64) map[string]*Rating {
	ratings := make(map[string]*Rating)
	for _, r := range results {
		m1 := mkey(r.P1Result.ModelKey)
		if ratings[m1] == nil {
			ratings[m1] = &Rating{
				Key:    r.P1Result.ModelKey,
				Rating: initialRating,
			}
		}
		m2 := mkey(r.P2Result.ModelKey)
		if ratings[m2] == nil {
			ratings[m2] = &Rating{
				Key:    r.P2Result.ModelKey,
				Rating: initialRating,
			}
		}
		// Update game stats
		draws := int(r.Games - (r.P1Result.Wins + r.P2Result.Wins))
		ratings[m1].Games += int(r.Games)
		ratings[m2].Games += int(r.Games)
		ratings[m1].Wins += int(r.P1Result.Wins)
		ratings[m2].Wins += int(r.P2Result.Wins)
		ratings[m1].Draws += draws
		ratings[m2].Draws += draws
	}
	return ratings
}

func Ratings(results []*npb.BenchmarkResult) []*Rating {
	var initialRating float64 = 1500
	var k float64 = 32
	ratings := initialRatings(results, initialRating)

	for _, r := range results {
		m1 := mkey(r.P1Result.ModelKey)
		m2 := mkey(r.P2Result.ModelKey)
		// Expected scores
		expectedM1 := float64(r.Games) * expected(ratings[m1].Rating, ratings[m2].Rating)
		expectedM2 := float64(r.Games) - expectedM1
		// Actual scores
		draws := float64(r.Games - (r.P1Result.Wins + r.P2Result.Wins))
		scoreM1 := float64(r.P1Result.Wins) + draws/2
		scoreM2 := float64(r.P2Result.Wins) + draws/2

		// fmt.Printf("Match %d: %s - %s: %.1f/%.0f/%.1f\n", i, m1, m2, scoreM1, draws, scoreM2)
		// Update Elo scores
		// fmt.Printf(" Elos before: %.1f - %.1f\n", elos[m1].Rating, elos[m2].Rating)
		ratings[m1].Rating += k * (scoreM1 - expectedM1)
		ratings[m2].Rating += k * (scoreM2 - expectedM2)
		// fmt.Printf(" Elos after: %.1f - %.1f\n", elos[m1].Rating, elos[m2].Rating)
	}

	var result []*Rating
	for _, e := range ratings {
		result = append(result, e)
	}
	slices.SortFunc(result, func(a, b *Rating) int {
		if c := cmp.Compare(b.Rating, a.Rating); c != 0 {
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
