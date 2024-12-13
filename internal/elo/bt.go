package elo

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	npb "github.com/dnswlt/hexz/pkg/nbenchpb"
)

func BradleyTerry(results []*npb.BenchmarkResult) ([]*Rating, error) {
	const tolerance = 1e-5

	ratings := initialRatings(results, 1.)

	// winMat[i][j] stores the wins of player i against player j.
	// We add a fictitious player against which each player wins
	// once, and loses once. This is done to avoid numerical instability
	// in the presence of players that always win, or always lose.
	// It even seems to be a theoretically well-founded approach, see Sec. 5 of
	// https://jmlr.org/papers/volume24/22-1086/22-1086.pdf
	l := len(ratings)
	winMat := make([][]float64, l+1)
	for i := 0; i < l; i++ {
		winMat[i] = make([]float64, l+1)
		winMat[i][l] = 1
	}
	winMat[l] = make([]float64, l+1)
	for j := 0; j < l; j++ {
		winMat[l][j] = 1
	}

	// Map model key to (lexicographically ordered) index in winMat
	pMap := make(map[string]int)
	{
		var keys []string
		for k := range ratings {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		j := 0
		for _, k := range keys {
			pMap[k] = j
			j++
		}
	}
	for _, r := range results {
		i1 := pMap[mkey(r.P1Result.ModelKey)]
		i2 := pMap[mkey(r.P2Result.ModelKey)]
		// TODO: draws?
		winMat[i1][i2] += float64(r.P1Result.Wins)
		winMat[i2][i1] += float64(r.P2Result.Wins)
	}
	// Set initial ratings to 1.0
	ps := make([]float64, len(winMat))
	for i := 0; i < len(ps); i++ {
		ps[i] = 1.0
	}
	// Allow at least 2x as many iterations as there are players.
	maxIter := max(2*l, 10)
	converged := false
	var maxDiff float64
	for iter := 0; iter < maxIter; iter++ {
		maxDiff = 0.0
		for i := 0; i < len(winMat); i++ {
			numer := 0.0
			denom := 0.0
			for j := 0; j < len(winMat); j++ {
				if i == j {
					continue
				}
				numer += winMat[i][j] * ps[j] / (ps[i] + ps[j])
				denom += winMat[j][i] / (ps[i] + ps[j])
			}
			pNext := numer / denom
			maxDiff = max(maxDiff, math.Abs(ps[i]-pNext))
			ps[i] = pNext
		}
		// Normalize by geometric mean of the strenght of all players.
		log_sum := 0.0
		for _, p := range ps {
			log_sum += math.Log(p)
		}
		geo_mean := math.Exp(log_sum / float64(len(ps)))
		for i := 0; i < len(ps); i++ {
			ps[i] /= geo_mean
		}
		if maxDiff < tolerance {
			converged = true
			break
		}
	}
	if !converged {
		return nil, fmt.Errorf("Bradley-Terry did not converge after %d iterations (maxDiff > tolerance: %.6f > %.6f)", maxIter, maxDiff, tolerance)
	}
	var res []*Rating
	for _, r := range ratings {
		idx := pMap[mkey(r.Key)]
		r.Rating = ps[idx]
		res = append(res, r)
	}
	// Sort by rating in descending order (best player first)
	slices.SortFunc(res, func(a, b *Rating) int {
		return cmp.Compare(b.Rating, a.Rating)
	})
	return res, nil
}
