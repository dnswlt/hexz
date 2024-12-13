package xrand

import (
	"fmt"
	"hash/maphash"
	"math"
	"math/bits"
	"math/rand/v2"
	"sort"
)

// Random numbers without the rand.Source and rand.Rand hassle.
// The functions in this file are safe to use from multiple goroutines.
// Found here: https://github.com/golang/go/issues/49892.

const (
	int53Mask = 1<<53 - 1
	f53Mul    = 0x1.0p-53
)

// rand64 returns a uniformly distributed random uint64, fast.
func rand64() uint64 {
	return maphash.Bytes(maphash.MakeSeed(), nil)
}

// Float64 returns a uniformly distributed random number in the interval [0.0, 1.0).
func Float64() float64 {
	return float64(rand64()&int53Mask) * f53Mul
}

// Intn returns a uniformly distributed random number in the interval [0, n).
// n must be a positive int32.
func Intn(n int) int {
	if n <= 0 || n > math.MaxInt32 {
		panic("randIntn: invalid argument")
	}
	r, _ := bits.Mul64(uint64(n), rand64())
	return int(r)
}

// SampleWeighted samples n items from items without replacement.
// Panics if n < 0 or n > len(items).
func SampleWeighted[T any](items []T, weights []float64, n int) []T {
	if len(items) != len(weights) {
		panic("SampleWeighted: items and weights must have the same length")
	}
	if n < 0 || n > len(items) {
		panic(fmt.Sprintf("SampleWeighted: n > len(items): %d > %d", n, len(items)))
	}
	if n == 0 {
		return nil
	}

	// Create a copy of items and weights to avoid modifying the input
	remainingItems := make([]T, len(items))
	copy(remainingItems, items)
	remainingWeights := make([]float64, len(weights))
	copy(remainingWeights, weights)

	sampled := make([]T, 0, n)
	for i := 0; i < n; i++ {
		// Compute prefix sums for weights
		prefixSums := make([]float64, len(remainingWeights))
		prefixSums[0] = remainingWeights[0]
		for j := 1; j < len(remainingWeights); j++ {
			prefixSums[j] = prefixSums[j-1] + remainingWeights[j]
		}

		// Draw a random number in [0, totalWeight)
		r := rand.Float64() * prefixSums[len(prefixSums)-1]

		// Use binary search to find the index
		index := sort.Search(len(prefixSums), func(j int) bool {
			return prefixSums[j] >= r
		})

		sampled = append(sampled, remainingItems[index])

		// Remove the selected item by swapping it with the last item and slicing off the end
		remainingItems[index] = remainingItems[len(remainingItems)-1]
		remainingWeights[index] = remainingWeights[len(remainingWeights)-1]
		remainingItems = remainingItems[:len(remainingItems)-1]
		remainingWeights = remainingWeights[:len(remainingWeights)-1]
	}

	return sampled
}
