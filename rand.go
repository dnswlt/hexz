package hexz

import (
	"hash/maphash"
	"math"
	"math/bits"
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

// randFloat64 returns a uniformly distributed random number in the interval [0.0, 1.0).
func randFloat64() float64 {
	return float64(rand64()&int53Mask) * f53Mul
}

// randIntn returns a uniformly distributed random number in the interval [0, n).
// n must be a positive int32.
func randIntn(n int) int {
	if n <= 0 || n > math.MaxInt32 {
		panic("randIntn: invalid argument")
	}
	r, _ := bits.Mul64(uint64(n), rand64())
	return int(r)
}
