package hexz

import (
	"flag"
	"math"
	"testing"
)

// Quick check that sqrt and log are just as fast on float32 as they are on
// float64, despite the casting nuisances.

var useFloat32 = flag.Bool("use-float32", false, "set to true to benchmark with float32")

func BenchmarkSqrt(b *testing.B) {
	const iterations = 100000
	if *useFloat32 {
		for i := 0; i < b.N; i++ {
			var s float32
			for x := float32(1); x < iterations; x += 1 {
				s += float32(math.Sqrt(float64(x)))
			}
			if s < 0 {
				b.Errorf("Wrong sum: %f", s)
			}
		}
	} else {
		for i := 0; i < b.N; i++ {
			var s float64
			for x := float64(1); x < iterations; x += 1 {
				s += math.Sqrt(x)
			}
			if s < 0 {
				b.Errorf("Wrong sum: %f", s)
			}
		}
	}
}

/*
Log is slightly slower for float32 on arm64 (M1 Pro):

go test -bench "^BenchmarkLog$" -run ^$ -count=10  -use-float32 | tee log32.txt
go test -bench "^BenchmarkLog$" -run ^$ -count=10  | tee log64.txt
benchstat log32.txt log64.txt
goos: darwin
goarch: arm64
pkg: github.com/dnswlt/hackz/hexz
       │  log32.txt  │             log64.txt              │
       │   sec/op    │   sec/op     vs base               │
Log-10   527.4µ ± 0%   509.3µ ± 0%  -3.44% (p=0.000 n=10)

*/

func BenchmarkLog(b *testing.B) {
	const iterations = 100000
	if *useFloat32 {
		for i := 0; i < b.N; i++ {
			var s float32
			for x := float32(1); x < iterations; x += 1 {
				s += float32(math.Log(float64(x)))
			}
			if s < 0 {
				b.Errorf("Wrong sum: %f", s)
			}
		}
	} else {
		for i := 0; i < b.N; i++ {
			var s float64
			for x := float64(1); x < iterations; x += 1 {
				s += math.Log(x)
			}
			if s < 0 {
				b.Errorf("Wrong sum: %f", s)
			}
		}
	}
}
