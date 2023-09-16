package hexz

import "testing"

func TestRand64(t *testing.T) {
	// all bits in a rand64 should be uniformly distributed and independent.
	// So if we get 100 of them, each bit should have been 0 and 1 at least once.
	c := make(map[int]int)
	for i := 0; i < 100; i++ {
		r := rand64()
		for j := 0; j < 64; j++ {
			c[j] += int(r & 1)
			r >>= 1
		}
	}
	for i := 0; i < 64; i++ {
		if c[i] == 0 || c[i] == 100 {
			t.Errorf("Bit %d was always 0 or always 1", i)
		}
	}
}

func TestRandFloat64(t *testing.T) {
	// randFloat64 should return a number between 0 and 1.
	for i := 0; i < 100; i++ {
		r := randFloat64()
		if r < 0 || r >= 1 {
			t.Errorf("randFloat64 returned %f, want [0,1)", r)
		}
	}
}

func TestRandIntn(t *testing.T) {
	// randIntn should return a number between 0 and n.
	for i := 1; i <= 100; i++ {
		r := randIntn(i)
		if r < 0 || r >= i {
			t.Errorf("randIntn returned %d, want [0,%d)", r, i)
		}
	}
}
