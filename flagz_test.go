package hexz

import (
	"math/rand"
	"testing"
)

func BenchmarkPlayFlagzGame(b *testing.B) {
	winCounts := make(map[int]int)
	src := rand.NewSource(123)
	for i := 0; i < b.N; i++ {
		ge := NewGameEngineFlagz(src)

		for !ge.IsDone() {
			m, err := ge.RandomMove()
			if err != nil {
				b.Fatal("Could not suggest a move:", err.Error())
			}
			if !ge.MakeMove(m) {
				b.Fatal("Could not make a move")
				return
			}
		}
		winCounts[ge.Winner()]++
	}
	b.Logf("winCounts: %v", winCounts)
}
