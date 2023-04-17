package hexz

import "testing"

func BenchmarkPlayFlagzGame(b *testing.B) {
	winCounts := make(map[int]int)
	for i := 0; i < b.N; i++ {
		ge := &GameEngineFlagz{}
		ge.Init()
		ge.Start()

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
