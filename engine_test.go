package hexz

import (
	"fmt"
	"testing"
	"time"
)

// func makeTestBoard() *Board {
// 	flatFields, fields := makeFields()
// 	b := &Board{
// 		Turn:       1, // Player 1 begins
// 		FlatFields: flatFields,
// 		Fields:     fields,
// 		State:      Initial,
// 		Score:      []int{0, 0},
// 	}
// 	numPlayers := 2
// 	b.Score = make([]int, numPlayers)
// 	b.Resources = make([]ResourceInfo, numPlayers)
// 	for i := 0; i < numPlayers; i++ {
// 		b.Resources[i] = g.InitialResources()
// 	}
// 	return b
// }

func TestScoreBasedSingleWinner(t *testing.T) {
	tests := []struct {
		score []int
		want  int
	}{
		{[]int{0, 0}, 0},
		{[]int{1, 1}, 0},
		{[]int{0, 1}, 2},
		{[]int{3, 2}, 1},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if got := scoreBasedSingleWinner(test.score); got != test.want {
				t.Errorf("want: %v, got: %v", test.want, got)
			}
		})
	}
}

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

func TestMCTS(t *testing.T) {
	ge := &GameEngineFlagz{}
	ge.Init()
	ge.Start()

	mcts := NewMCTS(ge)
	var want *MCTSStats
	if _, stats := mcts.SuggestMove(time.Duration(10) * time.Second); stats != want {
		t.Errorf("Got: %s", stats)
	}

}

func TestMCTSFull(tt *testing.T) {
	ge := &GameEngineFlagz{}
	ge.Init()
	ge.Start()

	var allStats [2][]*MCTSStats
	mcts := []*MCTS{
		NewMCTS(ge),
		NewMCTS(ge),
	}
	// mcts[0].maxFlagPositions = 10
	mcts[0].UctFactor = 2.0
	for !ge.IsDone() {
		t := ge.Board().Turn - 1
		thinkTime := 5000
		if t == 1 {
			thinkTime = 5000
		}
		m, stats := mcts[t].SuggestMove(time.Duration(thinkTime) * time.Millisecond)
		allStats[t] = append(allStats[t], stats)
		fmt.Print(stats)
		if !ge.MakeMove(m) {
			tt.Fatal("Cannot make move")
		}
		fmt.Printf("score: %v moved:%d\n", ge.Board().Score, t)
	}
	// Print aggregate stats
	agg := []*MCTSStats{{}, {}}
	for i := 0; i < 2; i++ {
		for j := 0; j < len(allStats[i]); j++ {
			s := allStats[i][j]
			agg[i].Iterations += s.Iterations
			agg[i].Elapsed += s.Elapsed
			if agg[i].MaxDepth < s.MaxDepth {
				agg[i].MaxDepth = s.MaxDepth
			}
			agg[i].NotMoved += s.NotMoved
			if agg[i].TreeSize < s.TreeSize {
				agg[i].TreeSize = s.TreeSize
			}
		}
	}
	fmt.Print("Aggregated stats P1:\n", agg[0].String())
	fmt.Print("Aggregated stats P2:\n", agg[1].String())
}
