package hexz

import (
	"fmt"
	"testing"
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
