package hexz

import (
	"fmt"
	"testing"

	pb "github.com/dnswlt/hexz/hexzpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
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

func TestBoardProto(t *testing.T) {
	board := NewBoard()
	orig := board.copy() // Decode into a copy so we can compare.
	bp := board.Proto()
	err := board.DecodeProto(bp)
	if err != nil {
		t.Fatal("cannot decode: ", err)
	}
	if diff := cmp.Diff(orig, board); diff != "" {
		t.Errorf("board mismatch (-want +got):\n%s", diff)
	}
}

func BenchmarkBoardProtoMarshalUnmarshal(b *testing.B) {
	for i := 0; i < b.N; i++ {
		board := NewBoard()
		bp := board.Proto()
		data, err := proto.Marshal(bp)
		if err != nil {
			b.Fatal("cannot marshal: ", err)
		}
		bp2 := &pb.Board{}
		err = proto.Unmarshal(data, bp2)
		if err != nil {
			b.Fatal("cannot unmarshal: ", err)
		}
		err = board.DecodeProto(bp2)
		if err != nil {
			b.Fatal("cannot decode: ", err)
		}
	}
}
