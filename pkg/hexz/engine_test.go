package hexz

import (
	"fmt"
	"testing"

	"github.com/dnswlt/hexz/internal/xrand"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
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
	// Create random board, make some changes to it.
	board := NewBoard()
	board.State = Finished
	board.Turn = 2
	board.Move = 10
	board.Fields[0][0].Type = cellFlag
	board.Fields[0][0].Owner = 1
	// Encode and decode
	bp := board.Proto()
	decoded := NewBoard()
	err := decoded.FromProto(bp)
	if err != nil {
		t.Fatal("cannot decode: ", err)
	}
	// Compare
	if diff := cmp.Diff(board, decoded); diff != "" {
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
		err = board.FromProto(bp2)
		if err != nil {
			b.Fatal("cannot decode: ", err)
		}
	}
}

func testFlagzGameRepr(t testing.TB) *GameRepr {
	t.Helper()
	ge := NewGameEngineFlagz()
	enc := ge.Proto()
	return NewGameRepr(&pb.GameState{
		GameInfo: &pb.GameInfo{
			Id:      "TTESTT",
			Host:    "testhost",
			Started: tpb.Now(),
			Type:    string(gameTypeFlagz),
		},
		Players: []*pb.Player{
			{Id: "P1", Name: "P1"},
			{Id: "P2", Name: "P2"},
		},
		EngineState: enc,
		UndoRedoState: &pb.GameState_UndoRedoState{
			InitialState: enc,
		},
	})
}

func TestGameReprUndoRedo(t *testing.T) {
	g := testFlagzGameRepr(t)
	numMoves := 4
	// Make random moves
	for i := 0; i < numMoves; i++ {
		moves := g.Engine().(*GameEngineFlagz).ValidMoves()
		j := xrand.Intn(len(moves))
		if err := g.MakeMove(*moves[j]); err != nil {
			t.Fatal("Cannot make move:", err)
		}
	}
	// Undo N moves, redo them
	for i := 1; i < numMoves; i++ {
		for j := 0; j < i; j++ {
			if err := g.Undo(); err != nil {
				t.Fatalf("undo failed at %d:%d: %v", i, j, err)
			}
		}
		for j := 0; j < i; j++ {
			if err := g.Redo(); err != nil {
				t.Fatalf("redo failed at %d:%d: %v", i, j, err)
			}
		}
		if n := g.Engine().Board().Move; n != numMoves {
			t.Errorf("not at move %d: %d", numMoves, n)
		}
	}
}

func BenchmarkUndoRedoLastMove(b *testing.B) {
	// The most expensive Undo is undoing the last move, since it
	// has to repeat all moves from the beginning.
	g := testFlagzGameRepr(b)
	for !g.Engine().IsDone() {
		moves := g.Engine().(*GameEngineFlagz).ValidMoves()
		j := xrand.Intn(len(moves))
		if err := g.MakeMove(*moves[j]); err != nil {
			b.Fatal("Cannot make move:", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := g.Undo(); err != nil {
			b.Fatal("undo failed", err)
		}
		if err := g.Redo(); err != nil {
			b.Fatal("redo failed", err)
		}
	}

}

func BenchmarkUndoRedoFirstMove(b *testing.B) {
	// The cheapest Undo is undoing the first move, since it
	// only has to repeat one move.
	//
	// This takes ~ 12us on an M1, while UndoRedoLastMove
	// takes 14us. So virtually all time is spent recreating
	// the game engine from its proto representation. Making
	// moves costs almost nothing.
	g := testFlagzGameRepr(b)
	moves := g.Engine().(*GameEngineFlagz).ValidMoves()
	j := xrand.Intn(len(moves))
	if err := g.MakeMove(*moves[j]); err != nil {
		b.Fatal("Cannot make move:", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := g.Undo(); err != nil {
			b.Fatal("undo failed", err)
		}
		if err := g.Redo(); err != nil {
			b.Fatal("redo failed", err)
		}
	}

}

func TestGameReprEncodedSize(t *testing.T) {
	// This test ensures that encoded game state sizes stay withing
	// reasonable bounds, to avoid blowing up the DB or memory stores.
	g := testFlagzGameRepr(t)
	{
		data, err := proto.Marshal(g.state)
		if err != nil {
			t.Fatal("Cannot marshal proto: ", err)
		}
		l := len(data)
		if l < 1500 || l > 2500 {
			t.Errorf("Unexpected size of marshalled GameState: got %d, want [1500, 2500]", l)
		}
	}
	// Play till the end
	for !g.Engine().IsDone() {
		mv, err := g.Engine().(*GameEngineFlagz).RandomMove()
		if err != nil {
			t.Fatal("RandomMove: ", err)
		}
		g.MakeMove(mv)
	}

	{
		data, err := proto.Marshal(g.state)
		if err != nil {
			t.Fatal("Cannot marshal proto: ", err)
		}
		l := len(data)
		if l < 3500 || l > 4500 {
			t.Errorf("Unexpected size of marshalled GameState: got %d, want [3500, 4500]", l)
		}
	}

}
