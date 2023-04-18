package hexz

import "testing"

func mov(ge *GameEngineClassic, row, col int, ct CellType) GameEngineMove {
	return GameEngineMove{
		playerNum: ge.board.Turn,
		row:       row,
		col:       col,
		move:      ge.board.Move,
		cellType:  ct,
	}
}

func TestClassicFullGame(t *testing.T) {

	ge := &GameEngineClassic{}
	ge.Init()
	// Play all cells from (0, 0) row by row.
	for r := 0; r < len(ge.board.Fields); r++ {
		for c := 0; c < len(ge.board.Fields[r]); c++ {
			if !ge.MakeMove(mov(ge, r, c, cellNormal)) {
				t.Fatalf("Cannot make move at (%d,%d)", r, c)
			}
		}
	}
	if !ge.IsDone() {
		t.Error("Game is not done")
	}
	if ge.Winner() != 1 {
		t.Errorf("Want winner 1, got %d", ge.Winner())
	}
	for r := 0; r < len(ge.board.Fields); r++ {
		for c := 0; c < len(ge.board.Fields[r]); c++ {
			f := &ge.board.Fields[r][c]
			if f.Owner <= 0 {
				t.Errorf("Cell has no owner: (%d,%d)", r, c)
			}
			if f.Hidden {
				t.Errorf("Cell is hidden: (%d,%d)", r, c)
			}
		}
	}
}

func TestClassicFloodFill(t *testing.T) {
	ge := &GameEngineClassic{}
	ge.Init()
	// Occupy all cells in row 6. All cells in rows 7 onwards should be occupied afterwards.
	row := 6
	wantOwner := 0
	for c := 0; c < len(ge.board.Fields[row]); c++ {
		wantOwner = ge.board.Turn
		ge.MakeMove(mov(ge, row, c, cellNormal))
	}
	if wantOwner == 0 {
		t.Fatalf("Turn was zero?!")
	}
	for r := 0; r < row; r++ {
		for c := 0; c < len(ge.board.Fields[r]); c++ {
			if gotOwner := ge.board.Fields[r][c].Owner; gotOwner != 0 {
				t.Errorf("Cell (%d,%d) has an unexpected owner P%d", gotOwner, r, c)
			}
		}
	}
	for r := 7; r < len(ge.board.Fields); r++ {
		for c := 0; c < len(ge.board.Fields[r]); c++ {
			if ge.board.Fields[r][c].Owner != wantOwner {
				t.Errorf("Cell (%d,%d) not owned by P%d", wantOwner, r, c)
			}
		}
	}
}

func TestClassicUncoverAfterTwoMoves(t *testing.T) {

	ge := &GameEngineClassic{}
	ge.Init()

	r := 0
	for c := 0; c < 3; c++ {
		if !ge.MakeMove(mov(ge, r, c, cellNormal)) {
			t.Fatalf("Cannot make move at (%d,%d)", r, c)
		}
		wantOwner := c%2 + 1
		if gotOwner := ge.board.Fields[r][c].Owner; gotOwner != wantOwner {
			t.Errorf("Wrong owner in cell (%d,%d): want %v, got %v", r, c, wantOwner, gotOwner)
		}
		// Only the 4th move should uncover the cells.
		wantHidden := c != 3
		if gotHidden := ge.board.Fields[r][c].Hidden; gotHidden != wantHidden {
			t.Errorf("Wrong hidden state in cell (%d,%d): want %v, got %v", r, c, wantHidden, gotHidden)
		}
	}
}

func TestClassicFireCell(t *testing.T) {

	ge := &GameEngineClassic{}
	ge.Init()

	r, c := 4, 4
	if !ge.MakeMove(mov(ge, r, c, cellFire)) {
		t.Fatalf("Cannot make move at (%d,%d)", r, c)
	}
	var ns [6]idx
	n := ge.board.neighbors(idx{r, c}, ns[:])
	if n != 6 {
		t.Fatalf("Cell for test should have 6 neighbors. Got: %d", n)
	}
	for i := 0; i < n; i++ {
		if ge.board.Fields[ns[i].r][ns[i].c].Type != cellDead {
			t.Errorf("No fire in cell (%+v)", ns[i])
		}
	}
}

func TestClassicDeathCell(t *testing.T) {

	ge := &GameEngineClassic{}
	ge.Init()

	r := 0
	for c := 0; c < 4; c++ {
		if !ge.MakeMove(mov(ge, r, c, cellNormal)) {
			t.Fatalf("Cannot make move at (%d,%d)", r, c)
		}
	}
	c := 1
	if !ge.MakeMove(mov(ge, r, c, cellDeath)) {
		t.Errorf("Cannot place death cell at (%d,%d)", r, c)
	}
	if ge.board.Fields[r][c].Type != cellDeath {
		t.Errorf("Wrong cell type. Want %v, got %v", cellDeath, ge.board.Fields[r][c].Type)
	}
	if ge.board.Fields[r][c].Owner != 1 {
		t.Errorf("Wrong owner. Want 1, got %v", ge.board.Fields[r][c].Owner)
	}
}
