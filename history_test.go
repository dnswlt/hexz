package hexz

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestGameIdPath(t *testing.T) {
	tests := []struct {
		name   string
		gameId string
		want   string
	}{
		{"lowercase", "abcdef", "AB/abcdef.gob"},
		{"uppercase", "ABCDEF", "AB/ABCDEF.gob"},
		{"empty", "", "_/_.gob"},
		{"short", "A", "A/A.gob"},
		{"short", "AB", "AB/AB.gob"},
		{"long", "ABCDEF123123", "AB/ABCDEF123123.gob"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := gameIdPath(test.gameId); got != test.want {
				t.Errorf("Unexpected path for gameId: want: %q, got: %q", test.want, got)
			}
		})
	}
}

func TestSaveReadGame(t *testing.T) {
	dir := t.TempDir()
	gameId := GenerateGameId()
	board0 := NewBoard().ViewFor(0)
	board1 := NewBoard().ViewFor(0)
	board1.Move = board0.Move + 1 // Ensure board1 is different from board0
	hist := []*GameHistoryEntry{
		{Board: board0},
		{Board: board1},
	}
	w, err := NewHistoryWriter(dir, gameId)
	if err != nil {
		t.Fatalf("could not create history writer: %s", err)
	}
	for _, h := range hist {
		w.Write(h)
	}
	w.Close()
	readHist, err := ReadGameHistory(dir, gameId)
	if err != nil {
		t.Fatalf("cannot read game history: %s", err.Error())
	}
	if len(readHist) != len(hist) {
		t.Errorf("wrong number of history entries: want %d, got %d", len(hist), len(readHist))
	}
	// Use EquateEmpty here b/c gob decodes empty slices as nil.
	if diff := cmp.Diff(hist, readHist, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("read history not equal to write history: -want +got: %s", diff)
	}
}
