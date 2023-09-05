package hexz

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestGameIdPath(t *testing.T) {
	tests := []struct {
		name   string
		gameId string
		want   string
	}{
		{"lowercase", "abcdef", "AB/abcdef.json"},
		{"uppercase", "ABCDEF", "AB/ABCDEF.json"},
		{"empty", "", "_/_.json"},
		{"short", "A", "A/A.json"},
		{"short", "AB", "AB/AB.json"},
		{"long", "ABCDEF123123", "AB/ABCDEF123123.json"},
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
	board := NewBoard().ViewFor(0)
	hist := []*GameHistoryEntry{
		{Board: board},
	}
	WriteGameHistory(dir, gameId, hist)
	readHist, err := ReadGameHistory(dir, gameId)
	if err != nil {
		t.Fatalf("cannot read game history: %s", err.Error())
	}
	if len(readHist) != len(hist) {
		t.Fatalf("wrong number of history entries: want %d, got %d", len(hist), len(readHist))
	}
	if diff := cmp.Diff(hist, readHist); diff != "" {
		t.Errorf("read history not equal to write history: -want +got: %s", diff)
	}
}
