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
		{"lowercase", "abcdef", "AB/abcdef.ggz"},
		{"uppercase", "ABCDEF", "AB/ABCDEF.ggz"},
		{"empty", "", "_/_.ggz"},
		{"short", "A", "A/A.ggz"},
		{"short", "AB", "AB/AB.ggz"},
		{"long", "ABCDEF123123", "AB/ABCDEF123123.ggz"},
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
	entries := []*GameHistoryEntry{
		{EntryType: "move", Board: board0},
		{EntryType: "move", Board: board1},
	}
	w, err := NewHistoryWriter(dir, gameId)
	if err != nil {
		t.Fatalf("could not create history writer: %s", err)
	}
	for _, h := range entries {
		w.Write(h)
	}
	w.Close()
	hist, err := ReadGameHistory(dir, gameId)
	if err != nil {
		t.Fatalf("cannot read game history: %s", err.Error())
	}
	if len(hist.Entries) != len(entries) {
		t.Errorf("wrong number of history entries: want %d, got %d", len(entries), len(hist.Entries))
	}
	// Use EquateEmpty here b/c gob decodes empty slices as nil.
	if diff := cmp.Diff(entries, hist.Entries, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("read history not equal to write history: -want +got: %s", diff)
	}
}

func TestReadGameHistoryHeader(t *testing.T) {
	dir := t.TempDir()
	gameId := GenerateGameId()
	w, err := NewHistoryWriter(dir, gameId)
	if err != nil {
		t.Fatalf("could not create history writer: %s", err)
	}
	header := &GameHistoryHeader{
		GameId:      gameId,
		GameType:    gameTypeFlagz,
		PlayerNames: []string{"peter", "paul"},
	}
	w.WriteHeader(header)
	w.Close()
	hist, err := ReadGameHistory(dir, gameId)
	if err != nil {
		t.Fatalf("cannot read game history: %s", err.Error())
	}
	if hist.Header == nil {
		t.Fatalf("no header read")
	}
	if diff := cmp.Diff(hist.Header, header, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("read header not equal to original: -want +got: %s", diff)
	}
}
