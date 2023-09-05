package hexz

import (
	"testing"
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
