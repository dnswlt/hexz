package hexz

import (
	"bytes"
	"encoding/gob"
	"io"
	"math/rand"
	"testing"
)

func BenchmarkPlayFlagzGame(b *testing.B) {
	winCounts := make(map[int]int)
	src := rand.NewSource(123)
	for i := 0; i < b.N; i++ {
		ge := NewGameEngineFlagz(src)

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

func TestGobEncodeGameEngineFlagz(t *testing.T) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	src := rand.NewSource(123)
	g1 := NewGameEngineFlagz(src)
	g2 := NewGameEngineFlagz(src)
	enc.Encode(g1)
	enc.Encode(g2)
	if buf.Len() > 5000 {
		t.Errorf("Want buffer length <=%d, got %d", 5000, buf.Len())
	}
	dec := gob.NewDecoder(&buf)
	var r1 *GameEngineFlagz
	if err := dec.Decode(&r1); err != nil {
		t.Error("Cannot decode: ", err)
	}
	if r1.FreeCells != g1.FreeCells {
		t.Errorf("Wrong FreeCells: want %d, got %d", g1.FreeCells, r1.FreeCells)
	}
	var r2 *GameEngineFlagz
	if err := dec.Decode(&r2); err != nil {
		t.Error("Cannot decode: ", err)
	}
	var r3 *GameEngineFlagz
	if err := dec.Decode(&r3); err != io.EOF {
		t.Error("Expected EOF, got: ", err)
	}
}
