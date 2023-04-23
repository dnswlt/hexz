package hexz

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"strconv"
	"strings"
	"testing"
	"time"
)

func (b *Board) DebugString() string {
	var sb strings.Builder
	for r := range b.Fields {
		if r&1 == 1 {
			sb.WriteString(" ")
		}
		for c := range b.Fields[r] {
			f := &b.Fields[r][c]
			r := "."
			if f.Owner == 1 {
				if f.Type == cellFlag {
					r = "F"
				} else {
					r = "X"
					// r = fmt.Sprintf("%c", 'a'+f.Value)
				}
			} else if f.Owner == 2 {
				if f.Type == cellFlag {
					r = "f"
				} else {
					r = "O"
					// r = fmt.Sprintf("%c", 'A'+f.Value)
				}
			} else {
				if f.Type == cellGrass {
					r = strconv.Itoa(f.Value)
				} else if f.Type == cellRock {
					r = "-"
				} else {
					r = "."
				}
			}
			sb.WriteString(r + " ")
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

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

func TestPlayGreedyFlagzGame(t *testing.T) {
	const nRuns = 10000
	var winCounts [3]int
	src := rand.NewSource(time.Now().UnixNano())
	for i := 0; i < nRuns; i++ {
		ge := NewGameEngineFlagz(src)
		for !ge.IsDone() {
			var m GameEngineMove
			var err error
			if ge.B.Turn == 1 {
				m, err = ge.RandomMove()
			} else {
				m, err = ge.RandomMoveGreedy()
			}
			if err != nil {
				t.Fatal("Could not suggest a move:", err.Error())
			}
			if !ge.MakeMove(m) {
				t.Fatal("Could not make a move")
				return
			}
			if nRuns == 1 {
				// Only debug first game.
				fmt.Println(ge.B.DebugString())
			}
		}
		winCounts[ge.Winner()]++
	}
	t.Logf("winCounts: %v", winCounts)
}

func TestCompareCellValueByRandomGamePlay(t *testing.T) {
	if testing.Short() {
		return
	}
	src := rand.NewSource(123)
	ge0 := NewGameEngineFlagz(src)
	wins := make([][]int, len(ge0.Board().Fields))
	played := make([][]int, len(ge0.Board().Fields))
	for r := range ge0.Board().Fields {
		wins[r] = make([]int, len(ge0.Board().Fields[r]))
		played[r] = make([]int, len(ge0.Board().Fields[r]))
	}

	b0 := ge0.Board()
	const nRounds = 10000
	for i := 0; i < nRounds; i++ {
		for r := range ge0.Board().Fields {
			for c := range ge0.Board().Fields[r] {
				if b0.Fields[r][c].occupied() {
					continue
				}
				ge := ge0.Clone(src)
				if !ge.MakeMove(GameEngineMove{playerNum: 1, move: 0, row: r, col: c, cellType: cellFlag}) {
					t.Fatal("cannot make initial move")
				}
				for !ge.IsDone() {
					m, err := ge.RandomMove()
					if err != nil {
						t.Fatal("Could not suggest a move:", err.Error())
					}
					if !ge.MakeMove(m) {
						t.Fatal("Could not make a move")
						return
					}
				}
				played[r][c]++
				if ge.Winner() == 1 {
					wins[r][c]++
				}
			}
		}
	}
	maxWins := 0
	minWins := nRounds + 1
	allWins := make(map[int]int)
	for r := range wins {
		for c := range wins[r] {
			if wins[r][c] > maxWins {
				maxWins = wins[r][c]
			}
			if played[r][c] > 0 {
				allWins[wins[r][c]]++
				if wins[r][c] < minWins {
					minWins = wins[r][c]
				}
			}
		}
	}
	j, _ := json.Marshal(allWins)
	t.Errorf("Max: %d, min: %d, %v", maxWins, minWins, string(j))
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
