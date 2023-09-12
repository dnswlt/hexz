package hexz

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

var (
	runExperimentsAsTests = flag.Bool("run-experiments-as-tests", false,
		"Set to true to run (failing) test cases that are used for experiments")
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
	if testing.Short() {
		return
	}
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
	// Count % of won games if the first move is a flag placed in cell (r, c), for all r, c.
	if !*runExperimentsAsTests {
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
				if !ge.MakeMove(GameEngineMove{PlayerNum: 1, Move: 0, Row: r, Col: c, CellType: cellFlag}) {
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
	maxWinRate := 0.0
	minWinRate := math.MaxFloat64
	allWins := make(map[string]float64)
	for r := range wins {
		for c := range wins[r] {
			if played[r][c] > 0 {
				winRate := float64(wins[r][c]) / float64(played[r][c])
				allWins[fmt.Sprintf("%d:%d", r, c)] = winRate
				if winRate < minWinRate {
					minWinRate = winRate
				}
				if winRate > maxWinRate {
					maxWinRate = winRate
				}
			}
		}
	}
	j, _ := json.Marshal(allWins)
	t.Errorf("Max: %f, min: %f, %v", maxWinRate, minWinRate, string(j))
}

type fieldDesc struct {
	freeNeighbors int // Number of free neighboring cells.
	areas         int // Number of areas that this cell connects.
	dist          int // min distance to the edge of the board.
}

func getFd(b *Board, r, c int) fieldDesc {
	var fd fieldDesc
	var ns [6]idx
	n := b.neighbors(idx{r, c}, ns[:])
	inArea := false
	for i := 0; i < n; i++ {
		if !b.Fields[ns[i].r][ns[i].c].occupied() {
			fd.freeNeighbors++
			if !inArea {
				fd.areas++
			}
			inArea = true
		} else {
			inArea = false
		}
	}
	// correct areas if first and last cell (out of 6) were both not occupied.
	// since they belong to the same area.
	if n == 6 && !b.Fields[ns[0].r][ns[0].c].occupied() && !b.Fields[ns[n-1].r][ns[n-1].c].occupied() && fd.areas > 1 {
		fd.areas--
	}
	fd.dist = r
	if c < r {
		fd.dist = c
	}
	if len(b.Fields)-r-1 < fd.dist {
		fd.dist = len(b.Fields) - r - 1
	}
	if len(b.Fields[r])-c-1 < fd.dist {
		fd.dist = len(b.Fields[r]) - c - 1
	}
	return fd
}

func TestCompareCellValueByCharacteristic(t *testing.T) {
	// Characterise each cell by a few properties such as free neighbor cells.
	// Then compute % of won games that had a flag in those cell "types".
	if !*runExperimentsAsTests {
		return
	}
	const nRounds = 1000000
	src := rand.NewSource(123)
	wins := make(map[fieldDesc]int)
	played := make(map[fieldDesc]int)

	for i := 0; i < nRounds; i++ {
		ge := NewGameEngineFlagz(src)
		// Compute cell characteristics before playing the game, as they will change during the game.
		b := ge.B
		fds := make([][]fieldDesc, len(b.Fields))
		for r := range b.Fields {
			fds[r] = make([]fieldDesc, len(b.Fields[r]))
			for c := range b.Fields[r] {
				fds[r][c] = getFd(b, r, c)
			}
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
		// Find cells where flags were placed.
		winner := ge.Winner()
		for r := range b.Fields {
			for c := range b.Fields[r] {
				if b.Fields[r][c].Type == cellFlag {
					fd := fds[r][c]
					played[fd]++
					if b.Fields[r][c].Owner == winner {
						wins[fd]++
					}
				}
			}
		}
	}
	type kv struct {
		fieldDesc
		winRate float64
		played  int
	}
	st := make([]kv, len(played))
	i := 0
	for fd, n := range played {
		st[i] = kv{
			fieldDesc: fd, played: n, winRate: float64(wins[fd]) / float64(played[fd]),
		}
		i++
	}
	sort.Slice(st, func(i, j int) bool {
		return st[i].winRate > st[j].winRate
	})
	var sb strings.Builder
	for i := range st {
		sb.WriteString(fmt.Sprintf("free_nb:%d areas:%d dist:%d winr:%.3f N:%d\n", st[i].freeNeighbors, st[i].areas, st[i].dist,
			st[i].winRate, st[i].played))
	}
	t.Errorf("Stats: %v", sb.String())
}

func TestGobEncodeGameEngineFlagz(t *testing.T) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	src := rand.NewSource(123)
	g1 := NewGameEngineFlagz(src)
	g2 := NewGameEngineFlagz(src)
	if err := enc.Encode(g1); err != nil {
		t.Fatal("Cannot encode first: ", err)
	}
	if err := enc.Encode(g2); err != nil {
		t.Fatal("Cannot encode second: ", err)
	}
	if buf.Len() > 5000 {
		t.Errorf("Want buffer length <=%d, got %d", 5000, buf.Len())
	}
	dec := gob.NewDecoder(&buf)
	var r1 *GameEngineFlagz
	if err := dec.Decode(&r1); err != nil {
		t.Fatal("Cannot decode: ", err)
	}
	if r1.FreeCells != g1.FreeCells {
		t.Errorf("Wrong FreeCells: want %d, got %d", g1.FreeCells, r1.FreeCells)
	}
	var r2 *GameEngineFlagz
	if err := dec.Decode(&r2); err != nil {
		t.Fatal("Cannot decode: ", err)
	}
	var r3 *GameEngineFlagz
	if err := dec.Decode(&r3); err != io.EOF {
		t.Error("Expected EOF, got: ", err)
	}
}
