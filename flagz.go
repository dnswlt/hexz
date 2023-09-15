package hexz

// The Flagz game. The best one we have.

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math/rand"
)

type GameEngineFlagz struct {
	B *Board
	// Used to efficiently process moves and determine game state for flagz.
	FreeCells   int    // Number of unoccupied cells
	NormalMoves [2]int // Number of normal cell moves the players can make
	// Source of random numbers. Useful to make games repeatable.
	// Not cloned nor serialized!
	rnd *rand.Rand
	// History of moves made so far
	Moves []GameEngineMove
}

func (g *GameEngineFlagz) GameType() GameType { return gameTypeFlagz }

const (
	flagzNumRockCells  = 15 // Odd number, so we have an even number of free cells.
	flagzNumGrassCells = 5
	flagzMaxValue      = 5 // Maximum value a cell can take.
)

func NewGameEngineFlagz(src rand.Source) *GameEngineFlagz {
	g := &GameEngineFlagz{
		rnd: rand.New(src),
	}
	g.Reset()
	return g
}

func (g *GameEngineFlagz) ValidCellTypes() []CellType {
	return []CellType{cellNormal, cellFlag}
}

func (g *GameEngineFlagz) MoveHistory() []GameEngineMove {
	return g.Moves
}

func (g *GameEngineFlagz) PopulateInitialCells() {
	i := 0
	n := len(g.B.FlatFields)
	// j is only a safeguard for invalid calls to this method on a non-empty board.
	for j := 0; j < n && i < flagzNumRockCells; j++ {
		k := g.rnd.Intn(n)
		if !g.B.FlatFields[k].occupied() {
			i++
			f := &g.B.FlatFields[k]
			f.Type = cellRock
			f.Lifetime = -1
		}
	}
	// Place some grass cells.
	v := 0
	for j := 0; j < n && v < flagzNumGrassCells; j++ {
		k := g.rnd.Intn(n)
		if !g.B.FlatFields[k].occupied() {
			v++
			f := &g.B.FlatFields[k]
			f.Type = cellGrass
			f.Lifetime = -1
			f.Value = v
		}
	}
	// Reset freeCells and normalMoves.
	g.FreeCells = 0
	for i := 0; i < len(g.B.FlatFields); i++ {
		if !g.B.FlatFields[i].occupied() {
			g.FreeCells++
		}
	}
	g.NormalMoves = [2]int{0, 0}
}

func (g *GameEngineFlagz) InitializeResources() {
	g.B.Resources = make([]ResourceInfo, 2)
	var ps [cellTypeLen]int
	ps[cellNormal] = -1
	ps[cellFlag] = 3
	for i := 0; i < len(g.B.Resources); i++ {
		g.B.Resources[i].NumPieces = ps
	}
}

func (g *GameEngineFlagz) NumPlayers() int { return 2 }

func (g *GameEngineFlagz) Reset() {
	g.B = NewBoard()
	g.B.Score = make([]int, 2)
	g.InitializeResources()
	g.PopulateInitialCells()
	g.B.State = Running
}

func (g *GameEngineFlagz) recomputeState() {
	b := g.B
	canMove1 := g.NormalMoves[0] > 0 || (g.FreeCells > 0 && b.Resources[0].NumPieces[cellFlag] > 0)
	canMove2 := g.NormalMoves[1] > 0 || (g.FreeCells > 0 && b.Resources[1].NumPieces[cellFlag] > 0)
	// If only one player has valid moves left, it's their turn.
	if !canMove1 && canMove2 {
		b.Turn = 2
	} else if canMove1 && !canMove2 {
		b.Turn = 1
	}
	// Check if the game is over:
	// * No open cells left
	// * None of the players can move
	// * One player cannot move, the other one has a higher score
	s := b.Score
	if g.FreeCells == 0 || !(canMove1 || canMove2) ||
		(s[0] > s[1] && !canMove2) || (s[1] > s[0] && !canMove1) {
		b.State = Finished
	}
}

func (g *GameEngineFlagz) lifetime(ct CellType) int {
	switch ct {
	case cellNormal, cellFlag:
		return -1
	}
	return 0
}

// Updates the Blocked and nextVal status of all cells neighboring (r, c).
// Also updates the counts of availalbe moves per player in g.
// (r, c) must already have been updated.
func (g *GameEngineFlagz) updateNeighborCells(r, c int) {
	b := g.B
	var ns [6]idx
	f := &b.Fields[r][c]
	if f.Owner == 0 {
		return
	}
	pIdx := f.Owner - 1
	n := b.neighbors(idx{r, c}, ns[:])
	for i := 0; i < n; i++ {
		nb := &b.Fields[ns[i].r][ns[i].c]
		if !nb.occupied() {
			if f.Value == flagzMaxValue {
				// New 5 => neighbors get blocked for normal moves.
				if nb.NextVal[pIdx] > 0 {
					g.NormalMoves[pIdx]--
				}
				nb.Blocked |= 1 << pIdx
				nb.NextVal[pIdx] = -1
			} else if nb.NextVal[pIdx] == 0 {
				g.NormalMoves[pIdx]++
				nb.NextVal[pIdx] = f.Value + 1
			} else if nb.NextVal[pIdx] > f.Value+1 {
				nb.NextVal[pIdx] = f.Value + 1
			}
		} else if nb.Type == cellGrass && nb.Value <= f.Value {
			nb.Type = cellNormal
			nb.Owner = f.Owner
			b.Score[pIdx] += nb.Value
			// Now recurse to process grass cell neighbors.
			g.updateNeighborCells(ns[i].r, ns[i].c)
		}
	}
}

func (g *GameEngineFlagz) MakeMove(m GameEngineMove) bool {
	b := g.B
	turn := b.Turn
	pIdx := turn - 1
	if m.PlayerNum != turn || m.Move != b.Move {
		// Only allow moves by players whose turn it is.
		return false
	}
	if !b.valid(idx{m.Row, m.Col}) {
		// Invalid move request.
		return false
	}
	f := &b.Fields[m.Row][m.Col]
	if f.occupied() {
		return false
	}
	if b.Resources[pIdx].NumPieces[m.CellType] == 0 {
		// No pieces left of requested type
		return false
	}
	if m.CellType == cellNormal {
		val := f.NextVal[pIdx]
		if val <= 0 {
			return false
		}
		f.Owner = turn
		f.Type = cellNormal
		f.Lifetime = g.lifetime(cellNormal)
		f.Hidden = false
		f.Value = val
		b.Score[pIdx] += val
		g.FreeCells--
		g.NormalMoves[pIdx]--
		if f.NextVal[1-pIdx] > 0 {
			g.NormalMoves[1-pIdx]--
		}
		g.updateNeighborCells(m.Row, m.Col)
	} else if m.CellType == cellFlag {
		// A flag can be placed on any free cell. It does not add to the score.
		f.Owner = turn
		f.Type = cellFlag
		f.Lifetime = g.lifetime(cellFlag)
		f.Hidden = false
		f.Value = 0
		b.Resources[turn-1].NumPieces[cellFlag]--
		g.FreeCells--
		// Adjust available normal moves for both players if f was available to them.
		if f.NextVal[pIdx] > 0 {
			g.NormalMoves[pIdx]--
		}
		if f.NextVal[1-pIdx] > 0 {
			g.NormalMoves[1-pIdx]--
		}
		g.updateNeighborCells(m.Row, m.Col)
	} else {
		// Invalid piece. Should be caught by resource check already, so never reached.
		return false
	}
	b.Turn = 3 - b.Turn // Usually it's the other player's turn. If not, recomputeState will fix that.
	b.Move++
	g.recomputeState()
	g.Moves = append(g.Moves, m)
	return true
}

func (g *GameEngineFlagz) Board() *Board { return g.B }
func (g *GameEngineFlagz) Clone(s rand.Source) SinglePlayerGameEngine {
	return &GameEngineFlagz{
		B:           g.B.copy(),
		rnd:         rand.New(s),
		FreeCells:   g.FreeCells,
		NormalMoves: g.NormalMoves,
	}
}

// Serializes the state of this game engine.
// Currently the encoding is as a Gob, but that can change at any time.
func (g *GameEngineFlagz) Encode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	flds := g.B.Fields
	g.B.Fields = nil // Don't serialize the fields, since they are redundant.
	defer func() { g.B.Fields = flds }()
	if err := enc.Encode(g); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Decodes the given encoded state of a game engine and sets this game engine to the given state.
// The random source of the existing game engine is kept, since the serialized state does not contain one.
func (g *GameEngineFlagz) Decode(encoded []byte) error {
	dec := gob.NewDecoder(bytes.NewReader(encoded))
	var ge GameEngineFlagz
	if err := dec.Decode(&ge); err != nil {
		return err
	}
	if ge.B == nil {
		return fmt.Errorf("board is nil")
	}
	// Check that the board has the expected size.
	const numFields = numFieldsFirstRow*((numBoardRows+1)/2) + (numFieldsFirstRow-1)*(numBoardRows/2)
	if len(ge.B.FlatFields) != numFields {
		return fmt.Errorf("board has %d fields, want %d", len(ge.B.Fields), numFields)
	}
	// Regenerate fields slices.
	ge.B.Fields = make([][]Field, numBoardRows)
	start := 0
	for i := 0; i < len(ge.B.Fields); i++ {
		end := start + numFieldsFirstRow - i%2
		ge.B.Fields[i] = ge.B.FlatFields[start:end]
		start = end
	}
	// Keep old random source, which is not serialized.
	rnd := g.rnd
	*g = ge
	g.rnd = rnd
	return nil
}

func (g *GameEngineFlagz) IsDone() bool {
	return g.B.State == Finished
}

func (g *GameEngineFlagz) Winner() (playerNum int) {
	if !g.IsDone() {
		return 0
	}
	return scoreBasedSingleWinner(g.B.Score)
}

// Suggests a move for the player whose turn it is.
// Uses a random strategy. Probably not very smart.
func (g *GameEngineFlagz) RandomMove() (*GameEngineMove, error) {
	if g.B.State != Running {
		return nil, fmt.Errorf("game is not running")
	}
	b := g.B
	pIdx := b.Turn - 1
	nMoves := g.NormalMoves[pIdx]
	flagsLeft := b.Resources[pIdx].NumPieces[cellFlag] > 0
	pickFlag := false
	if g.FreeCells > 0 && flagsLeft && (nMoves == 0 || g.rnd.Float64() <= float64(1)/float64(nMoves+1)) {
		pickFlag = true
	}
	if pickFlag {
		// Find a random free cell and place a flag there.
		nthFlag := g.rnd.Intn(g.FreeCells)
		n := 0
		for r := range b.Fields {
			for c := range b.Fields[r] {
				if !b.Fields[r][c].occupied() {
					if n == nthFlag {
						return &GameEngineMove{PlayerNum: b.Turn, Move: b.Move, Row: r, Col: c, CellType: cellFlag}, nil
					}
					n++
				}
			}
		}
	} else {
		// Pick a normal move
		nthMove := g.rnd.Intn(nMoves)
		n := 0
		for r := range b.Fields {
			for c := range b.Fields[r] {
				f := &b.Fields[r][c]
				if !f.occupied() && f.isAvail(b.Turn) {
					if n == nthMove {
						return &GameEngineMove{PlayerNum: b.Turn, Move: b.Move, Row: r, Col: c, CellType: cellNormal}, nil
					}
					n++
				}
			}
		}
	}
	data, _ := json.MarshalIndent(g, "", "  ")
	panic(fmt.Sprintf("no legal move found: %v", string(data)))
}

func (g *GameEngineFlagz) ValidMoves() []*GameEngineMove {
	result := make([]*GameEngineMove, 0, 64)
	if g.B.State != Running {
		return result
	}
	b := g.B
	flagsLeft := b.Resources[b.Turn-1].NumPieces[cellFlag] > 0
	for r := range b.Fields {
		for c := range b.Fields[r] {
			f := &b.Fields[r][c]
			if !f.occupied() && flagsLeft {
				result = append(result, &GameEngineMove{
					PlayerNum: b.Turn, Move: b.Move, Row: r, Col: c, CellType: cellFlag,
				})
			}
			if !f.occupied() && f.isAvail(b.Turn) {
				result = append(result, &GameEngineMove{PlayerNum: b.Turn, Move: b.Move, Row: r, Col: c, CellType: cellNormal})
			}
		}
	}
	return result
}
