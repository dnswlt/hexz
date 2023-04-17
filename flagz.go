package hexz

// The Flagz game. The best one we have.

import (
	"fmt"
	"math/rand"
	"time"
)

type GameEngineFlagz struct {
	board *Board
	rnd   *rand.Rand
	// Used to efficiently process moves and determine game state for flagz.
	freeCells   int    // Number of unoccupied cells
	normalMoves [2]int // Number of normal cell moves the players can make
}

func (g *GameEngineFlagz) GameType() GameType { return gameTypeFlagz }

const (
	flagzNumRockCells  = 15 // Odd number, so we have an even number of free cells.
	flagzNumGrassCells = 5
	flagzMaxValue      = 5 // Maximum value a cell can take.
)

func (g *GameEngineFlagz) Init() {
	g.board = InitBoard(g)
	g.rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
}

func (g *GameEngineFlagz) Start() {
	i := 0
	n := len(g.board.FlatFields)
	// j is only a safeguard for invalid calls to this method on a non-empty board.
	for j := 0; j < n && i < flagzNumRockCells; j++ {
		k := g.rnd.Intn(n)
		if !g.board.FlatFields[k].occupied() {
			i++
			f := &g.board.FlatFields[k]
			f.Type = cellRock
			f.lifetime = -1
		}
	}
	// Place some grass cells.
	v := 0
	for j := 0; j < n && v < flagzNumGrassCells; j++ {
		k := g.rnd.Intn(n)
		if !g.board.FlatFields[k].occupied() {
			v++
			f := &g.board.FlatFields[k]
			f.Type = cellGrass
			f.lifetime = -1
			f.Value = v
		}
	}
	// Reset freeCells and normalMoves.
	g.freeCells = 0
	for i := 0; i < len(g.board.FlatFields); i++ {
		if !g.board.FlatFields[i].occupied() {
			g.freeCells++
		}
	}
	g.normalMoves = [2]int{0, 0}

	g.board.State = Running
}

func (g *GameEngineFlagz) InitialResources() ResourceInfo {
	return ResourceInfo{
		NumPieces: map[CellType]int{
			cellNormal: -1, // unlimited
			cellFlag:   3,
		},
	}
}

func (g *GameEngineFlagz) NumPlayers() int { return 2 }
func (g *GameEngineFlagz) Reset() {
	g.Init()
}

func (g *GameEngineFlagz) recomputeState() {
	b := g.board
	canMove1 := g.normalMoves[0] > 0 || (g.freeCells > 0 && b.Resources[0].NumPieces[cellFlag] > 0)
	canMove2 := g.normalMoves[1] > 0 || (g.freeCells > 0 && b.Resources[1].NumPieces[cellFlag] > 0)
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
	if g.freeCells == 0 || !(canMove1 || canMove2) ||
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
	b := g.board
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
				if nb.nextVal[pIdx] > 0 {
					g.normalMoves[pIdx]--
				}
				nb.Blocked |= 1 << pIdx
				nb.nextVal[pIdx] = -1
			} else if nb.nextVal[pIdx] == 0 {
				g.normalMoves[pIdx]++
				nb.nextVal[pIdx] = f.Value + 1
			} else if nb.nextVal[pIdx] > f.Value+1 {
				nb.nextVal[pIdx] = f.Value + 1
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
	b := g.board
	turn := b.Turn
	pIdx := turn - 1
	if m.playerNum != turn || m.move != b.Move {
		// Only allow moves by players whose turn it is.
		return false
	}
	if !b.valid(idx{m.row, m.col}) {
		// Invalid move request.
		return false
	}
	f := &b.Fields[m.row][m.col]
	if f.occupied() {
		return false
	}
	if b.Resources[pIdx].NumPieces[m.cellType] == 0 {
		// No pieces left of requested type
		return false
	}
	if m.cellType == cellNormal {
		val := f.nextVal[pIdx]
		if val <= 0 {
			return false
		}
		f.Owner = turn
		f.Type = cellNormal
		f.lifetime = g.lifetime(cellNormal)
		f.Hidden = false
		f.Value = val
		b.Score[pIdx] += val
		g.freeCells--
		g.normalMoves[pIdx]--
		if f.nextVal[1-pIdx] > 0 {
			g.normalMoves[1-pIdx]--
		}
		g.updateNeighborCells(m.row, m.col)
	} else if m.cellType == cellFlag {
		// A flag can be placed on any free cell. It does not add to the score.
		f.Owner = turn
		f.Type = cellFlag
		f.lifetime = g.lifetime(cellFlag)
		f.Hidden = false
		f.Value = 0
		b.Resources[turn-1].NumPieces[cellFlag]--
		g.freeCells--
		// Adjust available normal moves for both players if f was available to them.
		if f.nextVal[pIdx] > 0 {
			g.normalMoves[pIdx]--
		}
		if f.nextVal[1-pIdx] > 0 {
			g.normalMoves[1-pIdx]--
		}
		g.updateNeighborCells(m.row, m.col)
	} else {
		// Invalid piece. Should be caught by resource check already, so never reached.
		return false
	}
	b.Turn = 3 - b.Turn // Usually it's the other player's turn. If not, recomputeState will fix that.
	b.Move++
	g.recomputeState()
	return true
}

func (g *GameEngineFlagz) Board() *Board { return g.board }
func (g *GameEngineFlagz) Clone(s rand.Source) SinglePlayerGameEngine {
	return &GameEngineFlagz{
		board:       g.board.copy(),
		rnd:         rand.New(s),
		freeCells:   g.freeCells,
		normalMoves: g.normalMoves,
	}
}

func (g *GameEngineFlagz) IsDone() bool {
	return g.board.State == Finished
}

func (g *GameEngineFlagz) Winner() (playerNum int) {
	if !g.IsDone() {
		return 0
	}
	return scoreBasedSingleWinner(g.board.Score)
}

// Suggests a move for the player whose turn it is.
// Uses a random strategy. Probably not very smart.
func (g *GameEngineFlagz) RandomMove() (GameEngineMove, error) {
	if g.board.State != Running {
		return GameEngineMove{}, fmt.Errorf("game is not running")
	}
	b := g.board
	pIdx := b.Turn - 1
	nMoves := g.normalMoves[pIdx]
	flagsLeft := b.Resources[pIdx].NumPieces[cellFlag] > 0
	pickFlag := false
	if g.freeCells > 0 && flagsLeft && (nMoves == 0 || g.rnd.Float64() <= float64(1)/float64(nMoves+1)) {
		pickFlag = true
	}
	if pickFlag {
		nthFlag := g.rnd.Intn(g.freeCells)
		n := 0
		for r := 0; r < len(b.Fields); r++ {
			for c := 0; c < len(b.Fields[r]); c++ {
				if !b.Fields[r][c].occupied() {
					if n == nthFlag {
						return GameEngineMove{
							playerNum: b.Turn,
							move:      b.Move,
							row:       r,
							col:       c,
							cellType:  cellFlag,
						}, nil
					}
					n++
				}
			}
		}
	} else {
		// Pick a normal move
		nthMove := g.rnd.Intn(nMoves)
		n := 0
		for r := 0; r < len(b.Fields); r++ {
			for c := 0; c < len(b.Fields[r]); c++ {
				f := &b.Fields[r][c]
				if !f.occupied() && f.isAvail(b.Turn) {
					if n == nthMove {
						return GameEngineMove{
							playerNum: b.Turn,
							move:      b.Move,
							row:       r,
							col:       c,
							cellType:  cellNormal,
						}, nil
					}
					n++
				}
			}
		}
	}
	panic("no legal move found")
}
