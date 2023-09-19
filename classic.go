package hexz

import (
	"fmt"

	"github.com/dnswlt/hexz/hexzpb"
)

//
// The "classic" hexz game
//

type GameEngineClassic struct {
	board *Board
}

func (g *GameEngineClassic) GameType() GameType { return gameTypeClassic }
func (g *GameEngineClassic) Board() *Board      { return g.board }
func (g *GameEngineClassic) Init() {
	b := NewBoard()
	numPlayers := g.NumPlayers()
	b.Score = make([]int, numPlayers)
	b.Resources = make([]ResourceInfo, numPlayers)
	for i := 0; i < numPlayers; i++ {
		b.Resources[i] = g.InitialResources()
	}
	g.board = b
	g.board.State = Running
}

func (g *GameEngineClassic) Reset() {
	g.Init()
}

func (g *GameEngineClassic) NumPlayers() int {
	return 2
}

func (g *GameEngineClassic) MoveHistory() []GameEngineMove {
	panic("Not implemented")
}

func (g *GameEngineClassic) ValidCellTypes() []CellType {
	r := make([]CellType, 0, cellTypeLen)
	for i, v := range g.InitialResources().NumPieces {
		if v != 0 {
			r = append(r, CellType(i))
		}
	}
	return r
}

func (g *GameEngineClassic) InitialResources() ResourceInfo {
	var ps [cellTypeLen]int
	ps[cellNormal] = -1 // unlimited
	ps[cellFire] = 1
	ps[cellFlag] = 1
	ps[cellPest] = 1
	ps[cellDeath] = 1
	return ResourceInfo{
		NumPieces: ps}
}

func (g *GameEngineClassic) IsDone() bool { return g.board.State == Finished }

func (g *GameEngineClassic) Winner() (playerNum int) {
	if !g.IsDone() {
		return 0
	}
	return scoreBasedSingleWinner(g.board.Score)
}

func (g *GameEngineClassic) recomputeScoreAndState() {
	b := g.board
	s := []int{0, 0}
	hasOpenCells := false
	for r := range b.Fields {
		for c := range b.Fields[r] {
			if !b.Fields[r][c].occupied() {
				// Don't finish the game until all cells are owned and not hidden, or dead.
				hasOpenCells = true
				break
			}
		}
	}
	if !hasOpenCells {
		// No more inconclusive cells: game is finished. Reveal hidden moves before
		// computing score.
		g.revealHiddenMoves()
		b.State = Finished
	}
	for r := range b.Fields {
		for c := range b.Fields[r] {
			fld := &b.Fields[r][c]
			if fld.Owner > 0 && !fld.Hidden {
				s[fld.Owner-1]++
			}
		}
	}
	b.Score = s
}

func (g *GameEngineClassic) floodFill(x idx, cb func(idx) bool) {
	b := g.board
	var ns [6]idx
	if !cb(x) {
		return
	}
	n := b.neighbors(x, ns[:])
	for i := 0; i < n; i++ {
		g.floodFill(ns[i], cb)
	}
}

func (g *GameEngineClassic) lifetime(c CellType) int {
	switch c {
	case cellFire, cellDead, cellDeath:
		return 1
	case cellPest:
		return 3
	}
	return -1 // Live forever
}

func (g *GameEngineClassic) occupyFields(playerNum, r, c int, ct CellType) int {
	b := g.board
	// Create a board-shaped 2d array that indicates which neighboring cell of (i, j)
	// it shares the free area with.
	// Then find the smallest of these areas and occupy every free cell in it.
	ms := make([][]int, len(b.Fields))
	for k := 0; k < len(ms); k++ {
		ms[k] = make([]int, len(b.Fields[k]))
		for m := 0; m < len(ms[k]); m++ {
			ms[k][m] = -1
		}
	}
	f := &b.Fields[r][c]
	f.Owner = playerNum
	f.Type = ct
	f.Hidden = true
	f.Lifetime = g.lifetime(ct)
	var areas [6]struct {
		size      int    // Number of free cells in the area
		flags     [2]int // Number of flags along the boundary
		deadCells int    // Number of dead cells along the boundary
	}
	// If the current move sets a flag, this flag counts in all directions.
	if ct == cellFlag {
		for i := 0; i < len(areas); i++ {
			areas[i].flags[playerNum-1]++
		}
	}
	// Flood fill starting from each of (r, c)'s neighbors.
	var ns [6]idx
	n := b.neighbors(idx{r, c}, ns[:])
	for k := 0; k < n; k++ {
		if ms[ns[k].r][ns[k].c] != -1 {
			// k's area is the same area as k-c for some c.
			continue
		}
		g.floodFill(ns[k], func(x idx) bool {
			if ms[x.r][x.c] == k {
				// Already seen in this iteration.
				return false
			}
			f := &b.Fields[x.r][x.c]
			if f.occupied() {
				// Occupied fields act as boundaries.
				if f.Type == cellFlag {
					areas[k].flags[f.Owner-1]++
				} else if f.Type == cellDead {
					areas[k].deadCells++
				}
				// Mark as seen to avoid revisiting boundaries.
				ms[x.r][x.c] = k
				return false
			}
			// Mark as seen and update area size.
			ms[x.r][x.c] = k
			areas[k].size++
			return true
		})
	}
	// If there is more than one area, we know we introduced a split, since the areas
	// would have been connected by the previously free cell (r, c).
	numOccupiedFields := 1
	numAreas := 0
	minK := -1
	// Count the number of separated areas and find the smallest one.
	for k := 0; k < n; k++ {
		if areas[k].size > 0 && areas[k].deadCells == 0 {
			numAreas++
			if minK == -1 || areas[minK].size > areas[k].size {
				minK = k
			}
		}
	}
	if numAreas > 1 {
		// Now assign fields to player with most flags, or to current player on a tie.
		occupator := playerNum
		if areas[minK].flags[2-playerNum] > areas[minK].flags[playerNum-1] {
			occupator = 3 - playerNum // The other player has more flags
		}
		for r := 0; r < len(b.Fields); r++ {
			for c := 0; c < len(b.Fields[r]); c++ {
				f := &b.Fields[r][c]
				if ms[r][c] == minK && !f.occupied() {
					numOccupiedFields++
					f.Owner = occupator
					f.Lifetime = g.lifetime(cellNormal)
				}
			}
		}
	}
	return numOccupiedFields
}

func (g *GameEngineClassic) applyFireEffect(r, c int) {
	b := g.board
	var ns [6]idx
	n := b.neighbors(idx{r, c}, ns[:])
	for i := 0; i < n; i++ {
		f := &b.Fields[ns[i].r][ns[i].c]
		f.Owner = 0
		f.Type = cellDead
		f.Hidden = false
		f.Lifetime = g.lifetime(cellDead)
	}
}

func (g *GameEngineClassic) applyPestEffect() {
	b := g.board
	var ns [6]idx
	for r := 0; r < len(b.Fields); r++ {
		for c := 0; c < len(b.Fields[r]); c++ {
			// A pest cell does not propagate in its first round.
			if b.Fields[r][c].Type == cellPest && b.Fields[r][c].Lifetime < g.lifetime(cellPest) {
				n := b.neighbors(idx{r, c}, ns[:])
				for i := 0; i < n; i++ {
					f := &b.Fields[ns[i].r][ns[i].c]
					if f.Owner > 0 && f.Owner != b.Fields[r][c].Owner && f.Type == cellNormal {
						// Pest only affects the opponent's normal cells.
						f.Owner = b.Fields[r][c].Owner
						f.Type = cellPest
						f.Lifetime = g.lifetime(cellPest)
					}
				}
			}
		}
	}
}

func (g *GameEngineClassic) isPlayerPiece(c CellType) bool {
	return c == cellNormal || c >= cellFire && c <= cellDeath
}

func (g *GameEngineClassic) revealHiddenMoves() {
	for r := 0; r < len(g.board.Fields); r++ {
		for c := 0; c < len(g.board.Fields[r]); c++ {
			g.board.Fields[r][c].Hidden = false
		}
	}
}

func (g *GameEngineClassic) MakeMove(m GameEngineMove) bool {
	board := g.board
	turn := board.Turn
	if m.PlayerNum != turn || m.Move != board.Move {
		// Only allow moves by players whose turn it is.
		return false
	}
	if !board.valid(idx{m.Row, m.Col}) || !g.isPlayerPiece(m.CellType) {
		// Invalid move request.
		return false
	}
	if m.CellType != cellNormal && board.Resources[turn-1].NumPieces[m.CellType] == 0 {
		// No pieces left of requested type
		return false
	}
	numOccupiedFields := 0
	revealBoard := m.CellType != cellNormal && m.CellType != cellFlag
	if board.Fields[m.Row][m.Col].occupied() {
		if board.Fields[m.Row][m.Col].Hidden && board.Fields[m.Row][m.Col].Owner == (3-turn) {
			// Conflicting hidden moves. Leads to dead cell.
			board.Move++
			f := &board.Fields[m.Row][m.Col]
			f.Owner = 0
			f.Type = cellDead
			f.Lifetime = g.lifetime(cellDead)
			revealBoard = true
		} else if m.CellType == cellDeath {
			// Death cell can be placed anywhere and will "kill" whatever was there before.
			f := &board.Fields[m.Row][m.Col]
			f.Owner = turn
			f.Type = cellDeath
			f.Hidden = false
			f.Lifetime = g.lifetime(cellDeath)
		} else {
			// Cannot make move on already occupied field.
			return false
		}
	} else {
		// Free cell: occupy it.
		board.Move++
		f := &board.Fields[m.Row][m.Col]
		if m.CellType == cellFire {
			// Fire cells take effect immediately.
			f.Owner = turn
			f.Type = m.CellType
			f.Lifetime = g.lifetime(cellFire)
			g.applyFireEffect(m.Row, m.Col)
		} else {
			numOccupiedFields = g.occupyFields(turn, m.Row, m.Col, m.CellType)
		}
	}
	if m.CellType != cellNormal {
		board.Resources[turn-1].NumPieces[m.CellType]--
	}
	// Update turn.
	board.Turn++
	if board.Turn > 2 {
		board.Turn = 1
	}
	if numOccupiedFields > 1 || board.Move-board.LastRevealed == 4 || revealBoard {
		// Reveal hidden moves.
		g.revealHiddenMoves()
		g.applyPestEffect()
		// Clean up old special cells.
		for r := 0; r < len(board.Fields); r++ {
			for c := 0; c < len(board.Fields[r]); c++ {
				f := &board.Fields[r][c]
				if f.occupied() && f.Lifetime == 0 {
					f.Owner = 0
					f.Hidden = false
					f.Type = cellNormal
					f.Lifetime = g.lifetime(cellNormal)
				}
				if f.Lifetime > 0 {
					f.Lifetime--
				}
			}
		}

		board.LastRevealed = board.Move
	}
	g.recomputeScoreAndState()
	return true
}

func (g *GameEngineClassic) Encode() (*hexzpb.GameEngineState, error) {
	return nil, fmt.Errorf("not implemented")
}

func (g *GameEngineClassic) Decode(*hexzpb.GameEngineState) error {
	return fmt.Errorf("not implemented")
}
