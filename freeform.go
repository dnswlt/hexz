package hexz

//
// The freeform single-player hexz game.
//

type GameEngineFreeform struct {
	board *Board
}

func (g *GameEngineFreeform) GameType() GameType { return gameTypeFreeform }
func (g *GameEngineFreeform) Board() *Board      { return g.board }

func (g *GameEngineFreeform) Init() {
	g.board = InitBoard(g)
}
func (g *GameEngineFreeform) Start() {
	g.board.State = Running
}
func (g *GameEngineFreeform) NumPlayers() int {
	return 1
}

func (g *GameEngineFreeform) InitialResources() ResourceInfo {
	return ResourceInfo{
		NumPieces: map[CellType]int{
			cellNormal: -1, // unlimited
			cellFire:   -1,
			cellFlag:   -1,
			cellPest:   -1,
			cellDeath:  -1,
		},
	}
}

func (g *GameEngineFreeform) Reset()       { g.Init() }
func (g *GameEngineFreeform) IsDone() bool { return false }
func (g *GameEngineFreeform) Winner() (playerNum int) {
	return 0 // No one ever wins here.
}

func (g *GameEngineFreeform) MakeMove(m GameEngineMove) bool {
	board := g.board
	if !board.valid(idx{m.row, m.col}) {
		// Invalid move request.
		return false
	}
	board.Move++
	f := &board.Fields[m.row][m.col]
	f.Owner = board.Turn
	f.Type = m.cellType
	board.Turn++
	if board.Turn > 2 {
		board.Turn = 1
	}
	f.Value = 1
	return true
}
