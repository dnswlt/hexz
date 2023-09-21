package hexz

import (
	"fmt"

	"github.com/dnswlt/hexz/hexzpb"
	pb "github.com/dnswlt/hexz/hexzpb"
)

//
// The freeform single-player hexz game.
//

type GameEngineFreeform struct {
	board *Board
}

func NewGameEngineFreeform() *GameEngineFreeform {
	g := &GameEngineFreeform{}
	g.Init()
	return g
}

func (g *GameEngineFreeform) GameType() GameType { return gameTypeFreeform }
func (g *GameEngineFreeform) Board() *Board      { return g.board }

func (g *GameEngineFreeform) Init() {
	flatFields, fields := makeFields()
	b := &Board{
		Turn:       1, // Player 1 begins
		FlatFields: flatFields,
		Fields:     fields,
		State:      Initial,
	}
	numPlayers := g.NumPlayers()
	b.Score = make([]int, numPlayers)
	b.Resources = make([]ResourceInfo, numPlayers)
	for i := 0; i < numPlayers; i++ {
		b.Resources[i] = g.InitialResources()
	}
	g.board = b
	g.board.State = Running
}

func (g *GameEngineFreeform) NumPlayers() int {
	return 1
}
func (g *GameEngineFreeform) MoveHistory() []GameEngineMove {
	panic("Not implemented")
}

func (g *GameEngineFreeform) ValidCellTypes() []CellType {
	r := make([]CellType, 0, cellTypeLen)
	for i, v := range g.InitialResources().NumPieces {
		if v != 0 {
			r = append(r, CellType(i))
		}
	}
	return r
}

func (g *GameEngineFreeform) InitialResources() ResourceInfo {
	var ps [cellTypeLen]int
	ps[cellNormal] = -1 // unlimited
	ps[cellFire] = -1
	ps[cellFlag] = -1
	ps[cellPest] = -1
	ps[cellDeath] = -1
	return ResourceInfo{
		NumPieces: ps}
}

func (g *GameEngineFreeform) Reset()       { g.Init() }
func (g *GameEngineFreeform) IsDone() bool { return false }
func (g *GameEngineFreeform) Winner() (playerNum int) {
	return 0 // No one ever wins here.
}

func (g *GameEngineFreeform) MakeMove(m GameEngineMove) bool {
	board := g.board
	if !board.valid(idx{m.Row, m.Col}) {
		// Invalid move request.
		return false
	}
	board.Move++
	f := &board.Fields[m.Row][m.Col]
	f.Owner = board.Turn
	f.Type = m.CellType
	if m.CellType == cellNormal {
		f.Value = 1
	}
	return true
}

func (g *GameEngineFreeform) Encode() (*hexzpb.GameEngineState, error) {
	freeform := &pb.GameEngineFreeformState{
		Board: g.Board().Proto(),
	}
	s := &pb.GameEngineState{
		State: &pb.GameEngineState_Freeform{
			Freeform: freeform,
		},
	}
	return s, nil
}

func (g *GameEngineFreeform) Decode(s *hexzpb.GameEngineState) error {
	if s.GetFreeform() == nil {
		return fmt.Errorf("invalid game state: missing freeform")
	}
	freeform := s.GetFreeform()
	if err := g.board.DecodeProto(freeform.Board); err != nil {
		return err
	}
	return nil
}
