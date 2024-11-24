package hexz

// func (m *GameMaster) processControlEventValidMoves(e ControlEventValidMoves) {
// 	defer close(e.reply)
// 	if engine, ok := m.gameEngine.(*GameEngineFlagz); ok {
// 		validMoves := engine.ValidMoves()
// 		moves := make([]*MoveRequest, len(validMoves))
// 		for i, m := range validMoves {
// 			moves[i] = &MoveRequest{
// 				Move: m.Move,
// 				Row:  m.Row,
// 				Col:  m.Col,
// 				Type: m.CellType,
// 			}
// 		}
// 		e.reply <- moves
// 	}
// }
