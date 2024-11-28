package hexz

import (
	"fmt"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/internal/hlog"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

const (
	numFieldsFirstRow = 10
	numBoardRows      = 11
)

type Board struct {
	Turn         int
	Move         int
	LastRevealed int       // Move at which fields were last revealed
	FlatFields   []Field   // The 1-d array backing the "2d" Fields.
	Fields       [][]Field // The board's fields. Subslices of FlatFields.
	Score        []int     // Depending on the number of players, 1 or 2 elements.
	Resources    []ResourceInfo
	State        GameState
}

type GameEngine interface {
	Reset()
	NumPlayers() int
	ValidCellTypes() []CellType
	MakeMove(move GameEngineMove) bool
	MakeMoveError(move GameEngineMove) error
	Board() *Board
	IsDone() bool
	Winner() (playerNum int) // Results are only meaningful if IsDone() is true. 0 for draw.
	GameType() api.GameType
	// Encodes the current state of the game engine.
	Encode() (*pb.GameEngineState, error)
	// Sets this game engine into the state defined by the given encoded state.
	Decode(s *pb.GameEngineState) error
}

const (
	gameTypeClassic  api.GameType = "Classic"
	gameTypeFlagz    api.GameType = "Flagz"
	gameTypeFreeform api.GameType = "Freeform"
)

var (
	allGameTypes = map[api.GameType]bool{
		gameTypeClassic:  true,
		gameTypeFlagz:    true,
		gameTypeFreeform: true,
	}
)

func validGameType(gameType string) bool {
	return allGameTypes[api.GameType(gameType)]
}

func supportsSinglePlayer(t api.GameType) bool {
	return t == gameTypeFlagz
}

// Each player has a different view of the board. In particular, player A
// should not see the hidden moves of player B. To not give cheaters a chance,
// we should never send the hidden moves out to other players at all
// (i.e., we shouldn't just rely on our UI which would not show them; cheaters
// can easily intercept the http response.)
func (b *Board) ViewFor(playerNum int) *BoardView {
	score := make([]int, len(b.Score))
	copy(score, b.Score)
	resources := make([]ResourceInfo, len(b.Resources))
	for i, r := range b.Resources {
		resources[i] = ResourceInfo{
			NumPieces: r.NumPieces,
		}
	}
	flat, fields := copyFields(b)
	// Hide other player's hidden fields.
	if playerNum > 0 {
		for i := 0; i < len(flat); i++ {
			f := &flat[i]
			if f.Owner != playerNum && f.Hidden {
				*f = Field{}
			}
		}
	}
	return &BoardView{
		Turn:      b.Turn,
		Move:      b.Move,
		Score:     score,
		Resources: resources,
		State:     b.State,
		Fields:    fields,
	}
}

func (b *Board) Copy() *Board {
	var score []int
	if b.Score != nil {
		score = make([]int, len(b.Score))
		copy(score, b.Score)
	}
	var resources []ResourceInfo
	if b.Resources != nil {
		resources = make([]ResourceInfo, len(b.Resources))
		copy(resources, b.Resources)
	}
	flat, fields := copyFields(b)
	return &Board{
		Turn:         b.Turn,
		Move:         b.Move,
		Score:        score,
		Resources:    resources,
		State:        b.State,
		Fields:       fields,
		FlatFields:   flat,
		LastRevealed: b.LastRevealed,
	}
}

func (b *Board) copyFrom(src *Board) {
	if len(b.Score) != len(src.Score) {
		b.Score = make([]int, len(src.Score))
	}
	copy(b.Score, src.Score)
	if len(b.Resources) != len(src.Resources) {
		b.Resources = make([]ResourceInfo, len(src.Resources))
	}
	copy(b.Resources, src.Resources)
	b.Turn = src.Turn
	b.Move = src.Move
	b.State = src.State
	b.LastRevealed = src.LastRevealed
	if len(b.FlatFields) != len(src.FlatFields) {
		panic("cannot copy board: field length mismatch")
	}
	copy(b.FlatFields, src.FlatFields)
}

func (f *Field) occupied() bool {
	return f.Type != cellNormal || f.Owner > 0
}

func (f *Field) isAvail(playerNum int) bool {
	return f.NextVal[playerNum-1] > 0
}

type GameEngineMove struct {
	PlayerNum int
	Move      int
	Row       int
	Col       int
	CellType  CellType
}

func (m *GameEngineMove) String() string {
	return fmt.Sprintf("P%d#%d (%d,%d/%d)", m.PlayerNum, m.Move, m.Row, m.Col, m.CellType)
}

func (m *GameEngineMove) Proto() *pb.GameEngineMove {
	return &pb.GameEngineMove{
		PlayerNum: int32(m.PlayerNum),
		Move:      int32(m.Move),
		Row:       int32(m.Row),
		Col:       int32(m.Col),
		CellType:  pb.Field_CellType(m.CellType),
	}
}

func (m *GameEngineMove) FromProto(pm *pb.GameEngineMove) {
	m.PlayerNum = int(pm.PlayerNum)
	m.Move = int(pm.Move)
	m.Row = int(pm.Row)
	m.Col = int(pm.Col)
	m.CellType = CellType(pm.CellType)
}

// Dispatches on the gameType to create a corresponding GameEngine.
// The returned GameEngine is initialized and ready to play.
func NewGameEngine(gameType api.GameType) GameEngine {
	var ge GameEngine
	switch gameType {
	case gameTypeClassic:
		ge = NewGameEngineClassic()
	case gameTypeFlagz:
		ge = NewGameEngineFlagz()
	case gameTypeFreeform:
		ge = NewGameEngineFreeform()
	default:
		panic("Unconsidered game type: " + gameType)
	}
	return ge
}

func DecodeGameEngine(s *pb.GameEngineState) (GameEngine, error) {
	var g GameEngine
	switch s.State.(type) {
	case *pb.GameEngineState_Flagz:
		g = NewGameEngineFlagz()
	case *pb.GameEngineState_Classic:
		g = NewGameEngineClassic()
	case *pb.GameEngineState_Freeform:
		g = NewGameEngineFreeform()
	default:
		panic("unhandled game type")
	}
	if err := g.Decode(s); err != nil {
		return nil, err
	}
	return g, nil
}

var (
	errMakeMove      = fmt.Errorf("could not make the move")
	errUndoRedoEmpty = fmt.Errorf("undo/redo buffer is empty")
)

// GameRepr is the "handle" to a game that HTTP handlers should use.
// It contains a proto representation of the game, as well as a
// (lazily initialized) GameEngine representation.
type GameRepr struct {
	state  *pb.GameState
	engine GameEngine
}

func NewGameRepr(state *pb.GameState) *GameRepr {
	return &GameRepr{
		state: state,
	}
}

func (g *GameRepr) Engine() GameEngine {
	if g.engine == nil {
		var err error
		g.engine, err = DecodeGameEngine(g.state.EngineState)
		if err != nil {
			hlog.Fatalf("Cannot decode game engine: %v", err)
		}
	}
	return g.engine
}

func (g *GameRepr) State() *pb.GameState {
	return g.state
}

func (g *GameRepr) isCPUTurn() bool {
	if g.Engine().IsDone() {
		return false
	}
	turn := g.Engine().Board().Turn
	mode := g.State().GameInfo.CpuPlayer
	return turn == 2 && (mode == pb.CPUPlayerMode_EMBEDDED_CPU ||
		mode == pb.CPUPlayerMode_REMOTE_CPU)
}

// Returns the most recent move made in the game, or nil if no move has been made yet.
func (g *GameRepr) LastMove() *pb.GameEngineMove {
	moves := g.State().GetUndoRedoState().GetMoves()
	if len(moves) == 0 {
		return nil
	}
	return moves[len(moves)-1]
}

func (g *GameRepr) PlayerNames() []string {
	names := make([]string, len(g.state.Players))
	for i, p := range g.state.Players {
		names[i] = p.Name
	}
	return names
}

func (g *GameRepr) PlayerNum(playerId string) int {
	for i, p := range g.state.Players {
		if p.Id == playerId {
			return i + 1
		}
	}
	return 0 // spectator
}

func (g *GameRepr) AddPlayer(p *pb.Player) {
	g.State().Players = append(g.State().Players, p)
}

func (g *GameRepr) AllPlayersJoined() bool {
	return len(g.State().Players) == g.Engine().NumPlayers()
}

func (g *GameRepr) Reset() error {
	g.Engine().Reset()
	var err error
	g.state.EngineState, err = g.Engine().Encode()
	return err
}

func (g *GameRepr) MakeMove(move GameEngineMove) error {
	if err := g.Engine().MakeMoveError(move); err != nil {
		return err
	}
	enc, err := g.Engine().Encode()
	if err != nil {
		return err
	}
	g.state.EngineState = enc
	g.state.UndoRedoState.Moves = append(g.state.UndoRedoState.Moves, move.Proto())
	return nil
}

// func (g *GameRepr) debugPrintUndoRedoStack(label string) {
// 	t := g.state.UndoRedoState.InitialState
// 	g.state.UndoRedoState.InitialState = nil
// 	fmt.Print(label, prototext.Format(g.state.UndoRedoState))
// 	g.state.UndoRedoState.InitialState = t
// }

func (g *GameRepr) Undo() error {
	s := g.State()
	if len(s.GetUndoRedoState().GetMoves()) == 0 {
		return errUndoRedoEmpty
	}
	ge, err := DecodeGameEngine(s.GetUndoRedoState().GetInitialState())
	if err != nil {
		return err
	}
	moves := s.UndoRedoState.Moves
	// Push last move onto the redo stack
	s.UndoRedoState.Moves = moves[:len(moves)-1]
	s.UndoRedoState.RedoMoves = append(s.UndoRedoState.RedoMoves, moves[len(moves)-1])
	// Make all moves from initial state to reach previous state.
	for i, move := range s.UndoRedoState.Moves {
		geMove := GameEngineMove{}
		geMove.FromProto(move)
		if err := ge.MakeMoveError(geMove); err != nil {
			return fmt.Errorf("undo: error making move #%d: %v: %v", i, move, err)
		}
	}
	enc, err := ge.Encode()
	if err != nil {
		return fmt.Errorf("failed to encode GameEngine: %v", err)
	}
	g.engine = ge
	s.EngineState = enc

	return nil
}

func (g *GameRepr) Redo() error {
	s := g.State()
	redoMoves := s.GetUndoRedoState().GetRedoMoves()
	if len(redoMoves) == 0 {
		return errUndoRedoEmpty
	}
	move := redoMoves[len(redoMoves)-1]

	// Remove move from redo stack. It will get pushed onto the undo stack in MakeMove.
	s.UndoRedoState.RedoMoves = redoMoves[:len(redoMoves)-1]
	geMove := GameEngineMove{}
	geMove.FromProto(move)
	if err := g.MakeMove(geMove); err != nil {
		return fmt.Errorf("redo: error making move: %v: %v", move, err)
	}

	return nil
}

// Creates a new, empty 2d field array.
func makeFields() ([]Field, [][]Field) {
	const numFields = numFieldsFirstRow*((numBoardRows+1)/2) + (numFieldsFirstRow-1)*(numBoardRows/2)
	flat := make([]Field, numFields)
	fields := make([][]Field, numBoardRows)
	start := 0
	for i := 0; i < len(fields); i++ {
		end := start + numFieldsFirstRow - i%2
		fields[i] = flat[start:end]
		start = end
	}
	return flat, fields
}

// Creates a deep copy of the board's fields.
// Useful to send slightly modified variations of the same board to different
// players.
func copyFields(b *Board) ([]Field, [][]Field) {
	flat := make([]Field, len(b.FlatFields))
	copy(flat, b.FlatFields)
	fields := make([][]Field, len(b.Fields))
	start := 0
	for i, fs := range b.Fields {
		end := start + len(fs)
		fields[i] = flat[start:end]
		start = end
	}
	return flat, fields
}

// Creates a new, empty board with nil score and nil resources.
func NewBoard() *Board {
	flatFields, fields := makeFields()
	return &Board{
		Turn:       1, // Player 1 begins
		FlatFields: flatFields,
		Fields:     fields,
		State:      Initial,
	}
}

type idx struct {
	r, c int
}

func (b *Board) valid(x idx) bool {
	return x.r >= 0 && x.r < len(b.Fields) && x.c >= 0 && x.c < len(b.Fields[x.r])
}

// Populates ns with valid indices of all neighbor cells. Returns the number of neighbor cells.
// ns must have enough capacity to hold all neighbors. You should pass in a [6]idx slice.
func (b *Board) neighbors(x idx, ns []idx) int {
	shift := x.r & 1 // Depending on the row, neighbors below and above are shifted.
	k := 0
	ns[k] = idx{x.r, x.c + 1}
	if b.valid(ns[k]) {
		k++
	}
	ns[k] = idx{x.r - 1, x.c + shift}
	if b.valid(ns[k]) {
		k++
	}
	ns[k] = idx{x.r - 1, x.c - 1 + shift}
	if b.valid(ns[k]) {
		k++
	}
	ns[k] = idx{x.r, x.c - 1}
	if b.valid(ns[k]) {
		k++
	}
	ns[k] = idx{x.r + 1, x.c - 1 + shift}
	if b.valid(ns[k]) {
		k++
	}
	ns[k] = idx{x.r + 1, x.c + shift}
	if b.valid(ns[k]) {
		k++
	}
	return k
}

func scoreBasedSingleWinner(score []int) (playerNum int) {
	maxIdx := -1
	maxScore := -1
	uniq := false
	for i, s := range score {
		if s > maxScore {
			maxScore = s
			maxIdx = i
			uniq = true
		} else if s == maxScore {
			uniq = false
		}
	}
	if uniq {
		return maxIdx + 1 // Return as playerNum
	}
	return 0
}

func (b *Board) Proto() *pb.Board {
	var state pb.Board_GameState
	switch b.State {
	case Initial:
		state = pb.Board_INITIAL
	case Running:
		state = pb.Board_RUNNING
	case Finished:
		state = pb.Board_FINISHED
	default:
		panic(fmt.Sprintf("unconsidered game state: %v", b.State))
	}
	bp := &pb.Board{
		Turn:         int32(b.Turn),
		Move:         int32(b.Move),
		LastRevealed: int32(b.LastRevealed),
		FlatFields:   make([]*pb.Field, len(b.FlatFields)),
		Score:        make([]int32, len(b.Score)),
		Resources:    make([]*pb.ResourceInfo, len(b.Resources)),
		State:        state,
	}
	for i, f := range b.FlatFields {
		bp.FlatFields[i] = &pb.Field{
			Type:     pb.Field_CellType(f.Type),
			Owner:    int32(f.Owner),
			Hidden:   f.Hidden,
			Value:    int32(f.Value),
			Blocked:  int32(f.Blocked),
			Lifetime: int32(f.Lifetime),
			NextVal:  []int32{int32(f.NextVal[0]), int32(f.NextVal[1])},
		}
	}
	for i, s := range b.Score {
		bp.Score[i] = int32(s)
	}
	for i, r := range b.Resources {
		bp.Resources[i] = &pb.ResourceInfo{
			NumPieces: make([]int32, len(r.NumPieces)),
		}
		for j, n := range r.NumPieces {
			bp.Resources[i].NumPieces[j] = int32(n)
		}
	}
	return bp
}

func (b *Board) DecodeProto(bp *pb.Board) error {
	b.Turn = int(bp.Turn)
	b.Move = int(bp.Move)
	b.LastRevealed = int(bp.LastRevealed)
	switch bp.State {
	case pb.Board_INITIAL:
		b.State = Initial
	case pb.Board_RUNNING:
		b.State = Running
	case pb.Board_FINISHED:
		b.State = Finished
	default:
		return fmt.Errorf("cannot decode board: unknown game state: %v", bp.State)
	}
	if len(b.FlatFields) != len(bp.FlatFields) {
		return fmt.Errorf("cannot decode board: field length mismatch: want %d, got %d", len(b.FlatFields), len(bp.FlatFields))
	}
	for i, f := range bp.FlatFields {
		b.FlatFields[i] = Field{
			Type:     CellType(f.Type),
			Owner:    int(f.Owner),
			Hidden:   f.Hidden,
			Value:    int(f.Value),
			Blocked:  uint8(f.Blocked),
			Lifetime: int(f.Lifetime),
			NextVal:  [2]int{int(f.NextVal[0]), int(f.NextVal[1])},
		}
	}
	// Nice: the Fields automatically point to the right sections in FlatFields,
	// since we don't reallocate that array.
	if len(b.Score) != len(bp.Score) {
		return fmt.Errorf("cannot decode board: score length mismatch: want %d, got %d", len(b.Score), len(bp.Score))
	}
	for i, s := range bp.Score {
		b.Score[i] = int(s)
	}
	if len(b.Resources) != len(bp.Resources) {
		return fmt.Errorf("cannot decode board: resources length mismatch: want %d, got %d", len(b.Resources), len(bp.Resources))
	}
	for i, r := range bp.Resources {
		if len(b.Resources[i].NumPieces) != len(r.NumPieces) {
			return fmt.Errorf("cannot decode board: resources[%d] length mismatch: want %d, got %d", i, len(b.Resources[i].NumPieces), len(r.NumPieces))
		}
		for j, n := range r.NumPieces {
			b.Resources[i].NumPieces[j] = int(n)
		}
	}
	return nil
}
