package hexz

import (
	"fmt"
	"math/rand"
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
		numPieces := make(map[CellType]int)
		for k, v := range r.NumPieces {
			numPieces[k] = v
		}
		resources[i] = ResourceInfo{
			NumPieces: numPieces,
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

func (b *Board) copy() *Board {
	score := make([]int, len(b.Score))
	copy(score, b.Score)
	resources := make([]ResourceInfo, len(b.Resources))
	for i, r := range b.Resources {
		numPieces := make(map[CellType]int)
		for k, v := range r.NumPieces {
			numPieces[k] = v
		}
		resources[i] = ResourceInfo{
			NumPieces: numPieces,
		}
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

func (f *Field) occupied() bool {
	return f.Type != cellNormal || f.Owner > 0
}

func (f *Field) isAvail(playerNum int) bool {
	return f.nextVal[playerNum-1] > 0
}

type GameType string

const (
	gameTypeClassic  GameType = "Classic"
	gameTypeFlagz    GameType = "Flagz"
	gameTypeFreeform GameType = "Freeform"
)

func validGameType(gameType string) bool {
	allGameTypes := map[GameType]bool{
		gameTypeClassic:  true,
		gameTypeFlagz:    true,
		gameTypeFreeform: true,
	}
	return allGameTypes[GameType(gameType)]
}

func supportsSinglePlayer(t GameType) bool {
	return t == gameTypeFlagz
}

type GameEngineMove struct {
	playerNum int
	move      int
	row       int
	col       int
	cellType  CellType
}

func (m *GameEngineMove) String() string {
	return fmt.Sprintf("P%d@%d (%d,%d/%d)", m.playerNum, m.move, m.row, m.col, m.cellType)
}

type GameEngine interface {
	Init()
	Start()
	InitialResources() ResourceInfo
	NumPlayers() int
	Reset()
	MakeMove(move GameEngineMove) bool
	Board() *Board
	IsDone() bool
	Winner() (playerNum int) // Results are only meaningful if IsDone() is true. 0 for draw.
	GameType() GameType
}

type SinglePlayerGameEngine interface {
	GameEngine
	// Returns a random move that can be played in the engine's current state.
	RandomMove() (GameEngineMove, error)
	// Returns a clone of the engine, e.g. to use in MCTS.
	// The source of randomness needs to be provided by callers. If the cloned
	// engine is only used in the same goroutine as the original one G, it is safe
	// to reuse G's source.
	Clone(s rand.Source) SinglePlayerGameEngine
}

// Dispatches on the gameType to create a corresponding GameEngine.
// The returned GameEngine is initialized and ready to play.
func NewGameEngine(gameType GameType) GameEngine {
	var ge GameEngine
	switch gameType {
	case gameTypeClassic:
		ge = &GameEngineClassic{}
	case gameTypeFlagz:
		ge = &GameEngineFlagz{}
	case gameTypeFreeform:
		ge = &GameEngineFreeform{}
	default:
		panic("Unconsidered game type: " + gameType)
	}
	ge.Init()
	return ge
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

func InitBoard(g GameEngine) *Board {
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
	return b
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
