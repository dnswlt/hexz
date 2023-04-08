package hexz

type GameState string

const (
	Initial  GameState = "initial"
	Running  GameState = "running"
	Finished GameState = "finished"
)

// JSON for server responses.

type ServerEvent struct {
	Timestamp     string     `json:"timestamp"`
	Board         *BoardView `json:"board"`
	Role          int        `json:"role"` // 0: spectator, 1, 2: players
	Announcements []string   `json:"announcements"`
	DebugMessage  string     `json:"debugMessage"`
	LastEvent     bool       `json:"lastEvent"` // Signals to clients that this is the last event they will receive.
}

// A player's or spectator's view of the board.
// See type Board for the internal representation that holds the complete information.
type BoardView struct {
	Turn      int            `json:"turn"`
	Move      int            `json:"move"`
	Fields    [][]Field      `json:"fields"` // The board's fields.
	Score     []int          `json:"score"`  // Depending on the number of players, 1 or 2 elements.
	Resources []ResourceInfo `json:"resources"`
	State     GameState      `json:"state"`
}

type Field struct {
	Type     CellType `json:"type"`
	Owner    int      `json:"owner"` // Player number owning this field. 0 for unowned fields.
	Hidden   bool     `json:"hidden,omitempty"`
	Value    int      `json:"v"`                 // Some games assign different values to cells.
	Lifetime int      `json:"-"`                 // Moves left until this cell gets cleared. -1 means infinity.
	Blocked  int      `json:"blocked,omitempty"` // Bitmap indicating which players this field is blocked for.
}

// Information about the resources each player has left.
type ResourceInfo struct {
	NumPieces map[CellType]int `json:"numPieces"`
}

type CellType int

const (
	cellNormal CellType = iota
	cellDead
	cellGrass
	cellFire
	cellFlag
	cellPest
	cellDeath
)

func (c CellType) valid() bool {
	return c >= cellNormal && c <= cellDeath
}
