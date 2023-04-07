package hexz

type GameState string

const (
	Initial  GameState = "initial"
	Running  GameState = "running"
	Finished GameState = "finished"
)

// JSON for server responses.

type ServerEvent struct {
	Timestamp     string   `json:"timestamp"`
	Board         *Board   `json:"board"`
	Role          int      `json:"role"` // 0: spectator, 1, 2: players
	Announcements []string `json:"announcements"`
	DebugMessage  string   `json:"debugMessage"`
	LastEvent     bool     `json:"lastEvent"` // Signals to clients that this is the last event they will receive.
}

type Board struct {
	Turn         int            `json:"turn"`
	Move         int            `json:"move"`
	LastRevealed int            `json:"-"`      // Move at which fields were last revealed
	FlatFields   []Field        `json:"-"`      //
	Fields       [][]Field      `json:"fields"` // Subslices of FlatFields
	Score        []int          `json:"score"`
	Resources    []ResourceInfo `json:"resources"`
	State        GameState      `json:"state"`
}

type Field struct {
	Type     CellType `json:"type"`
	Owner    int      `json:"owner"` // Player number owning this field. 0 for unowned fields.
	Hidden   bool     `json:"hidden"`
	Value    int      `json:"v"` // Some games assign different values to cells.
	Lifetime int      `json:"-"` // Moves left until this cell gets cleared. -1 means infinity.
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
