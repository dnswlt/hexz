package hexz

import "time"

type GameState string

const (
	Initial  GameState = "initial"
	Running  GameState = "running"
	Finished GameState = "finished"
)

// JSON for server responses.

type ServerEvent struct {
	Timestamp time.Time  `json:"timestamp"` // RFC3339 formatted.
	Board     *BoardView `json:"board"`
	// Role of the client receiving the event. 0: spectator, 1, 2: players.
	Role          int      `json:"role"`
	PlayerNames   []string `json:"playerNames"`
	Announcements []string `json:"announcements"`
	DebugMessage  string   `json:"debugMessage"`
	// Number of the player that wins. 0 if no winner yet or draw.
	Winner int `json:"winner,omitempty"`
	// Only sent in the first event to clients.
	GameInfo *ServerEventGameInfo `json:"gameInfo,omitempty"`
	// If true, the client should not display undo/redo buttons.
	DisableUndo bool `json:"disableUndo,omitempty"`
	// Signals to clients that this is the last event they will receive.
	LastEvent bool `json:"lastEvent"`
}

// Sent in an initial message to clients.
type ServerEventGameInfo struct {
	// Indicates which cell types exist in this type of game.
	ValidCellTypes []CellType `json:"validCellTypes"`
	// The type of game we're playing.
	GameType GameType `json:"gameType"`
	// True if this is a game of one player against a CPU player, which
	// should be run on the client side.
	ClientSideCPUPlayer bool `json:"clientSideCPUPlayer"`
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

type MoveScores struct {
	NormalCell [][]float64 `json:"normalCell"` // Scores for placing a normal cell on a field.
	Flag       [][]float64 `json:"flag"`       // Scores for placing a flag on a field.
}

type Field struct {
	Type    CellType `json:"type"`
	Owner   int      `json:"owner,omitempty"` // Player number owning this field. 0 for unowned fields.
	Hidden  bool     `json:"hidden,omitempty"`
	Value   int      `json:"v"`                 // Some games assign different values to cells.
	Blocked uint8    `json:"blocked,omitempty"` // Indicates which players this field is blocked for.
	// Internal fields, not exported in JSON
	Lifetime int    `json:"-"` // Moves left until this cell gets cleared. -1 means infinity.
	NextVal  [2]int `json:"-"` // If this cell would be occupied, what value would it have? (For Flagz)
}

// Information about the resources each player has left.
type ResourceInfo struct {
	NumPieces [cellTypeLen]int `json:"numPieces"`
}

type CellType int

// Remember to update the known cell types in game.html if you make changes here!
// Add new types at the end, since otherwise loading old games will break.
const (
	cellNormal CellType = iota // Empty cells if not owned, otherwise the player's regular cell.
	// Non-player cells.
	cellDead  // A dead cell. Usually generated from a piece placement conflict.
	cellGrass // Introduced in the Flagz game. Cells that can be collected.
	cellRock  // Unowned and similar to a dead cell. Can be used to build static obstacles.
	// Player's special pieces. Update isPlayerPiece() if you make changes.
	cellFire
	cellFlag
	cellPest
	cellDeath
	// Add new types here, right before cellTypeLen.
	cellTypeLen // End marker for CellType. Should never be used.
)

func (c CellType) valid() bool {
	return c >= cellNormal && c < cellTypeLen
}

// JSON for game history.
type GameHistoryResponse struct {
	GameId      string                      `json:"gameId"`
	PlayerNames []string                    `json:"playerNames"`
	GameType    GameType                    `json:"gameType,omitempty"`
	Entries     []*GameHistoryResponseEntry `json:"entries"`
}

type GameHistoryResponseEntry struct {
	Timestamp time.Time    `json:"timestamp"` // RFC3339 formatted.
	EntryType string       `json:"entryType"` // One of {"move", "undo", "redo", "reset"}.
	Move      *MoveRequest `json:"move"`      // Only populated if the EntryType is "move"
	Board     *BoardView   `json:"board"`
	// For single-player flagz: scores that the CPU assigns to each move.
	MoveScores *MoveScores `json:"moveScores,omitempty"`
}

// JSON for incoming requests from UI clients.
type MoveRequest struct {
	Move int      `json:"move"` // Used to discard move requests that do not match the game's current state.
	Row  int      `json:"row"`
	Col  int      `json:"col"`
	Type CellType `json:"type"`
}

type ResetRequest struct {
	Message string `json:"message"`
}

type UndoRequest struct {
	Move int `json:"move"`
}

type RedoRequest struct {
	Move int `json:"move"`
}

// Used in /hexz/status responses.
type GameStateResponse struct {
	GameId           string `json:"gameId"`
	EncodedGameState []byte `json:"encodedGameState"`
}

// Used in responses to list active games (/hexz/gamez).
type GameInfo struct {
	Id       string    `json:"id"`
	Host     string    `json:"host"`
	Started  time.Time `json:"started"`
	GameType GameType  `json:"gameType"`
}

// Used to report CPU stats by clients.
type WASMStatsRequest struct {
	GameId   string    `json:"gameId"`
	GameType GameType  `json:"gameType"`
	Move     int       `json:"move"`
	UserInfo UserInfo  `json:"userInfo"`
	Stats    WASMStats `json:"stats"`
}

type WASMStats struct {
	// MCTS stats.
	TreeSize   int           `json:"treeSize"`
	MaxDepth   int           `json:"maxDepth"`
	Iterations int           `json:"iterations"`
	Elapsed    time.Duration `json:"elapsed"`
	// Memory allocations, in MiB (1024*1024 bytes).
	TotalAllocMiB float64 `json:"totalAllocMiB"`
	HeapAllocMiB  float64 `json:"heapAllocMiB"`
}

type UserInfo struct {
	// The User-Agent header.
	UserAgent string `json:"userAgent"`
	// Taken from navigator.language.
	Language string `json:"language"`
	// Resolution is the screen resolution in pixels [window.screen.width, window.screen.height].
	Resolution [2]int `json:"resolution"`
	// Viewport is the size of the viewport in pixels [window.innerWidth, window.innerHeight].
	Viewport [2]int `json:"viewport"`
	// BrowserWindow is the size of the browser window in pixels [window.outerWidth, window.outerHeight].
	BrowserWindow [2]int `json:"browserWindow"`
	// HardwareConcurrency is the number of logical processors available to run threads
	// on the user's computer (navigator.hardwareConcurrency).
	HardwareConcurrency int `json:"hardwareConcurrency"`
}

// /statusz messages.

type StatuszCounter struct {
	Name  string `json:"name"`
	Value int64  `json:"value"`
}

type StatuszDistribBucket struct {
	Lower float64 `json:"lower"`
	Upper float64 `json:"upper"` // exclusive
	Count int64   `json:"count"`
}

type StatuszDistrib struct {
	Name    string                 `json:"name"`
	Buckets []StatuszDistribBucket `json:"buckets"`
}

type StatuszResponse struct {
	Started            time.Time         `json:"started"`
	UptimeSeconds      int               `json:"uptimeSeconds"`
	Uptime             string            `json:"uptime"` // 1h30m3.5s
	NumOngoingGames    int               `json:"numOngoingGames"`
	NumLoggedInPlayers *int              `json:"numLoggedInPlayers,omitempty"` // pointer to make this one optional (remote store does not support count).
	Counters           []StatuszCounter  `json:"counters"`
	Distributions      []*StatuszDistrib `json:"distributions"`
}
