package hexz

import (
	"fmt"
	"math/rand"
	"time"
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
}

type SinglePlayerGameEngine interface {
	GameEngine
	RandomMove() (GameEngineMove, error)
	SetBoard(b *Board)
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

//
// The "classic" hexz game
//

type GameEngineClassic struct {
	board *Board
}

func (g *GameEngineClassic) Board() *Board { return g.board }
func (g *GameEngineClassic) Init() {
	g.board = InitBoard(g)
}
func (g *GameEngineClassic) Start() {
	g.board.State = Running
}

func (g *GameEngineClassic) Reset() {
	g.Init()
	g.Start()
}

func (g *GameEngineClassic) NumPlayers() int {
	return 2
}

func (g *GameEngineClassic) InitialResources() ResourceInfo {
	return ResourceInfo{
		NumPieces: map[CellType]int{
			cellNormal: -1, // unlimited
			cellFire:   1,
			cellFlag:   1,
			cellPest:   1,
			cellDeath:  1,
		},
	}
}

func (g *GameEngineClassic) IsDone() bool { return g.board.State == Finished }

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

func (g *GameEngineClassic) Winner() (playerNum int) {
	if !g.IsDone() {
		return 0
	}
	return scoreBasedSingleWinner(g.board.Score)
}

func (f *Field) occupied() bool {
	return f.Type != cellNormal || f.Owner > 0
}

func (g *GameEngineClassic) recomputeScoreAndState() {
	b := g.board
	s := []int{0, 0}
	openCells := 0
	for _, row := range b.Fields {
		for _, fld := range row {
			if fld.Owner > 0 && !fld.Hidden {
				s[fld.Owner-1]++
			}
			if !fld.occupied() || fld.Hidden {
				// Don't finish the game until all cells are owned and not hidden, or dead.
				openCells++
			}
		}
	}
	b.Score = s
	if openCells == 0 {
		// No more inconclusive cells: game is finished
		b.State = Finished
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

func floodFill(b *Board, x idx, cb func(idx) bool) {
	var ns [6]idx
	if !cb(x) {
		return
	}
	n := b.neighbors(x, ns[:])
	for i := 0; i < n; i++ {
		floodFill(b, ns[i], cb)
	}
}

func (c CellType) lifetime() int {
	switch c {
	case cellFire, cellDead, cellDeath:
		return 1
	case cellPest:
		return 3
	}
	return -1 // Live forever
}

func occupyFields(b *Board, playerNum, r, c int, ct CellType) int {
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
	f.lifetime = ct.lifetime()
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
		floodFill(b, ns[k], func(x idx) bool {
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
					f.lifetime = cellNormal.lifetime()
				}
			}
		}
	}
	return numOccupiedFields
}

func applyFireEffect(b *Board, r, c int) {
	var ns [6]idx
	n := b.neighbors(idx{r, c}, ns[:])
	for i := 0; i < n; i++ {
		f := &b.Fields[ns[i].r][ns[i].c]
		f.Owner = 0
		f.Type = cellDead
		f.Hidden = false
		f.lifetime = cellDead.lifetime()
	}
}

func applyPestEffect(b *Board) {
	var ns [6]idx
	for r := 0; r < len(b.Fields); r++ {
		for c := 0; c < len(b.Fields[r]); c++ {
			// A pest cell does not propagate in its first round.
			if b.Fields[r][c].Type == cellPest && b.Fields[r][c].lifetime < cellPest.lifetime() {
				n := b.neighbors(idx{r, c}, ns[:])
				for i := 0; i < n; i++ {
					f := &b.Fields[ns[i].r][ns[i].c]
					if f.Owner > 0 && f.Owner != b.Fields[r][c].Owner && f.Type == cellNormal {
						// Pest only affects the opponent's normal cells.
						f.Owner = b.Fields[r][c].Owner
						f.Type = cellPest
						f.lifetime = cellPest.lifetime()
					}
				}
			}
		}
	}
}

func (g *GameEngineClassic) MakeMove(m GameEngineMove) bool {
	board := g.board
	turn := board.Turn
	if m.playerNum != turn || m.move != board.Move {
		// Only allow moves by players whose turn it is.
		return false
	}
	if !board.valid(idx{m.row, m.col}) || !m.cellType.isPlayerPiece() {
		// Invalid move request.
		return false
	}
	if m.cellType != cellNormal && board.Resources[turn-1].NumPieces[m.cellType] == 0 {
		// No pieces left of requested type
		return false
	}
	numOccupiedFields := 0
	revealBoard := m.cellType != cellNormal && m.cellType != cellFlag
	if board.Fields[m.row][m.col].occupied() {
		if board.Fields[m.row][m.col].Hidden && board.Fields[m.row][m.col].Owner == (3-turn) {
			// Conflicting hidden moves. Leads to dead cell.
			board.Move++
			f := &board.Fields[m.row][m.col]
			f.Owner = 0
			f.Type = cellDead
			f.lifetime = cellDead.lifetime()
			revealBoard = true
		} else if m.cellType == cellDeath {
			// Death cell can be placed anywhere and will "kill" whatever was there before.
			f := &board.Fields[m.row][m.col]
			f.Owner = turn
			f.Type = cellDeath
			f.Hidden = false
			f.lifetime = cellDeath.lifetime()
		} else {
			// Cannot make move on already occupied field.
			return false
		}
	} else {
		// Free cell: occupy it.
		board.Move++
		f := &board.Fields[m.row][m.col]
		if m.cellType == cellFire {
			// Fire cells take effect immediately.
			f.Owner = turn
			f.Type = m.cellType
			f.lifetime = cellFire.lifetime()
			applyFireEffect(board, m.row, m.col)
		} else {
			numOccupiedFields = occupyFields(board, turn, m.row, m.col, m.cellType)
		}
	}
	if m.cellType != cellNormal {
		board.Resources[turn-1].NumPieces[m.cellType]--
	}
	// Update turn.
	board.Turn++
	if board.Turn > 2 {
		board.Turn = 1
	}
	if numOccupiedFields > 1 || board.Move-board.LastRevealed == 4 || revealBoard {
		// Reveal hidden moves.
		for r := 0; r < len(board.Fields); r++ {
			for c := 0; c < len(board.Fields[r]); c++ {
				f := &board.Fields[r][c]
				f.Hidden = false
			}
		}
		applyPestEffect(board)
		// Clean up old special cells.
		for r := 0; r < len(board.Fields); r++ {
			for c := 0; c < len(board.Fields[r]); c++ {
				f := &board.Fields[r][c]
				if f.occupied() && f.lifetime == 0 {
					f.Owner = 0
					f.Hidden = false
					f.Type = cellNormal
					f.lifetime = cellNormal.lifetime()
				}
				if f.lifetime > 0 {
					f.lifetime--
				}
			}
		}

		board.LastRevealed = board.Move
	}
	g.recomputeScoreAndState()
	return true
}

//
// The freeform single-player hexz game.
//

type GameEngineFreeform struct {
	board *Board
}

func (g *GameEngineFreeform) Board() *Board { return g.board }

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

type GameEngineFlagz struct {
	board *Board
	rnd   *rand.Rand
}

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
	openCells := false
	canMove1 := false
	canMove2 := false
Outer:
	for r := range b.Fields {
		for c := range b.Fields[r] {
			fld := &b.Fields[r][c]
			if !fld.occupied() {
				openCells = true
				if fld.isAvail(1) {
					canMove1 = true
				}
				if fld.isAvail(2) {
					canMove2 = true
				}
			}
			if canMove1 && canMove2 {
				break Outer
			}
		}
	}
	canMove1 = canMove1 || (openCells && b.Resources[0].NumPieces[cellFlag] > 0)
	canMove2 = canMove2 || (openCells && b.Resources[1].NumPieces[cellFlag] > 0)
	// If only one player has valid moves left, it's their turn.
	if !canMove1 && canMove2 {
		b.Turn = 2
	} else if canMove1 && !canMove2 {
		b.Turn = 1
	}
	// Check if the game is over:
	// * No open cells left
	// * None of the players can move
	// * One player cannot move, the other one is leading
	s := b.Score
	if !openCells || !(canMove1 || canMove2) ||
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

func (f *Field) isAvail(playerNum int) bool {
	return f.nextVal[playerNum-1] > 0
}

// Validates if (r, c) would be a legal move for playerNum.
// Returns the value that the cell would get if the move were made.
// This method does not validate that it's playerNum's turn and it assumes
// that (r, c) is a valid index.
func (g *GameEngineFlagz) validateNormalMove(playerNum, r, c int) (ok bool, val int) {
	// A normal cell can only be placed next to another normal cell
	// of the same color, or next to a flag of the same color.
	b := g.board
	fld := &b.Fields[r][c]
	if fld.occupied() {
		return false, 0
	}
	if !fld.isAvail(playerNum) {
		// Cannot place a cell if there is no neighboring cell of the same player.
		return false, 0
	}
	return true, fld.nextVal[playerNum-1]
}

// Marks all cells neighboring (r, c) as available/blocked for the player owning (r, c).
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
				nb.Blocked |= 1 << pIdx
				nb.nextVal[pIdx] = -1
			} else if nb.nextVal[pIdx] == 0 || nb.nextVal[pIdx] > f.Value+1 {
				nb.nextVal[pIdx] = f.Value + 1
			}
		}
	}
}

// Occupies all grass cells around (r, c) that have at most the value
// that (r, c) has.
func (g *GameEngineFlagz) occupyGrassCells(r, c int) {
	var ns [6]idx
	b := g.board
	f := &b.Fields[r][c]
	if f.Owner == 0 || f.Value == 0 {
		return
	}
	n := b.neighbors(idx{r, c}, ns[:])
	for i := 0; i < n; i++ {
		nb := &b.Fields[ns[i].r][ns[i].c]
		if nb.Type == cellGrass && nb.Value <= f.Value {
			nb.Type = cellNormal
			nb.Owner = f.Owner
			b.Score[f.Owner-1] += nb.Value
			g.updateNeighborCells(ns[i].r, ns[i].c)
		}
	}
}

func (g *GameEngineFlagz) MakeMove(m GameEngineMove) bool {
	b := g.board
	turn := b.Turn
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
	if b.Resources[turn-1].NumPieces[m.cellType] == 0 {
		// No pieces left of requested type
		return false
	}
	if m.cellType == cellNormal {
		ok, val := g.validateNormalMove(turn, m.row, m.col)
		if !ok {
			return false
		}
		f.Owner = turn
		f.Type = cellNormal
		f.lifetime = g.lifetime(cellNormal)
		f.Hidden = false
		f.Value = val
		b.Score[turn-1] += val
		g.updateNeighborCells(m.row, m.col)
		g.occupyGrassCells(m.row, m.col)

	} else if m.cellType == cellFlag {
		// A flag can be placed on any free cell. It does not add to the score.
		f.Owner = turn
		f.Type = cellFlag
		f.lifetime = g.lifetime(cellFlag)
		f.Hidden = false
		f.Value = 0
		g.updateNeighborCells(m.row, m.col)
		b.Resources[turn-1].NumPieces[cellFlag]--
	} else {
		// Invalid piece. Just be caught by resource check already, so never reached.
		return false
	}
	b.Turn = 3 - b.Turn
	b.Move++
	g.recomputeState()
	return true
}

func (g *GameEngineFlagz) Board() *Board     { return g.board }
func (g *GameEngineFlagz) SetBoard(b *Board) { g.board = b }

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
	playerNum := g.board.Turn
	type mov struct {
		row int8
		col int8
	}
	const maxMoves = 105
	var normalMoves [maxMoves]mov
	nMoves := 0
	var flagMoves [maxMoves]mov
	nFlags := 0
	flagsLeft := b.Resources[playerNum-1].NumPieces[cellFlag] > 0
	for r := 0; r < len(b.Fields); r++ {
		for c := 0; c < len(b.Fields[r]); c++ {
			f := &b.Fields[r][c]
			if f.occupied() {
				continue
			}
			if flagsLeft {
				flagMoves[nFlags] = mov{row: int8(r), col: int8(c)}
				nFlags += 1
			}
			if f.isAvail(playerNum) {
				normalMoves[nMoves] = mov{row: int8(r), col: int8(c)}
				nMoves++
			}
		}
	}
	if nFlags > 0 {
		// Place a flag with a probability depending on the % of flag moves, unless it's the only legal move.
		if nMoves == 0 || g.rnd.Float64() <= float64(1)/float64(nMoves+1) {
			m := flagMoves[g.rnd.Intn(nFlags)]
			return GameEngineMove{
				playerNum: playerNum,
				move:      b.Move,
				row:       int(m.row),
				col:       int(m.col),
				cellType:  cellFlag,
			}, nil
		}
	}
	if nMoves == 0 {
		panic("no legal moves")
	}
	m := normalMoves[g.rnd.Intn(nMoves)]
	return GameEngineMove{
		playerNum: playerNum,
		move:      b.Move,
		row:       int(m.row),
		col:       int(m.col),
		cellType:  cellNormal,
	}, nil
}
