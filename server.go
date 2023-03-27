package hexz

import (
	crand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

type ServerConfig struct {
	ServerAddress     string
	ServerPort        int
	DocumentRoot      string
	PlayerRemoveDelay time.Duration
	LoginTtl          time.Duration

	TlsCertChain string
	TlsPrivKey   string
	DebugMode    bool
}

var (
	// Regexp used to validate player names.
	playernameRegexp = regexp.MustCompile("^[a-zA-Z0-9][a-zA-Z0-9_.-]+$")
)

type Server struct {
	// Contains all ongoing games, mapped by their ID.
	ongoingGames    map[string]*Game
	ongoingGamesMut sync.Mutex

	// Contains all logged in players, mapped by their (cookie) playerId.
	loggedInPlayers    map[string]*Player
	loggedInPlayersMut sync.Mutex
	// Server configuration (set from command-line flags).
	config *ServerConfig
}

func NewServer(cfg *ServerConfig) *Server {
	return &Server{
		ongoingGames:    make(map[string]*Game),
		loggedInPlayers: make(map[string]*Player),
		config:          cfg,
	}
}

const (
	numFieldsFirstRow = 10
	numBoardRows      = 11

	maxLoggedInPlayers = 10000

	playerIdCookieName   = "playerId"
	gameHtmlFilename     = "game.html"
	loginHtmlFilename    = "login.html"
	userDatabaseFilename = "_users.json"
)

type Game struct {
	Id           string
	Started      time.Time
	controlEvent chan ControlEvent // The channel to communicate with the game coordinating goroutine.
	done         chan struct{}     // Closed by the game master goroutine when it is done.
}

type GameState string

const (
	Initial                GameState = "initial"
	WaitingForSecondPlayer GameState = "waiting"
	Running                GameState = "running"
	Finished               GameState = "finished"
	Aborted                GameState = "aborted"
)

// JSON for server responses.

type ServerEvent struct {
	Timestamp     string   `json:"timestamp"`
	Board         *Board   `json:"board"`
	Role          int      `json:"role"` // 0: spectator, 1, 2: players
	Announcements []string `json:"announcements"`
	DebugMessage  string   `json:"debugMessage"`
	ActiveGames   []string `json:"activeGames"`
	LastEvent     bool     `json:"lastEvent"` // Signals to clients that this is the last event they will receive.
}

type Board struct {
	Turn         int             `json:"turn"`
	Move         int             `json:"move"`
	LastRevealed int             `json:"-"` // Move at which fields were last revealed
	Fields       [][]Field       `json:"fields"`
	Score        [2]int          `json:"score"`
	Resources    [2]ResourceInfo `json:"resources"`
	State        GameState       `json:"state"`
}

type Field struct {
	Type         CellType `json:"type"`
	Owner        int      `json:"owner"` // Player number owning this field. 0 for unowned fields.
	Hidden       bool     `json:"hidden"`
	LastModified int      `json:"-"` // Move on which this field was last modified.
}

type CellType int

const (
	cellNormal CellType = iota
	cellDead
	cellFire
	cellFlag
	cellPest
	cellDeath
)

func (c CellType) valid() bool {
	return c >= cellNormal && c <= cellDeath
}

// Information about the resources each player has left.
type ResourceInfo struct {
	NumPieces map[CellType]int `json:"numPieces"`
}

// JSON for incoming requests from UI clients.
type MoveRequest struct {
	Row  int      `json:"row"`
	Col  int      `json:"col"`
	Type CellType `json:"type"`
}

type ResetRequest struct {
	Message string `json:"message"`
}

type StatuszResponse struct {
	NumOngoingGames    int
	NumLoggedInPlayers int
}

type Player struct {
	Id         string
	Name       string
	LastActive time.Time
}

func (s *Server) lookupPlayer(playerId string) *Player {
	s.loggedInPlayersMut.Lock()
	defer s.loggedInPlayersMut.Unlock()

	p, ok := s.loggedInPlayers[playerId]
	if !ok {
		return nil
	}
	p.LastActive = time.Now()
	return p
}

func (s *Server) loginPlayer(playerId string, name string) bool {
	s.loggedInPlayersMut.Lock()
	defer s.loggedInPlayersMut.Unlock()

	if len(s.loggedInPlayers) > maxLoggedInPlayers {
		// TODO: GC the logged in players to avoid running out of space.
		// The login logic is very hacky for the time being.
		return false
	}
	p := &Player{
		Id:         playerId,
		Name:       name,
		LastActive: time.Now(),
	}
	s.loggedInPlayers[playerId] = p
	return true
}

// Control events are sent to the game master goroutine.
type ControlEvent interface {
	controlEventImpl() // Interface marker function
}

type ControlEventRegister struct {
	Player    *Player
	ReplyChan chan chan ServerEvent
}

type ControlEventUnregister struct {
	PlayerId string
}

type ControlEventMove struct {
	PlayerId string
	Row      int
	Col      int
	Type     CellType
}

type ControlEventReset struct {
	playerId string
	message  string
}

func (e ControlEventRegister) controlEventImpl()   {}
func (e ControlEventUnregister) controlEventImpl() {}
func (e ControlEventMove) controlEventImpl()       {}
func (e ControlEventReset) controlEventImpl()      {}

func (g *Game) sendEvent(e ControlEvent) bool {
	select {
	case g.controlEvent <- e:
		return true
	case <-g.done:
		return false
	}
}

func (g *Game) registerPlayer(p *Player) (chan ServerEvent, error) {
	ch := make(chan chan ServerEvent)
	if g.sendEvent(ControlEventRegister{Player: p, ReplyChan: ch}) {
		return <-ch, nil
	}
	return nil, fmt.Errorf("cannot register player %s in game %s: game over", p.Id, g.Id)
}

func (g *Game) unregisterPlayer(playerId string) {
	g.sendEvent(ControlEventUnregister{PlayerId: playerId})
}

func (s *Server) readFile(filename string) ([]byte, error) {
	return os.ReadFile(path.Join(s.config.DocumentRoot, filename))
}

// Generates a random 128-bit hex string representing a player ID.
func generatePlayerId() string {
	p := make([]byte, 16)
	crand.Read(p)
	return hex.EncodeToString(p)
}

// Generates a 6-letter game ID.
func generateGameId() string {
	var alphabet = []rune("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	var b strings.Builder
	for i := 0; i < 6; i++ {
		b.WriteRune(alphabet[rand.Intn(len(alphabet))])
	}
	return b.String()
}

func NewGame(id string) *Game {
	return &Game{
		Id:           id,
		Started:      time.Now(),
		controlEvent: make(chan ControlEvent),
		done:         make(chan struct{}),
	}
}

func NewBoard() *Board {
	fields := make([][]Field, numBoardRows)
	for i := 0; i < len(fields); i++ {
		n := numFieldsFirstRow - i%2
		fields[i] = make([]Field, n)
	}
	initialPieces := func() map[CellType]int {
		return map[CellType]int{
			cellNormal: -1, // Unlimited
			cellFire:   1,
			cellFlag:   1,
			cellPest:   0,
			cellDeath:  0,
		}
	}
	return &Board{
		Turn:   1, // Player 1 begins
		Fields: fields,
		Resources: [2]ResourceInfo{
			{NumPieces: initialPieces()},
			{NumPieces: initialPieces()},
		},
	}
}

// Looks up the game ID from the URL path.
func gameIdFromPath(path string) string {
	pathSegs := strings.Split(path, "/")
	l := len(pathSegs)
	if l >= 2 && pathSegs[1] == "hexz" {
		return pathSegs[l-1]
	}
	return ""
}

func (f *Field) occupied() bool {
	return f.Type == cellDead || f.Owner > 0
}

func recomputeScoreAndState(b *Board) {
	s := [2]int{0, 0}
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
	b.Fields[r][c].Owner = playerNum
	b.Fields[r][c].Type = ct
	b.Fields[r][c].Hidden = true
	b.Fields[r][c].LastModified = b.Move
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
					f.LastModified = b.Move
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
		f.Hidden = false
		f.Type = cellDead
		f.LastModified = b.Move
	}
}

func makeMove(e ControlEventMove, board *Board, players [2]*Player) bool {
	turn := board.Turn
	p := players[turn-1]
	if p == nil || p.Id != e.PlayerId {
		// Only allow moves by players whose turn it is.
		return false
	}
	if !board.valid(idx{e.Row, e.Col}) || e.Type == cellDead {
		// Invalid move request.
		return false
	}
	if e.Type != cellNormal && board.Resources[turn-1].NumPieces[e.Type] == 0 {
		// No pieces left of requested type
		return false
	}
	numOccupiedFields := 0
	conflict := false
	if board.Fields[e.Row][e.Col].occupied() {
		if board.Fields[e.Row][e.Col].Hidden && board.Fields[e.Row][e.Col].Owner == (3-turn) {
			// Conflicting hidden moves. Leads to dead cell.
			board.Move++
			f := &board.Fields[e.Row][e.Col]
			f.Type = cellDead
			f.Owner = 0
			f.LastModified = board.Move
			conflict = true
		} else {
			// Cannot make move on already occupied field.
			return false
		}
	} else {
		// Free cell: occupy it.
		board.Move++
		numOccupiedFields = occupyFields(board, turn, e.Row, e.Col, e.Type)
	}
	board.Resources[turn-1].NumPieces[e.Type]--
	// Update turn.
	board.Turn++
	if board.Turn > 2 {
		board.Turn = 1
	}
	if numOccupiedFields > 1 || board.Move-board.LastRevealed == 4 || conflict {
		// Reveal hidden moves and apply effects.
		for r := 0; r < len(board.Fields); r++ {
			for c := 0; c < len(board.Fields[r]); c++ {
				f := &board.Fields[r][c]
				f.Hidden = false
				if f.Type == cellFire && f.LastModified > board.LastRevealed {
					applyFireEffect(board, r, c)
				}
			}
		}
		// Clean up old dead cells and fires.
		for r := 0; r < len(board.Fields); r++ {
			for c := 0; c < len(board.Fields[r]); c++ {
				f := &board.Fields[r][c]
				if (f.Type == cellDead || f.Type == cellFire) && f.LastModified <= board.LastRevealed {
					f.Type = cellNormal
					f.Owner = 0
					f.LastModified = board.Move
				}
			}
		}
		board.LastRevealed = board.Move
	}
	recomputeScoreAndState(board)
	return true
}

// Controller function for a running game. To be executed by a dedicated goroutine.
func (s *Server) gameMaster(game *Game) {
	defer close(game.done)
	defer s.deleteGame(game.Id)
	log.Printf("New gameMaster started for game %s", game.Id)
	board := NewBoard()
	eventListeners := make(map[string]chan ServerEvent)
	defer func() {
		// Signal that client SSE connections should be terminated.
		for _, ch := range eventListeners {
			close(ch)
		}
	}()
	var players [2]*Player
	playerRmCancel := make(map[string]chan struct{})
	playerRm := make(chan string)
	broadcast := func(e ServerEvent) {
		e.Timestamp = time.Now().Format(time.RFC3339)
		for _, ch := range eventListeners {
			ch <- e
		}
	}
	singlecast := func(playerId string, e ServerEvent) {
		if ch, ok := eventListeners[playerId]; ok {
			e.Timestamp = time.Now().Format(time.RFC3339)
			ch <- e
		}
	}
	for {
		tick := time.After(5 * time.Second)
		select {
		case ce := <-game.controlEvent:
			switch e := ce.(type) {
			case ControlEventRegister:
				playerIdx := 0
				for ; playerIdx < 2; playerIdx++ {
					if players[playerIdx] == nil {
						// The first two participants in the game are players.
						// Anyone arriving later will be a spectator.
						players[playerIdx] = e.Player
						break
					} else if players[playerIdx].Id == e.Player.Id {
						// Player reconnected. Cancel its removal.
						if cancel, ok := playerRmCancel[e.Player.Id]; ok {
							close(cancel)
							delete(playerRmCancel, e.Player.Id)
						}
						break
					}
				}
				ch := make(chan ServerEvent)
				eventListeners[e.Player.Id] = ch
				e.ReplyChan <- ch
				// Send board and player role initially so client can display the UI.
				role := (playerIdx + 1) % 3 // 0 == spectator
				singlecast(e.Player.Id, ServerEvent{Board: board, ActiveGames: s.listRecentGames(5), Role: role})
				if role == 1 {
					announcement := fmt.Sprintf("Welcome %s! Waiting for player 2.", e.Player.Name)
					broadcast(ServerEvent{Announcements: []string{announcement}})
				} else if role == 2 {
					announcement := fmt.Sprintf("Let the game %s vs %s begin!", players[0].Name, players[1].Name)
					broadcast(ServerEvent{Announcements: []string{announcement}})
				}
			case ControlEventUnregister:
				delete(eventListeners, e.PlayerId)
				if _, ok := playerRmCancel[e.PlayerId]; ok {
					// A repeated unregister should not happen. If it does, we ignore
					// it and just wait for the existing GC "callback" to happen.
					break
				}
				// Remove player after timeout. Don't remove them immediately as they might
				// just be reloading their page and rejoin soon.
				cancel := make(chan struct{})
				playerRmCancel[e.PlayerId] = cancel
				go func(playerId string) {
					t := time.After(s.config.PlayerRemoveDelay)
					select {
					case <-t:
						playerRm <- playerId
					case <-cancel:
					}
				}(e.PlayerId)
			case ControlEventMove:
				if makeMove(e, board, players) {
					announcements := []string{}
					if board.State == Finished {
						won := "~~ nobody ~~"
						if board.Score[0] > board.Score[1] {
							won = players[0].Name
						} else if board.Score[1] > board.Score[0] {
							won = players[1].Name
						}
						msg := fmt.Sprintf("&#127942; &#127942; &#127942; %s wins &#127942; &#127942; &#127942;", won)
						announcements = append(announcements, msg)
					}
					broadcast(ServerEvent{Board: board, Announcements: announcements})
				}
			case ControlEventReset:
				board = NewBoard()
				broadcast(ServerEvent{Board: board})
			}
		case <-tick:
			broadcast(ServerEvent{ActiveGames: s.listRecentGames(5), DebugMessage: "ping"})
		case playerId := <-playerRm:
			log.Printf("Player %s has left game %s. Game over.", playerId, game.Id)
			name := "?"
			for i := 0; i < len(players); i++ {
				if players[i] != nil && playerId == players[i].Id {
					name = players[i].Name
				}
			}
			broadcast(ServerEvent{
				Announcements: []string{fmt.Sprintf("Player %s left the game &#128546;. Game over.", name)},
			})
			return
		}
	}
}

func (s *Server) startNewGame() (*Game, error) {
	// Try a few times to find an unused game Id, else give up.
	// (I don't like forever loops... 100 attempts is plenty.)
	var game *Game
	for i := 0; i < 100; i++ {
		id := generateGameId()
		s.ongoingGamesMut.Lock()
		if _, ok := s.ongoingGames[id]; !ok {
			game = NewGame(id)
			s.ongoingGames[id] = game
		}
		s.ongoingGamesMut.Unlock()
		if game != nil {
			go s.gameMaster(game)
			return game, nil
		}
	}
	return nil, fmt.Errorf("cannot start a new game")
}

func (s *Server) deleteGame(id string) {
	s.ongoingGamesMut.Lock()
	defer s.ongoingGamesMut.Unlock()
	delete(s.ongoingGames, id)
}

func (s *Server) lookupGame(id string) *Game {
	s.ongoingGamesMut.Lock()
	defer s.ongoingGamesMut.Unlock()
	return s.ongoingGames[id]
}

func (s *Server) listRecentGames(limit int) []string {
	s.ongoingGamesMut.Lock()
	games := []*Game{}
	for _, g := range s.ongoingGames {
		games = append(games, g)
	}
	s.ongoingGamesMut.Unlock()
	sort.Slice(games, func(i, j int) bool {
		return games[i].Started.After(games[j].Started)
	})
	n := limit
	if limit > len(games) {
		n = len(games)
	}
	ids := make([]string, n)
	for i, g := range games[:n] {
		ids[i] = g.Id
	}
	return ids
}

func (s *Server) handleLoginPage(w http.ResponseWriter, r *http.Request) {
	html, err := s.readFile(loginHtmlFilename)
	if err != nil {
		http.Error(w, "Failed to load login screen", http.StatusInternalServerError)
		log.Fatal("Cannot read login HTML page: ", err.Error())
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(html)
}

func isValidPlayerName(name string) bool {
	return len(name) >= 3 && len(name) <= 20 && playernameRegexp.MatchString(name)
}

func (s *Server) handleLoginRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusBadRequest)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Invalid form", http.StatusBadRequest)
		return
	}
	name := r.Form.Get("name")
	if name == "" {
		http.Error(w, "Missing 'user' form parameter", http.StatusBadRequest)
		return
	}
	if !isValidPlayerName(name) {
		http.Error(w, fmt.Sprintf("Invalid username %q", name), http.StatusBadRequest)
		return
	}
	playerId := generatePlayerId()
	if !s.loginPlayer(playerId, name) {
		http.Error(w, "Cannot log in right now", http.StatusPreconditionFailed)
	}
	cookie := &http.Cookie{
		Name:     playerIdCookieName,
		Value:    playerId,
		Path:     "/hexz",
		MaxAge:   24 * 60 * 60,
		HttpOnly: true,
		Secure:   false, // also allow plain http
		SameSite: http.SameSiteLaxMode,
	}
	http.SetCookie(w, cookie)
	http.Redirect(w, r, "/hexz", http.StatusSeeOther)
}

func (s *Server) handleHexz(w http.ResponseWriter, r *http.Request) {
	_, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		s.handleLoginPage(w, r)
		return
	}

	// For now, immediately create a new game and redirect to it.
	game, err := s.startNewGame()
	if err != nil {
		http.Error(w, err.Error(), http.StatusPreconditionFailed)
	}
	http.Redirect(w, r, fmt.Sprintf("%s/%s", r.URL.Path, game.Id), http.StatusSeeOther)
}

func (s *Server) validatePostRequest(r *http.Request) (*Player, error) {
	if r.Method != http.MethodPost {
		return nil, fmt.Errorf("invalid method")
	}
	return s.lookupPlayerFromCookie(r)
}

func (s *Server) lookupPlayerFromCookie(r *http.Request) (*Player, error) {
	cookie, err := r.Cookie(playerIdCookieName)
	if err != nil {
		return nil, fmt.Errorf("missing cookie")
	}
	p := s.lookupPlayer(cookie.Value)
	if p == nil {
		return nil, fmt.Errorf("player not found")
	}
	return p, nil
}

func (s *Server) handleMove(w http.ResponseWriter, r *http.Request) {
	player, err := s.validatePostRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	dec := json.NewDecoder(r.Body)
	var req MoveRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if !req.Type.valid() {
		http.Error(w, "Invalid cell type", http.StatusBadRequest)
		return
	}
	gameId := gameIdFromPath(r.URL.Path)
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, fmt.Sprintf("No game with ID %q", gameId), http.StatusNotFound)
		return
	}
	game.sendEvent(ControlEventMove{PlayerId: player.Id, Row: req.Row, Col: req.Col, Type: req.Type})
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	p, err := s.validatePostRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	dec := json.NewDecoder(r.Body)
	var req ResetRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	gameId := gameIdFromPath(r.URL.Path)
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, fmt.Sprintf("No game with ID %q", gameId), http.StatusNotFound)
		return
	}
	game.sendEvent(ControlEventReset{playerId: p.Id, message: req.Message})

}

func (s *Server) handleSse(w http.ResponseWriter, r *http.Request) {
	// We expect a cookie to identify the p.
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	gameId := gameIdFromPath(r.URL.Path)
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, fmt.Sprintf("Game %s does not exist", gameId), http.StatusNotFound)
		return
	}
	serverEventChan, err := game.registerPlayer(p)
	if err != nil {
		http.Error(w, err.Error(), http.StatusPreconditionFailed)
		return
	}
	log.Printf("New SSE channel for player %s and game %s", p.Id, game.Id)
	// Headers to establish server-sent events (SSE) communication.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	for {
		select {
		case ev, ok := <-serverEventChan:
			if !ok {
				log.Printf("Closing SSE channel for player %s and game %s", p.Id, gameId)
				ev = ServerEvent{
					LastEvent: true,
				}
			}
			// Send ServerEvent JSON on SSE connection.
			var buf strings.Builder
			enc := json.NewEncoder(&buf)
			if err := enc.Encode(ev); err != nil {
				http.Error(w, "Serialization error", http.StatusInternalServerError)
				panic(fmt.Sprintf("Cannot serialize my own structs?! %s", err))
			}
			fmt.Fprintf(w, "data: %s\n\n", buf.String())
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			if !ok {
				// The game is over, time to close the SSE channel.
				return
			}
		case <-r.Context().Done():
			log.Printf("Player %s (%s) closed SSE channel", p.Id, r.RemoteAddr)
			game.unregisterPlayer(p.Id)
			return
		}
	}
}

func (s *Server) handleGame(w http.ResponseWriter, r *http.Request) {
	_, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
	}
	gameHtml, err := s.readFile(gameHtmlFilename)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(gameHtml)
}

func isFavicon(path string) bool {
	return path == "/favicon-16x16.png" || path == "/favicon-32x32.png" ||
		path == "/favicon-48x48.png" || path == "/apple-touch-icon.png"
}

func (s *Server) handleStatusz(w http.ResponseWriter, r *http.Request) {
	var resp StatuszResponse
	s.ongoingGamesMut.Lock()
	resp.NumOngoingGames = len(s.ongoingGames)
	s.ongoingGamesMut.Unlock()
	s.loggedInPlayersMut.Lock()
	resp.NumLoggedInPlayers = len(s.loggedInPlayers)
	s.loggedInPlayersMut.Unlock()
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(resp); err != nil {
		http.Error(w, "Serialization error", http.StatusInternalServerError)
		panic(fmt.Sprintf("Cannot serialize my own structs?! %s", err))
	}
}

func (s *Server) defaultHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
	}
	if isFavicon(r.URL.Path) {
		ico, err := os.ReadFile(path.Join(s.config.DocumentRoot, "images", path.Base(r.URL.Path)))
		if err != nil {
			http.Error(w, "favicon not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write(ico)
		return
	}
	// Ignore
	log.Print("Ignoring request for path: ", r.URL.Path, r.URL.RawQuery)
}

func (s *Server) loadUserDatabase() {
	r, err := os.Open(userDatabaseFilename)
	if err != nil {
		if err != os.ErrNotExist {
			log.Print("Failed to read user database: ", err.Error())
		}
		return
	}
	defer r.Close()
	dec := json.NewDecoder(r)
	var players []*Player
	if err := dec.Decode(&players); err != nil {
		log.Print("Corrupted user database: ", err.Error())
	}
	log.Printf("Loaded %d users from user db", len(players))
	s.loggedInPlayersMut.Lock()
	defer s.loggedInPlayersMut.Unlock()
	for _, p := range players {
		if _, ok := s.loggedInPlayers[p.Id]; !ok {
			// Only add players, don't overwrite anything existing in memory.
			s.loggedInPlayers[p.Id] = p
		}
	}
}

func saveUserDatabase(players []Player) {
	w, err := os.Create(userDatabaseFilename)
	if err != nil {
		log.Print("userMaintenance: cannot save user db: ", err.Error())
		return
	}
	defer w.Close()
	enc := json.NewEncoder(w)
	if err := enc.Encode(players); err != nil {
		log.Print("userMaintenance: error saving user db: ", err.Error())
	}
	log.Printf("Saved user db (%d users)", len(players))
}

func (s *Server) updateLoggedInPlayers() {
	lastIteration := time.Now()
	period := time.Duration(5) * time.Minute
	if s.config.DebugMode {
		// Clean up active users more frequently in debug mode.
		period = time.Duration(5) * time.Second
	}
	for {
		t := time.After(period)
		<-t
		activity := false
		now := time.Now()
		logoutThresh := now.Add(-s.config.LoginTtl)
		s.loggedInPlayersMut.Lock()
		del := []string{}
		for pId, p := range s.loggedInPlayers {
			if p.LastActive.Before(logoutThresh) {
				del = append(del, pId)
			} else if p.LastActive.After(lastIteration) {
				activity = true
			}
		}
		for _, pId := range del {
			delete(s.loggedInPlayers, pId)
		}
		s.loggedInPlayersMut.Unlock()
		// Do I/O outside the mutex.
		for _, pId := range del {
			log.Printf("Logged out player %s", pId)
		}
		if activity || len(del) > 0 {
			s.loggedInPlayersMut.Lock()
			// Create copies of the players to avoid data race during serialization.
			// (LastActive can get updated at any time by other goroutines.)
			players := make([]Player, len(s.loggedInPlayers))
			i := 0
			for _, p := range s.loggedInPlayers {
				players[i] = *p
				i++
			}
			s.loggedInPlayersMut.Unlock()
			saveUserDatabase(players)
		}
		lastIteration = now
	}
}

func (s *Server) Serve() {
	// Make sure we have access to the game HTML file.
	if _, err := s.readFile(gameHtmlFilename); err != nil {
		log.Fatal("Cannot load game HTML: ", err)
	}
	http.HandleFunc("/hexz/move/", s.handleMove)
	http.HandleFunc("/hexz/reset/", s.handleReset)
	http.HandleFunc("/hexz/sse/", s.handleSse)
	http.HandleFunc("/hexz/login", s.handleLoginRequest)
	http.HandleFunc("/hexz", s.handleHexz)
	http.HandleFunc("/hexz/", s.handleGame)
	http.HandleFunc("/statusz", s.handleStatusz)
	http.HandleFunc("/", s.defaultHandler)

	addr := fmt.Sprintf("%s:%d", s.config.ServerAddress, s.config.ServerPort)
	log.Printf("Listening on %s", addr)

	s.loadUserDatabase()
	// Start login GC routine
	go s.updateLoggedInPlayers()

	if s.config.TlsCertChain != "" && s.config.TlsPrivKey != "" {
		log.Fatal(http.ListenAndServeTLS(addr, s.config.TlsCertChain, s.config.TlsPrivKey, nil))
	}
	log.Fatal(http.ListenAndServe(addr, nil))
}
