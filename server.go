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
	ServerAddress string
	ServerPort    int
	DocumentRoot  string
	GameGcDelay   time.Duration
	LoginTtl      time.Duration

	TlsCertChain string
	TlsPrivKey   string
	DebugMode    bool
}

var (
	ongoingGames    = make(map[string]*Game)
	ongoingGamesMut sync.Mutex

	loggedInPlayers    = make(map[string]*Player)
	loggedInPlayersMut sync.Mutex

	usernameRegexp = regexp.MustCompile("^[a-zA-Z0-9][a-zA-Z0-9_.-]+$")

	serverConfig *ServerConfig
)

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

// JSON for server responses.
type Field struct {
	Value int `json:"value"`
	Owner int `json:"owner"` // Player number owning this field.
}

type GameState string

const (
	Initial                GameState = "initial"
	WaitingForSecondPlayer GameState = "waiting"
	Running                GameState = "running"
	Finished               GameState = "finished"
	Aborted                GameState = "aborted"
)

type Board struct {
	Turn         int       `json:"turn"`
	Move         int       `json:"move"`
	LastRevealed int       `json:"-"` // Move at which fields were last revealed
	Fields       [][]Field `json:"fields"`
	Score        []int     `json:"score"` // Always two elements
	State        GameState `json:"state"`
}

type ServerEvent struct {
	Timestamp     string   `json:"timestamp"`
	Board         *Board   `json:"board"`
	Role          int      `json:"role"` // 0: spectator, 1, 2: players
	Announcements []string `json:"announcements"`
	DebugMessage  string   `json:"debugMessage"`
	ActiveGames   []string `json:"activeGames"`
	LastEvent     bool     `json:"lastEvent"` // Signals to clients that this is the last event they will receive.
}

// JSON for incoming requests from UI clients.
type MoveRequest struct {
	Row int `json:"row"`
	Col int `json:"col"`
}
type ResetRequest struct {
	Message string `json:"message"`
}

type StatuszResponse struct {
	NumOngoingGames int `json:"numOngoingGames"`
}

type Player struct {
	Id         string
	Name       string
	LastActive time.Time
}

func lookupPlayer(playerId string) *Player {
	loggedInPlayersMut.Lock()
	defer loggedInPlayersMut.Unlock()

	p, ok := loggedInPlayers[playerId]
	if !ok {
		return nil
	}
	p.LastActive = time.Now()
	return p
}

func loginPlayer(playerId string, name string) bool {
	loggedInPlayersMut.Lock()
	defer loggedInPlayersMut.Unlock()

	if len(loggedInPlayers) > maxLoggedInPlayers {
		// TODO: GC the logged in players to avoid running out of space.
		// The login logic is very hacky for the time being.
		return false
	}
	p := &Player{
		Id:         playerId,
		Name:       name,
		LastActive: time.Now(),
	}
	loggedInPlayers[playerId] = p
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

func readFile(filename string) ([]byte, error) {
	return os.ReadFile(path.Join(serverConfig.DocumentRoot, filename))
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
	return &Board{
		Turn:   1, // Player 1 begins
		Fields: fields,
		Score:  []int{0, 0},
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

func recomputeScore(b *Board) {
	s := []int{0, 0}
	for _, row := range b.Fields {
		for _, fld := range row {
			if fld.Owner > 0 {
				s[fld.Owner-1]++
			}
		}
	}
	b.Score = s
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

func occupyFields(b *Board, playerNum, i, j int) int {
	// Create a copy of the board that indicates which neighboring cell of (i, j)
	// it shares the free or opponent's area with.
	// Then find the smallest of these areas and occupy every free cell in it.
	ms := make([][]int8, len(b.Fields))
	for k := 0; k < len(ms); k++ {
		ms[k] = make([]int8, len(b.Fields[k]))
	}
	b.Fields[i][j].Value = 2
	b.Fields[i][j].Owner = playerNum
	areaSizes := make(map[int8]int)
	var ns [6]idx
	n := b.neighbors(idx{i, j}, ns[:])
	for k := 0; k < n; k++ {
		k1 := int8(k + 1)
		floodFill(b, ns[k], func(x idx) bool {
			if ms[x.r][x.c] > 0 {
				// Already seen.
				return false
			}
			if b.Fields[x.r][x.c].Value > 0 {
				// Occupied fields act as boundaries.
				return false
			}
			// Mark field as visited in k-th loop iteration.
			ms[x.r][x.c] = k1
			areaSizes[k1]++
			return true
		})
	}
	// If there is more than one area, we know we introduced a split, since the areas
	// would have been connected by the previously free cell (i, j).
	numFields := 0
	if len(areaSizes) > 1 {
		minN := int8(0)
		for n, cnt := range areaSizes {
			if minN == 0 || areaSizes[minN] > cnt {
				minN = n
			}
		}
		for r := 0; r < len(b.Fields); r++ {
			for c := 0; c < len(b.Fields[r]); c++ {
				if ms[r][c] == minN {
					numFields++
					b.Fields[r][c].Value = 1
					b.Fields[r][c].Owner = playerNum
				}
			}
		}
	}
	return numFields
}

// Controller function for a running game. To be executed by a dedicated goroutine.
func gameMaster(game *Game) {
	const numPlayers = 2
	defer close(game.done)
	defer deleteGame(game.Id)
	log.Printf("New gameMaster started for game %s", game.Id)
	gcTimeout := serverConfig.GameGcDelay
	board := NewBoard()
	eventListeners := make(map[string]chan ServerEvent)
	defer func() {
		// Signal that client SSE connections should be terminated.
		for _, ch := range eventListeners {
			close(ch)
		}
	}()
	var players [2]*Player
	playerGcCancel := make(map[string]chan struct{})
	gcChan := make(chan string)
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
						// Player reconnected. Cancel its GC.
						if cancel, ok := playerGcCancel[e.Player.Id]; ok {
							log.Printf("Player %s reconnected. Cancelling GC.", e.Player)
							close(cancel)
							delete(playerGcCancel, e.Player.Id)
						}
						break
					}
				}
				ch := make(chan ServerEvent)
				eventListeners[e.Player.Id] = ch
				e.ReplyChan <- ch
				// Send board and player role initially so client can display the UI.
				role := (playerIdx + 1) % 3 // 0 == spectator
				singlecast(e.Player.Id, ServerEvent{Board: board, ActiveGames: listRecentGames(5), Role: role})
				if role == 1 {
					announcement := fmt.Sprintf("Welcome %s! Waiting for player 2.", e.Player.Name)
					broadcast(ServerEvent{Announcements: []string{announcement}})
				} else if role == 2 {
					announcement := fmt.Sprintf("The game %s vs %s begins!", players[0].Name, players[1].Name)
					broadcast(ServerEvent{Announcements: []string{announcement}})
				}
			case ControlEventUnregister:
				delete(eventListeners, e.PlayerId)
				if _, ok := playerGcCancel[e.PlayerId]; ok {
					// A repeated unregister should not happen. If it does, we ignore
					// it and just wait for the existing GC "callback" to happen.
					break
				}
				// Remove player after timeout. Don't remove them immediately as they might
				// just be reloading their page and rejoin soon.
				cancelChan := make(chan struct{})
				playerGcCancel[e.PlayerId] = cancelChan
				go func(playerId string) {
					t := time.After(gcTimeout)
					select {
					case <-t:
						gcChan <- playerId
					case <-cancelChan:
					}
				}(e.PlayerId)
			case ControlEventMove:
				turn := board.Turn
				p := players[turn-1]
				if p == nil || p.Id != e.PlayerId {
					// Only allow moves by players whose turn it is.
					break
				}
				if !board.valid(idx{e.Row, e.Col}) {
					// Invalid field indices.
					break
				}
				numOccupiedFields := 0
				conflict := false
				if board.Fields[e.Row][e.Col].Value > 0 {
					if board.Fields[e.Row][e.Col].Value == 2 && board.Fields[e.Row][e.Col].Owner != turn {
						// Conflicting hidden moves. Leads to dead cell.
						board.Fields[e.Row][e.Col].Value = 3
						board.Fields[e.Row][e.Col].Owner = 0
						board.Move++
						conflict = true
					} else {
						// Cannot make move on already occupied field.
						break
					}
				} else {
					// Free cell: occupy it.
					numOccupiedFields = occupyFields(board, turn, e.Row, e.Col)
					board.Move++
				}
				// Update turn.
				board.Turn++
				if board.Turn > numPlayers {
					board.Turn = 1
				}
				if numOccupiedFields > 1 || board.Move-board.LastRevealed == 4 || conflict {
					// Reveal hidden moves.
					for r := 0; r < len(board.Fields); r++ {
						for c := 0; c < len(board.Fields[r]); c++ {
							if board.Fields[r][c].Value == 2 {
								board.Fields[r][c].Value = 1
							}
						}
					}
					board.LastRevealed = board.Move
				}
				recomputeScore(board)
				broadcast(ServerEvent{Board: board})
			case ControlEventReset:
				board = NewBoard()
				broadcast(ServerEvent{Board: board})
			}
		case <-tick:
			broadcast(ServerEvent{ActiveGames: listRecentGames(5), DebugMessage: "ping"})
		case playerId := <-gcChan:
			if _, ok := playerGcCancel[playerId]; !ok {
				// Ignore zombie GC message. Player has already reconnected.
				log.Printf("Ignoring GC message for player %s in game %s", playerId, game.Id)
			}
			log.Printf("Player %s has left game %s. Game over.", playerId, game.Id)
			broadcast(ServerEvent{
				Announcements: []string{"A player left the game. Game over."},
			})
			return
		}
	}
}

func startNewGame() (*Game, error) {
	// Try a few times to find an unused game Id, else give up.
	// (I don't like forever loops... 100 attempts is plenty.)
	var game *Game
	for i := 0; i < 100; i++ {
		id := generateGameId()
		ongoingGamesMut.Lock()
		if _, ok := ongoingGames[id]; !ok {
			game = NewGame(id)
			ongoingGames[id] = game
		}
		ongoingGamesMut.Unlock()
		if game != nil {
			go gameMaster(game)
			return game, nil
		}
	}
	return nil, fmt.Errorf("cannot start a new game")
}

func deleteGame(id string) {
	ongoingGamesMut.Lock()
	defer ongoingGamesMut.Unlock()
	delete(ongoingGames, id)
}

func lookupGame(id string) *Game {
	ongoingGamesMut.Lock()
	defer ongoingGamesMut.Unlock()
	return ongoingGames[id]
}

func listRecentGames(limit int) []string {
	ongoingGamesMut.Lock()
	games := []*Game{}
	for _, g := range ongoingGames {
		games = append(games, g)
	}
	ongoingGamesMut.Unlock()
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

func handleLoginPage(w http.ResponseWriter, r *http.Request) {
	html, err := readFile(loginHtmlFilename)
	if err != nil {
		http.Error(w, "Failed to load login screen", http.StatusInternalServerError)
		panic(err.Error())
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(html)
}

func isValidPlayerName(name string) bool {
	return len(name) >= 3 && len(name) <= 20 && usernameRegexp.MatchString(name)
}

func handleLoginRequest(w http.ResponseWriter, r *http.Request) {
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
	if !loginPlayer(playerId, name) {
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

func handleHexz(w http.ResponseWriter, r *http.Request) {
	_, err := lookupPlayerFromCookie(r)
	if err != nil {
		handleLoginPage(w, r)
		return
	}

	// For now, immediately create a new game and redirect to it.
	game, err := startNewGame()
	if err != nil {
		http.Error(w, err.Error(), http.StatusPreconditionFailed)
	}
	http.Redirect(w, r, fmt.Sprintf("%s/%s", r.URL.Path, game.Id), http.StatusSeeOther)
}

func validatePostRequest(r *http.Request) (*Player, error) {
	if r.Method != http.MethodPost {
		return nil, fmt.Errorf("invalid method")
	}
	return lookupPlayerFromCookie(r)
}

func lookupPlayerFromCookie(r *http.Request) (*Player, error) {
	cookie, err := r.Cookie(playerIdCookieName)
	if err != nil {
		return nil, fmt.Errorf("missing cookie")
	}
	p := lookupPlayer(cookie.Value)
	if p == nil {
		return nil, fmt.Errorf("player not found")
	}
	return p, nil
}

func handleMove(w http.ResponseWriter, r *http.Request) {
	player, err := validatePostRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	dec := json.NewDecoder(r.Body)
	var req MoveRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	gameId := gameIdFromPath(r.URL.Path)
	game := lookupGame(gameId)
	if game == nil {
		http.Error(w, fmt.Sprintf("No game with ID %q", gameId), http.StatusNotFound)
		return
	}
	game.sendEvent(ControlEventMove{PlayerId: player.Id, Row: req.Row, Col: req.Col})
}

func handleReset(w http.ResponseWriter, r *http.Request) {
	player, err := validatePostRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	dec := json.NewDecoder(r.Body)
	var req ResetRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	gameId := gameIdFromPath(r.URL.Path)
	game := lookupGame(gameId)
	if game == nil {
		http.Error(w, fmt.Sprintf("No game with ID %q", gameId), http.StatusNotFound)
		return
	}
	game.sendEvent(ControlEventReset{playerId: player.Id, message: req.Message})

}

func handleSse(w http.ResponseWriter, r *http.Request) {
	// We expect a cookie to identify the p.
	p, err := lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	gameId := gameIdFromPath(r.URL.Path)
	game := lookupGame(gameId)
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

func handleGame(w http.ResponseWriter, r *http.Request) {
	_, err := lookupPlayerFromCookie(r)
	if err != nil {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
	}
	gameHtml, err := readFile(gameHtmlFilename)
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

func handleStatusz(w http.ResponseWriter, r *http.Request) {
	var resp StatuszResponse
	ongoingGamesMut.Lock()
	resp.NumOngoingGames = len(ongoingGames)
	ongoingGamesMut.Unlock()
	enc := json.NewEncoder(w)
	if err := enc.Encode(resp); err != nil {
		http.Error(w, "Serialization error", http.StatusInternalServerError)
		panic(fmt.Sprintf("Cannot serialize my own structs?! %s", err))
	}
}

func defaultHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
	}
	if isFavicon(r.URL.Path) {
		ico, err := os.ReadFile(path.Join(serverConfig.DocumentRoot, "images", path.Base(r.URL.Path)))
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

func loadUserDatabase() {
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
	loggedInPlayersMut.Lock()
	defer loggedInPlayersMut.Unlock()
	for _, p := range players {
		if _, ok := loggedInPlayers[p.Id]; !ok {
			// Only add players, don't overwrite anything existing in memory.
			loggedInPlayers[p.Id] = p
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

func userMaintenance() {
	lastIteration := time.Now()
	period := time.Duration(5) * time.Minute
	if serverConfig.DebugMode {
		// Clean up active users more frequently in debug mode.
		period = time.Duration(5) * time.Second
	}
	for {

		t := time.NewTicker(period)
		<-t.C
		activity := false
		now := time.Now()
		logoutThresh := now.Add(-serverConfig.LoginTtl)
		loggedInPlayersMut.Lock()
		del := []string{}
		for pId, p := range loggedInPlayers {
			if p.LastActive.Before(logoutThresh) {
				del = append(del, pId)
			} else if p.LastActive.After(lastIteration) {
				activity = true
			}
		}
		for _, pId := range del {
			delete(loggedInPlayers, pId)
		}
		loggedInPlayersMut.Unlock()
		// Do I/O outside the mutex.
		for _, pId := range del {
			log.Printf("Logged out player %s", pId)
		}
		if activity || len(del) > 0 {
			loggedInPlayersMut.Lock()
			// Create copies of the players to avoid data race during serialization.
			// (LastActive can get updated at any time by other goroutines.)
			players := make([]Player, len(loggedInPlayers))
			i := 0
			for _, p := range loggedInPlayers {
				players[i] = *p
				i++
			}
			loggedInPlayersMut.Unlock()
			saveUserDatabase(players)
		}
		lastIteration = now
	}
}

func Serve(cfg *ServerConfig) {
	serverConfig = cfg
	// Make sure we have access to the game HTML file.
	if _, err := readFile(gameHtmlFilename); err != nil {
		log.Fatal("Cannot load game HTML: ", err)
	}
	http.HandleFunc("/hexz/move/", handleMove)
	http.HandleFunc("/hexz/reset/", handleReset)
	http.HandleFunc("/hexz/sse/", handleSse)
	http.HandleFunc("/hexz/login", handleLoginRequest)
	http.HandleFunc("/hexz", handleHexz)
	http.HandleFunc("/hexz/", handleGame)
	http.HandleFunc("/statusz", handleStatusz)
	http.HandleFunc("/", defaultHandler)

	addr := fmt.Sprintf("%s:%d", cfg.ServerAddress, cfg.ServerPort)
	log.Printf("Listening on %s", addr)

	loadUserDatabase()
	// Start login GC routine
	go userMaintenance()

	if cfg.TlsCertChain != "" && cfg.TlsPrivKey != "" {
		log.Fatal(http.ListenAndServeTLS(addr, cfg.TlsCertChain, cfg.TlsPrivKey, nil))
	}
	log.Fatal(http.ListenAndServe(addr, nil))
}
