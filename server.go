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
	ongoingGames    map[string]*GameHandle
	ongoingGamesMut sync.Mutex

	// Contains all logged in players, mapped by their (cookie) playerId.
	loggedInPlayers    map[string]*Player
	loggedInPlayersMut sync.Mutex
	// Server configuration (set from command-line flags).
	config *ServerConfig
}

func NewServer(cfg *ServerConfig) *Server {
	return &Server{
		ongoingGames:    make(map[string]*GameHandle),
		loggedInPlayers: make(map[string]*Player),
		config:          cfg,
	}
}

const (
	maxLoggedInPlayers = 10000

	playerIdCookieName   = "playerId"
	gameHtmlFilename     = "game.html"
	loginHtmlFilename    = "login.html"
	newGameHtmlFilename  = "new.html"
	rulesHtmlFilename    = "rules.html"
	userDatabaseFilename = "_users.json"
)

var (
	// File extensions of resources we are willing to serve.
	// Mostly a safety measure against directory scanning attacks.
	validResourceFileExts = map[string]string{
		".html": "text/html",
		".png":  "image/png",
		".gif":  "image/gif",
	}
	// Paths that browsers may request to retrieve a favicon.
	// Keep in sync with files in the images/ folder.
	faviconUrlPaths = []string{
		"/favicon-16x16.png",
		"/favicon-32x32.png",
		"/favicon-48x48.png",
		"/apple-touch-icon.png",
	}
)

func isFavicon(path string) bool {
	for _, p := range faviconUrlPaths {
		if p == path {
			return true
		}
	}
	return false
}

type GameHandle struct {
	id           string
	started      time.Time
	gameType     GameType
	host         string            // Name of the player hosting the game (the one who created it)
	controlEvent chan ControlEvent // The channel to communicate with the game coordinating goroutine.
	done         chan struct{}     // Closed by the game master goroutine when it is done.
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

// Used in responses to list active games (/hexz/gamez).
type GameInfo struct {
	Id      string    `json:"id"`
	Host    string    `json:"host"`
	Started time.Time `json:"started"`
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
	MoveRequest
}

type ControlEventReset struct {
	playerId string
	message  string
}

func (e ControlEventRegister) controlEventImpl()   {}
func (e ControlEventUnregister) controlEventImpl() {}
func (e ControlEventMove) controlEventImpl()       {}
func (e ControlEventReset) controlEventImpl()      {}

func (g *GameHandle) sendEvent(e ControlEvent) bool {
	select {
	case g.controlEvent <- e:
		return true
	case <-g.done:
		return false
	}
}

func (g *GameHandle) registerPlayer(p *Player) (chan ServerEvent, error) {
	ch := make(chan chan ServerEvent)
	if g.sendEvent(ControlEventRegister{Player: p, ReplyChan: ch}) {
		return <-ch, nil
	}
	return nil, fmt.Errorf("cannot register player %s in game %s: game over", p.Id, g.id)
}

func (g *GameHandle) unregisterPlayer(playerId string) {
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

func NewGame(id, host string, gameType GameType) *GameHandle {
	return &GameHandle{
		id:           id,
		started:      time.Now(),
		gameType:     gameType,
		host:         host,
		controlEvent: make(chan ControlEvent),
		done:         make(chan struct{}),
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

// Controller function for a running game. To be executed by a dedicated goroutine.
func (s *Server) gameMaster(game *GameHandle) {
	defer close(game.done)
	defer s.deleteGame(game.id)
	log.Printf("Started new game %s", game.id)
	gameEngine := NewGameEngine(game.gameType)
	eventListeners := make(map[string]chan ServerEvent)
	defer func() {
		// Signal that client SSE connections should be terminated.
		for _, ch := range eventListeners {
			close(ch)
		}
	}()
	type pInfo struct {
		playerNum int
		player    *Player
	}
	players := make(map[string]pInfo)
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
				var playerNum int
				added := false
				if p, ok := players[e.Player.Id]; ok {
					// Player reconnected. Cancel its removal.
					if cancel, ok := playerRmCancel[e.Player.Id]; ok {
						close(cancel)
						delete(playerRmCancel, e.Player.Id)
					}
					playerNum = p.playerNum
				} else if len(players) < gameEngine.NumPlayers() {
					added = true
					playerNum = len(players) + 1
					players[e.Player.Id] = pInfo{playerNum, e.Player}
					if len(players) == gameEngine.NumPlayers() {
						gameEngine.Start()
					}
				}
				ch := make(chan ServerEvent)
				eventListeners[e.Player.Id] = ch
				e.ReplyChan <- ch
				// Send board and player role initially so client can display the UI.
				singlecast(e.Player.Id, ServerEvent{Board: gameEngine.Board(), Role: playerNum})
				announcements := []string{}
				if added {
					announcements = append(announcements, fmt.Sprintf("Welcome %s!", e.Player.Name))
				}
				if added && gameEngine.Board().State == Running {
					announcements = append(announcements, "The game begins!")
				}
				broadcast(ServerEvent{Board: gameEngine.Board(), Announcements: announcements})
			case ControlEventUnregister:
				delete(eventListeners, e.PlayerId)
				if _, ok := playerRmCancel[e.PlayerId]; ok {
					// A repeated unregister should not happen. If it does, we ignore
					// it and just wait for the existing scheduled removal to trigger.
					break
				}
				if _, ok := players[e.PlayerId]; ok {
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
				}
			case ControlEventMove:
				p, ok := players[e.PlayerId]
				if !ok || gameEngine.Board().State != Running {
					// Ignore invalid move request
					break
				}
				if s.config.DebugMode {
					debugReq, _ := json.Marshal(e.MoveRequest)
					log.Printf("%s: move request: P%d %s", game.id, p.playerNum, debugReq)
				}
				if gameEngine.MakeMove(GameEngineMove{playerNum: p.playerNum, row: e.Row, col: e.Col, cellType: e.Type}) {
					announcements := []string{}
					if gameEngine.IsDone() {
						winner := gameEngine.Winner()
						if winner > 0 {
							winnerName := ""
							for _, pi := range players {
								if pi.playerNum == winner {
									winnerName = pi.player.Name
								}
							}
							msg := fmt.Sprintf("&#127942; &#127942; &#127942; %s wins &#127942; &#127942; &#127942;",
								winnerName)
							announcements = append(announcements, msg)
						}
					}
					broadcast(ServerEvent{Board: gameEngine.Board(), Announcements: announcements})
				}
			case ControlEventReset:
				gameEngine.Reset()
				broadcast(ServerEvent{Board: gameEngine.Board()})
			}
		case <-tick:
			broadcast(ServerEvent{DebugMessage: "ping"})
		case playerId := <-playerRm:
			log.Printf("Player %s left game %s: game over", playerId, game.id)
			playerName := "?"
			if p, ok := players[playerId]; ok {
				playerName = p.player.Name
			}
			broadcast(ServerEvent{
				// Send sad emoji.
				Announcements: []string{fmt.Sprintf("Player %s left the game &#128546;. Game over.", playerName)},
			})
			return
		}
	}
}

func (s *Server) startNewGame(host string, gameType GameType) (*GameHandle, error) {
	// Try a few times to find an unused game Id, else give up.
	// (I don't like forever loops... 100 attempts is plenty.)
	var game *GameHandle
	for i := 0; i < 100; i++ {
		id := generateGameId()
		s.ongoingGamesMut.Lock()
		if _, ok := s.ongoingGames[id]; !ok {
			game = NewGame(id, host, gameType)
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

func (s *Server) lookupGame(id string) *GameHandle {
	s.ongoingGamesMut.Lock()
	defer s.ongoingGamesMut.Unlock()
	return s.ongoingGames[id]
}

func (s *Server) listRecentGames(limit int) []*GameInfo {
	s.ongoingGamesMut.Lock()
	gameInfos := []*GameInfo{}
	for _, g := range s.ongoingGames {
		gameInfos = append(gameInfos, &GameInfo{
			Id:      g.id,
			Host:    g.host,
			Started: g.started,
		})
	}
	s.ongoingGamesMut.Unlock()
	sort.Slice(gameInfos, func(i, j int) bool {
		return gameInfos[i].Started.After(gameInfos[j].Started)
	})
	if limit > len(gameInfos) {
		limit = len(gameInfos)
	}
	return gameInfos[:limit]
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
	html, err := s.readFile(newGameHtmlFilename)
	if err != nil {
		http.Error(w, "Failed to load html", http.StatusInternalServerError)
		log.Fatal("Cannot read new game HTML page: ", err.Error())
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(html)
}

func (s *Server) handleNewGame(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid method", http.StatusBadRequest)
		return
	}
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		s.handleLoginPage(w, r)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Invalid form", http.StatusBadRequest)
		return
	}
	typeParam := r.Form.Get("type")
	if typeParam == "" {
		http.Error(w, "Missing 'type' form parameter", http.StatusBadRequest)
		return
	}
	if !validGameType(typeParam) {
		http.Error(w, "Invalid value for 'type'", http.StatusBadRequest)
		return
	}
	game, err := s.startNewGame(p.Name, GameType(typeParam))
	if err != nil {
		http.Error(w, err.Error(), http.StatusPreconditionFailed)
	}
	http.Redirect(w, r, fmt.Sprintf("/hexz/%s", game.id), http.StatusSeeOther)
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
	game.sendEvent(ControlEventMove{PlayerId: player.Id, MoveRequest: req})
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
	// Headers to establish server-sent events (SSE) communication.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	for {
		select {
		case ev, ok := <-serverEventChan:
			if !ok {
				log.Printf("Closing SSE channel for player %s in game %s", p.Id, gameId)
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
			log.Printf("%s Player %s closed SSE channel", r.RemoteAddr, p.Id)
			game.unregisterPlayer(p.Id)
			return
		}
	}
}

func (s *Server) handleGamez(w http.ResponseWriter, r *http.Request) {
	gameInfos := s.listRecentGames(5)
	json, err := json.Marshal(gameInfos)
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		panic("Cannot marshal []GameInfo: " + err.Error())
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(json)
}

func (s *Server) handleGame(w http.ResponseWriter, r *http.Request) {
	_, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	g := s.lookupGame(gameIdFromPath(r.URL.Path))
	if g == nil {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	gameHtml, err := s.readFile(gameHtmlFilename)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(gameHtml)
}

func (s *Server) handleFile(filepath string, w http.ResponseWriter, r *http.Request) {
	contentType, ok := validResourceFileExts[path.Ext(filepath)]
	if !ok {
		http.Error(w, "", http.StatusNotFound)
		return
	}
	contents, err := s.readFile(filepath)
	if err != nil {
		http.Error(w, "", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.Write(contents)
}

func (s *Server) handleStaticResource(w http.ResponseWriter, r *http.Request) {
	hexz := "/hexz/"
	if !strings.HasPrefix(r.URL.Path, hexz) {
		http.Error(w, "", http.StatusNotFound)
		return
	}
	s.handleFile(r.URL.Path[len(hexz):], w, r)
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
		ico, err := s.readFile(path.Join("images", path.Base(r.URL.Path)))
		if err != nil {
			http.Error(w, "favicon not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write(ico)
		return
	}
	// Ignore
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

func (s *Server) loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// t := time.Now().Format("2006-01-02 15:04:05.999Z07:00")
		log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL.String())
		h.ServeHTTP(w, r)
	})
}

func (s *Server) Serve() {
	addr := fmt.Sprintf("%s:%d", s.config.ServerAddress, s.config.ServerPort)
	mux := &http.ServeMux{}
	srv := &http.Server{
		Addr:    addr,
		Handler: s.loggingHandler(mux),
	}

	// Quick sanity check that we have access to the game HTML file.
	if _, err := s.readFile(gameHtmlFilename); err != nil {
		log.Fatal("Cannot load game HTML: ", err)
	}
	mux.HandleFunc("/hexz/move/", s.handleMove)
	mux.HandleFunc("/hexz/reset/", s.handleReset)
	mux.HandleFunc("/hexz/sse/", s.handleSse)
	mux.HandleFunc("/hexz/login", s.handleLoginRequest)
	mux.HandleFunc("/hexz/rules", func(w http.ResponseWriter, r *http.Request) {
		s.handleFile(rulesHtmlFilename, w, r)
	})
	mux.HandleFunc("/hexz/images/", s.handleStaticResource)
	mux.HandleFunc("/hexz", s.handleHexz)
	mux.HandleFunc("/hexz/new", s.handleNewGame)
	mux.HandleFunc("/hexz/gamez", s.handleGamez)
	mux.HandleFunc("/hexz/", s.handleGame)
	mux.HandleFunc("/statusz", s.handleStatusz)
	mux.HandleFunc("/", s.defaultHandler)

	log.Printf("Listening on %s", addr)

	s.loadUserDatabase()
	// Start login GC routine
	go s.updateLoggedInPlayers()

	if s.config.TlsCertChain != "" && s.config.TlsPrivKey != "" {
		log.Fatal(srv.ListenAndServeTLS(s.config.TlsCertChain, s.config.TlsPrivKey))
	}
	log.Fatal(srv.ListenAndServe())
}
