package hexz

import (
	"compress/gzip"
	"context"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math/big"
	"net"
	"net/http"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ServerConfig struct {
	ServerHost        string
	ServerPort        int
	DocumentRoot      string        // Path to static resource files.
	GameHistoryRoot   string        // Path to game history files.
	LoginDatabasePath string        // Path to the file where the player DB is stored. If empty, no persistent storage is used.
	RedisAddr         string        // Address of the Redis server. If empty, local storage is used.
	PostgresURL       string        // URL of the PostgreSQL server. If empty, no persistent storage is used.
	InactivityTimeout time.Duration // Time after which a game is ended due to inactivity.
	PlayerRemoveDelay time.Duration // Time to wait before removing an unregistered player from the game.
	LoginTTL          time.Duration
	CpuThinkTime      time.Duration
	CpuMaxFlags       int
	AuthTokenSha256   string // Used in http Basic authentication for /statusz. Must be a SHA256 checksum.
	DisableUndo       bool   // If true, Undo/Redo is enabled for all games
	TlsCertChain      string
	TlsPrivKey        string
	DebugMode         bool
	// If true, run stateless server (e.g. for Cloud Run). RedisAddr must also be set in this case.
	Stateless bool
}

var (
	// Regexp used to validate player names.
	playernameRegexp = regexp.MustCompile(`^[\p{Latin}0-9_.-]+$`)

	// Loggers.
	// DO NOT USE FOR NEW CODE. Use the loggers and functions from the hlog subpackage.
	infoLog  = log.New(os.Stderr, "I ", log.Ldate|log.Ltime|log.Lshortfile)
	errorLog = log.New(os.Stderr, "E ", log.Ldate|log.Ltime|log.Lshortfile)
)

type Server struct {
	// Contains all ongoing games, mapped by their ID.
	ongoingGames    map[string]*GameHandle
	ongoingGamesMut sync.Mutex

	playerStore PlayerStore

	// Server configuration (set from command-line flags).
	config *ServerConfig

	// Counters
	counters    map[string]*Counter
	countersMut sync.Mutex

	// Distributions
	distrib    map[string]*Distribution
	distribMut sync.Mutex

	started time.Time
}

func NewServer(cfg *ServerConfig) (s *Server, err error) {
	playerStore, err := NewInMemoryPlayerStore(cfg.LoginTTL, cfg.LoginDatabasePath)
	if err != nil {
		return nil, err
	}
	s = &Server{
		ongoingGames: make(map[string]*GameHandle),
		playerStore:  playerStore,
		config:       cfg,
		counters:     make(map[string]*Counter),
		distrib:      make(map[string]*Distribution),
		started:      time.Now(),
	}
	s.InitCounters()
	return
}

func (s *Server) InitCounters() {
	checkedAdd := func(name string, bounds []float64) {
		if err := s.AddDistribution(name, bounds); err != nil {
			panic("Cannot create counter")
		}
	}
	checkedAdd(fmt.Sprintf("/games/%s/mcts/elapsed", gameTypeFlagz), DistribRange(0.001, 60*60, 1.1))
	checkedAdd(fmt.Sprintf("/games/%s/mcts/iterations", gameTypeFlagz), DistribRange(1, 1e9, 1.2))
	checkedAdd(fmt.Sprintf("/games/%s/mcts/tree_size", gameTypeFlagz), DistribRange(1, 1e9, 1.2))
	checkedAdd(fmt.Sprintf("/games/%s/mcts/iterations_per_sec", gameTypeFlagz), DistribRange(1, 1e6, 1.1))
}

func (s *Server) Counter(name string) *Counter {
	s.countersMut.Lock()
	defer s.countersMut.Unlock()

	if c, ok := s.counters[name]; ok {
		return c
	}
	c := NewCounter(name)
	s.counters[name] = c
	return c
}

func (s *Server) IncCounter(name string) {
	s.Counter(name).Increment()
}

func (s *Server) AddDistribution(name string, bounds []float64) error {
	s.distribMut.Lock()
	defer s.distribMut.Unlock()

	if _, ok := s.distrib[name]; ok {
		return fmt.Errorf("distribution %s already exists", name)
	}
	d, err := NewDistribution(name, bounds)
	if err != nil {
		return err
	}
	s.distrib[name] = d
	return nil
}

func (s *Server) AddDistribValue(name string, value float64) bool {
	s.distribMut.Lock()
	defer s.distribMut.Unlock()

	d, ok := s.distrib[name]
	if !ok {
		return false
	}
	d.Add(value)
	return true
}

const (
	playerIdCookieName   = "playerId"
	gameHtmlFilename     = "game.html"
	viewHtmlFilename     = "view.html"
	loginHtmlFilename    = "login.html"
	newGameHtmlFilename  = "new.html"
	rulesHtmlFilename    = "rules.html"
	userDatabaseFilename = "_users.json"
)

var (
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
	singlePlayer bool              // If true, only player 1 is human, the rest are computer-controlled.
	controlEvent chan ControlEvent // The channel to communicate with the game coordinating goroutine.
	ctx          context.Context   // This game's context. Closed by the game master when the game is done.
}

// Player has JSON annotations for serialization to disk.
// It is not used in the public API.
type Player struct {
	Id         PlayerId  `json:"id"`
	Name       string    `json:"name"`
	LastActive time.Time `json:"lastActive"`
}

// Control events are sent to the game master goroutine.
type ControlEvent interface {
	controlEventImpl() // Interface marker function
}

type ControlEventRegister struct {
	player    Player
	replyChan chan chan ServerEvent
}

type ControlEventUnregister struct {
	playerId PlayerId
}

type ControlEventMove struct {
	playerId    PlayerId
	moveRequest *MoveRequest
	mctsStats   *MCTSStats // Optional, only populated by CPU players.
}

type ControlEventReset struct {
	playerId PlayerId
	message  string
}

type ControlEventUndo struct {
	playerId PlayerId
	UndoRequest
}
type ControlEventRedo struct {
	playerId PlayerId
	RedoRequest
}

type ControlEventValidMoves struct {
	reply chan<- []*MoveRequest
}

func (e ControlEventRegister) controlEventImpl()   {}
func (e ControlEventUnregister) controlEventImpl() {}
func (e ControlEventMove) controlEventImpl()       {}
func (e ControlEventReset) controlEventImpl()      {}
func (e ControlEventUndo) controlEventImpl()       {}
func (e ControlEventRedo) controlEventImpl()       {}
func (e ControlEventValidMoves) controlEventImpl() {}

func (g *GameHandle) sendEvent(e ControlEvent) bool {
	select {
	case g.controlEvent <- e:
		return true
	case <-g.ctx.Done():
		return false
	}
}

func (g *GameHandle) registerPlayer(p Player) (chan ServerEvent, error) {
	ch := make(chan chan ServerEvent)
	if g.sendEvent(ControlEventRegister{player: p, replyChan: ch}) {
		return <-ch, nil
	}
	return nil, fmt.Errorf("cannot register player %s in game %s: game over", p.Id, g.id)
}

func (g *GameHandle) unregisterPlayer(playerId PlayerId) {
	g.sendEvent(ControlEventUnregister{playerId: playerId})
}

func (g *GameHandle) validMoves() []*MoveRequest {
	ch := make(chan []*MoveRequest)
	moves := []*MoveRequest{}
	if g.sendEvent(ControlEventValidMoves{reply: ch}) {
		if ms := <-ch; ms != nil {
			moves = ms
		}
	}
	return moves
}

// Sends the contents of filename to the ResponseWriter.
func (s *Server) serveHtmlFile(w http.ResponseWriter, filename string) {
	s.IncCounter("/storage/files/servehtml")
	if path.Ext(filename) != ".html" || strings.Contains(filename, "..") || strings.Contains(filename, "/") {
		// It's a programming error to call this method with non-html files.
		errorLog.Fatalf("Not a valid HTML file: %q\n", filename)
	}
	p := path.Join(s.config.DocumentRoot, filename)
	html, err := os.ReadFile(p)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			// We should only call this method for existing html files.
			errorLog.Fatalf("Cannot read %s: %s", p, err.Error())
		} else {
			// Whatever might cause us to fail reading our own files...
			errorLog.Printf("Could not read existing file %s: %s", p, err.Error())
			http.Error(w, "", http.StatusInternalServerError)
			return
		}
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(html)
}

func (s *Server) readStaticResource(filename string) ([]byte, error) {
	s.IncCounter("/storage/files/readfile")
	if strings.Contains(filename, "..") {
		return nil, fmt.Errorf("refusing to read %q", filename)
	}
	return os.ReadFile(path.Join(s.config.DocumentRoot, filename))
}

func (s *Server) readGameHistoryFromFile(gameId string) (*GameHistory, error) {
	s.IncCounter("/storage/files/gamehistory")
	return ReadGameHistory(s.config.GameHistoryRoot, gameId)
}

// Generates a random 128-bit hex string representing a player ID.
func generatePlayerId() PlayerId {
	p := make([]byte, 16)
	crand.Read(p)
	return PlayerId(hex.EncodeToString(p))
}

// Generates a 6-letter game ID.
func GenerateGameId() string {
	var alphabet = []rune("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	var b strings.Builder
	for i := 0; i < 6; i++ {
		max := big.NewInt(int64(len(alphabet)))
		n, err := crand.Int(crand.Reader, max)
		if err != nil {
			panic(fmt.Sprintf("cannot generate random number: %s", err.Error()))
		}
		b.WriteRune(alphabet[n.Int64()])
	}
	return b.String()
}

func isValidGameId(gameId string) bool {
	if len(gameId) != 6 {
		return false
	}
	for i := 0; i < len(gameId); i++ {
		if gameId[i] < 'A' || gameId[i] > 'Z' {
			return false
		}
	}
	return true
}

var (
	errInvalidGameId = errors.New("invalid game ID")
)

// Looks up the game ID as the last element of the URL path.
func gameIdFromPath(path string) (gameId string, err error) {
	pathSegs := strings.Split(path, "/")
	l := len(pathSegs)
	if l >= 2 && pathSegs[1] == "hexz" {
		gameId = pathSegs[l-1]
		if isValidGameId(gameId) {
			return
		}
	}
	return "", errInvalidGameId
}

func (s *Server) startNewGame(host string, gameType GameType, singlePlayer bool) (*GameHandle, error) {
	// Try a few times to find an unused game Id, else give up.
	// (I don't like forever loops... 100 attempts is plenty.)
	var game *GameHandle
	ctx, cancel := context.WithCancel(context.Background())
	for i := 0; i < 100; i++ {
		id := GenerateGameId()
		s.ongoingGamesMut.Lock()
		if _, ok := s.ongoingGames[id]; !ok {
			game = &GameHandle{
				id:           id,
				started:      time.Now(),
				gameType:     gameType,
				host:         host,
				singlePlayer: singlePlayer,
				controlEvent: make(chan ControlEvent),
				ctx:          ctx,
			}
			s.ongoingGames[id] = game
		}
		s.ongoingGamesMut.Unlock()
		if game != nil {
			m := NewGameMaster(s, game)
			go func() {
				m.Run(cancel)
				s.deleteGame(game.id)
			}()
			return game, nil
		}
	}
	cancel() // Avoid context leak
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
			Id:       g.id,
			Host:     g.host,
			Started:  g.started,
			GameType: g.gameType,
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

func isValidPlayerName(name string) bool {
	return len(name) >= 3 && len(name) <= 20 && playernameRegexp.MatchString(name)
}

func makePlayerCookie(playerId PlayerId, ttl time.Duration) *http.Cookie {
	return &http.Cookie{
		Name:     playerIdCookieName,
		Value:    string(playerId),
		Path:     "/hexz",
		MaxAge:   int(ttl.Seconds()),
		HttpOnly: true,  // Don't let JS access the cookie
		Secure:   false, // also allow plain http
		SameSite: http.SameSiteLaxMode,
	}
}

func (s *Server) handleLoginRequest(w http.ResponseWriter, r *http.Request) {
	s.IncCounter("/requests/login/total")
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Invalid form", http.StatusBadRequest)
		return
	}
	name := r.Form.Get("name")
	name = strings.TrimSpace(name)
	if name == "" {
		http.Error(w, "Missing 'name' form parameter", http.StatusBadRequest)
		return
	}
	if !isValidPlayerName(name) {
		http.Error(w, "Invalid username", http.StatusBadRequest)
		return
	}
	playerId := generatePlayerId()
	if err := s.playerStore.Login(r.Context(), playerId, name); err != nil {
		infoLog.Printf("Rejected login for player %s: %s", name, err)
		http.Error(w, "Cannot log in right now", http.StatusPreconditionFailed)
		return
	}
	s.IncCounter("/requests/login/success")
	http.SetCookie(w, makePlayerCookie(playerId, s.config.LoginTTL))
	http.Redirect(w, r, "/hexz", http.StatusSeeOther)
}

func (s *Server) handleHexz(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		s.serveHtmlFile(w, loginHtmlFilename)
		return
	}
	// Prolong cookie ttl.
	http.SetCookie(w, makePlayerCookie(p.Id, s.config.LoginTTL))
	s.serveHtmlFile(w, newGameHtmlFilename)
}

func (s *Server) handleNewGame(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "Player not logged in", http.StatusPreconditionFailed)
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
	gameType := GameType(typeParam)
	singlePlayer := false
	if r.Form.Has("singlePlayer") {
		singlePlayer, err = strconv.ParseBool(r.Form.Get("singlePlayer"))
		if err != nil {
			http.Error(w, "Invalid value for 'singlePlayer'", http.StatusBadRequest)
			return
		}
		if singlePlayer && !supportsSinglePlayer(gameType) {
			http.Error(w, "Single player mode not supported", http.StatusBadRequest)
			return
		}
	}
	game, err := s.startNewGame(p.Name, GameType(typeParam), singlePlayer)
	if err != nil {
		errorLog.Printf("cannot start new game: %s\n", err.Error())
		http.Error(w, "", http.StatusPreconditionFailed)
	}
	s.IncCounter("/games/started")
	http.Redirect(w, r, fmt.Sprintf("/hexz/%s", game.id), http.StatusSeeOther)
}

func (s *Server) lookupPlayerFromCookie(r *http.Request) (Player, error) {
	cookie, err := r.Cookie(playerIdCookieName)
	if err != nil {
		return Player{}, fmt.Errorf("missing cookie")
	}
	return s.playerStore.Lookup(r.Context(), PlayerId(cookie.Value))
}

func (s *Server) handleMove(w http.ResponseWriter, r *http.Request) {
	player, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "unknown player", http.StatusPreconditionFailed)
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	dec := json.NewDecoder(r.Body)
	var req *MoveRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "unmarshal error", http.StatusBadRequest)
		return
	}
	if !req.Type.valid() {
		http.Error(w, "Invalid cell type", http.StatusBadRequest)
		return
	}
	game.sendEvent(ControlEventMove{playerId: player.Id, moveRequest: req})
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "unknown player", http.StatusPreconditionFailed)
		return
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	dec := json.NewDecoder(r.Body)
	var req ResetRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "unmarshal error", http.StatusBadRequest)
		return
	}
	game.sendEvent(ControlEventReset{playerId: p.Id, message: req.Message})
}

func (s *Server) handleUndo(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	dec := json.NewDecoder(r.Body)
	var req UndoRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	game.sendEvent(ControlEventUndo{playerId: p.Id, UndoRequest: req})
}

func (s *Server) handleRedo(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	dec := json.NewDecoder(r.Body)
	var req RedoRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	game.sendEvent(ControlEventRedo{playerId: p.Id, RedoRequest: req})
}

func sendSSEEvent(w http.ResponseWriter, ev ServerEvent) error {
	if _, err := io.WriteString(w, "data: "); err != nil {
		return err
	}
	enc := json.NewEncoder(w)
	if err := enc.Encode(ev); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "\n\n"); err != nil {
		return err
	}
	if f, canFlush := w.(http.Flusher); canFlush {
		f.Flush()
	}
	return nil
}

func (s *Server) handleSSE(w http.ResponseWriter, r *http.Request) {
	s.IncCounter("/requests/sse/incoming")
	// We expect a cookie to identify the p.
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "unknown player", http.StatusPreconditionFailed)
		return
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	serverEventChan, err := game.registerPlayer(p)
	if err != nil {
		http.Error(w, "", http.StatusPreconditionFailed)
		return
	}
	s.IncCounter("/requests/sse/accepted")
	// Headers to establish server-sent events (SSE) communication.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	for {
		select {
		case ev := <-serverEventChan:
			s.IncCounter("/requests/sse/events")
			if err := sendSSEEvent(w, ev); err != nil {
				errorLog.Print("failed to send SSE: ", err)
				return // close the connection, let the client reconnect if it wants to.
			}
		case <-r.Context().Done():
			infoLog.Printf("%s player %s closed SSE channel for game %s", r.RemoteAddr, p.Name, game.id)
			game.unregisterPlayer(p.Id)
			return
		case <-game.ctx.Done():
			// The game is over, time to close the SSE channel.
			infoLog.Printf("Closing SSE channel for player %s in game %s", p.Id, gameId)
			if err := sendSSEEvent(w, ServerEvent{LastEvent: true}); err != nil {
				errorLog.Print("failed to send SSE: ", err)
			}
			return
		}
	}
}

// /hexz/gamez: async request by clients to obtain a list of games to join.
func (s *Server) handleGamez(w http.ResponseWriter, r *http.Request) {
	gameInfos := s.listRecentGames(5)
	json, err := json.Marshal(gameInfos)
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		errorLog.Fatal("Cannot marshal []GameInfo: " + err.Error())
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(json)
}

func (s *Server) handleGame(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	g := s.lookupGame(gameId)
	if g == nil {
		if GameHistoryExists(s.config.GameHistoryRoot, gameId) {
			// This game was played before, redirect to history viewer.
			http.Redirect(w, r, "/hexz/view/"+gameId, http.StatusSeeOther)
			return
		}
		// Offer to start a new game.
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	// Prolong cookie ttl.
	http.SetCookie(w, makePlayerCookie(p.Id, s.config.LoginTTL))
	s.serveHtmlFile(w, gameHtmlFilename)
}

func (s *Server) handleValidMoves(w http.ResponseWriter, r *http.Request) {
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	game := s.lookupGame(gameId)
	if game == nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	dec := json.NewEncoder(w)
	err = dec.Encode(game.validMoves())
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		errorLog.Fatal("Cannot marshal valid moves: " + err.Error())
	}
}

var (
	viewURLPathRE    = regexp.MustCompile(`^(/hexz/view/(?P<gameId>[A-Z]{6}))(/(?P<seqNum>\d+))?$`)
	historyURLPathRE = regexp.MustCompile(`^(/hexz/history/(?P<gameId>[A-Z]{6}))$`)
)

func (s *Server) handleView(w http.ResponseWriter, r *http.Request) {
	groups := viewURLPathRE.FindStringSubmatch(r.URL.Path)
	if groups == nil {
		http.Error(w, "", http.StatusNotFound)
		return
	}
	seqNum := groups[viewURLPathRE.SubexpIndex("seqNum")]
	if seqNum == "" {
		// No move number specified. Redirect to move 0.
		http.Redirect(w, r, fmt.Sprintf("%s/0", groups[1]), http.StatusSeeOther)
		return
	}
	s.serveHtmlFile(w, viewHtmlFilename)
}

func NewGameHistoryResponse(hist *GameHistory) *GameHistoryResponse {
	entries := make([]*GameHistoryResponseEntry, len(hist.Entries))
	for i, e := range hist.Entries {
		entries[i] = &GameHistoryResponseEntry{
			Timestamp:  e.Timestamp,
			EntryType:  e.EntryType,
			Move:       e.Move,
			Board:      e.Board,
			MoveScores: e.MoveScores,
		}
	}
	return &GameHistoryResponse{
		GameId:      hist.Header.GameId,
		PlayerNames: hist.Header.PlayerNames,
		GameType:    hist.Header.GameType,
		Entries:     entries,
	}
}

func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
	groups := historyURLPathRE.FindStringSubmatch(r.URL.Path)
	if groups == nil {
		http.Error(w, "", http.StatusNotFound)
		return
	}
	gameId := groups[historyURLPathRE.SubexpIndex("gameId")]
	hist, err := s.readGameHistoryFromFile(gameId)
	if err != nil {
		http.Error(w, "", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	var z io.Writer = w
	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
		w.Header().Set("Content-Encoding", "gzip")
		gz := gzip.NewWriter(w)
		defer gz.Close()
		z = gz
	}
	enc := json.NewEncoder(z)
	err = enc.Encode(NewGameHistoryResponse(hist))
	if err != nil {
		http.Error(w, "", http.StatusInternalServerError)
		errorLog.Fatalf("Failed to marshal history response: %s", err)
	}
}

func (s *Server) handleStatusz(w http.ResponseWriter, r *http.Request) {
	var resp StatuszResponse

	s.ongoingGamesMut.Lock()
	resp.NumOngoingGames = len(s.ongoingGames)
	s.ongoingGamesMut.Unlock()

	if ps, ok := s.playerStore.(*InMemoryPlayerStore); ok {
		n := ps.NumPlayers()
		resp.NumLoggedInPlayers = &n
	}

	s.countersMut.Lock()
	counters := make([]StatuszCounter, len(s.counters))
	i := 0
	for _, c := range s.counters {
		counters[i] = StatuszCounter{Name: c.Name(), Value: c.Value()}
		i++
	}
	s.countersMut.Unlock()
	sort.Slice(counters, func(i, j int) bool {
		return counters[i].Name < counters[j].Name
	})
	resp.Counters = counters

	// Distributions. Copy them under the mutex, then prepare JSON structs.
	s.distribMut.Lock()
	distribCopies := make([]*Distribution, len(s.distrib))
	j := 0
	for _, d := range s.distrib {
		distribCopies[j] = d.Copy()
		j++
	}
	s.distribMut.Unlock()
	distribs := make([]*StatuszDistrib, len(s.distrib))
	for i, d := range distribCopies {
		distribs[i] = &StatuszDistrib{
			Name:    d.name,
			Buckets: []StatuszDistribBucket{},
		}
		for j := range d.counts {
			if d.counts[j] == 0 {
				continue
			}
			var b StatuszDistribBucket
			b.Count = d.counts[j]
			if j == 0 {
				b.Lower = d.min
				b.Upper = d.upperBounds[j]
			} else if j == len(d.counts)-1 {
				b.Lower = d.upperBounds[j-1]
				b.Upper = d.max
			} else {
				b.Lower = d.upperBounds[j-1]
				b.Upper = d.upperBounds[j]
			}
			distribs[i].Buckets = append(distribs[i].Buckets, b)
		}
	}
	sort.Slice(distribs, func(i, j int) bool { return distribs[i].Name < distribs[j].Name })
	resp.Distributions = distribs

	resp.Started = s.started
	uptime := time.Since(s.started)
	resp.UptimeSeconds = int(uptime.Seconds())
	resp.Uptime = uptime.String()

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(resp); err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		errorLog.Printf("Failed to marshal /statusz response: %s", err)
	}
}

func (s *Server) defaultHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	if isFavicon(r.URL.Path) {
		s.IncCounter("/requests/favicon")
		ico, err := s.readStaticResource(path.Join("images", path.Base(r.URL.Path)))
		if err != nil {
			http.Error(w, "favicon not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write(ico)
		return
	}
	s.IncCounter("/requests/other")
	http.Error(w, "", http.StatusNotFound)
}

func (s *Server) loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.IncCounter("/requests/total")
		infoLog.Printf("Incoming request: %s %s %s", r.RemoteAddr, r.Method, r.URL.String())
		h.ServeHTTP(w, r)
	})
}

func postHandlerFunc(h http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST")
			http.Error(w, "", http.StatusMethodNotAllowed)
			return
		}
		h.ServeHTTP(w, r)
	})
}

func sha256HexDigest(pass string) string {
	passSha256Bytes := sha256.Sum256([]byte(pass))
	return fmt.Sprintf("%x", passSha256Bytes)
}

func isLocalAddr(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback()
}

func (s *Server) basicAuthHandlerFunc(h http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if isLocalAddr(r.RemoteAddr) {
				// No authentication required
				s.IncCounter("/auth/granted/local")
				h(w, r)
				return
			}
			if s.config.AuthTokenSha256 == "" {
				// No auth token: only local access is allowed.
				s.IncCounter("/auth/rejected/nonlocal")
				http.Error(w, "", http.StatusForbidden)
				return
			}
			_, pass, ok := r.BasicAuth()
			passSha256 := sha256HexDigest(pass)
			rejected := true
			if !ok {
				s.IncCounter("/auth/rejected/missing_token")
			} else if passSha256 != s.config.AuthTokenSha256 {
				s.IncCounter("/auth/rejected/bad_passwd")
			} else {
				rejected = false
			}
			if rejected {
				w.Header().Set("WWW-Authenticate", `Basic realm="restricted", charset="UTF-8"`)
				http.Error(w, "", http.StatusUnauthorized)
				return
			}
			s.IncCounter("/auth/granted/basic_auth")
			h(w, r)
		})
}

func (s *Server) createMux() *http.ServeMux {
	mux := &http.ServeMux{}
	// Static resources (images, JavaScript, ...) live under DocumentRoot.
	mux.Handle("/hexz/static/", http.StripPrefix("/hexz/static/",
		http.FileServer(http.Dir(s.config.DocumentRoot))))
	// POST method API
	mux.HandleFunc("/hexz/login", postHandlerFunc(s.handleLoginRequest))
	mux.HandleFunc("/hexz/new", postHandlerFunc(s.handleNewGame))
	mux.HandleFunc("/hexz/move/", postHandlerFunc(s.handleMove))
	mux.HandleFunc("/hexz/reset/", postHandlerFunc(s.handleReset))
	mux.HandleFunc("/hexz/undo/", postHandlerFunc(s.handleUndo))
	mux.HandleFunc("/hexz/redo/", postHandlerFunc(s.handleRedo))
	// Server-sent Event handling
	mux.HandleFunc("/hexz/sse/", s.handleSSE)

	mux.HandleFunc("/hexz/rules", func(w http.ResponseWriter, r *http.Request) {
		s.serveHtmlFile(w, rulesHtmlFilename)
	})
	// GET method API
	mux.HandleFunc("/hexz", s.handleHexz)
	mux.HandleFunc("/hexz/gamez", s.handleGamez)
	mux.HandleFunc("/hexz/view/", s.handleView)
	mux.HandleFunc("/hexz/history/", s.handleHistory)
	mux.HandleFunc("/hexz/moves/", s.handleValidMoves)
	mux.HandleFunc("/hexz/", s.handleGame) // /hexz/<GameId>
	// Technical services
	mux.Handle("/statusz", s.basicAuthHandlerFunc(s.handleStatusz))

	mux.HandleFunc("/", s.defaultHandler)

	return mux
}

func (s *Server) Serve() {
	addr := fmt.Sprintf("%s:%d", s.config.ServerHost, s.config.ServerPort)
	mux := s.createMux()
	srv := &http.Server{
		Addr:    addr,
		Handler: s.loggingHandler(mux),
	}

	// Quick sanity check that we have access to the game HTML file.
	if _, err := s.readStaticResource(gameHtmlFilename); err != nil {
		errorLog.Fatal("Cannot load game HTML: ", err)
	}

	infoLog.Printf("Stateful server listening on %s", addr)

	if s.config.TlsCertChain != "" && s.config.TlsPrivKey != "" {
		errorLog.Fatal(srv.ListenAndServeTLS(s.config.TlsCertChain, s.config.TlsPrivKey))
	}
	errorLog.Fatal(srv.ListenAndServe())
}
