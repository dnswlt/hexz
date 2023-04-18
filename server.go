package hexz

import (
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
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

type tok struct{}

type ServerConfig struct {
	ServerAddress     string
	ServerPort        int
	DocumentRoot      string
	PlayerRemoveDelay time.Duration
	LoginTtl          time.Duration
	CompThinkTime     time.Duration
	AuthTokenSha256   string // Used in http Basic authentication for /statusz. Must be a SHA256 checksum.

	TlsCertChain string
	TlsPrivKey   string
	DebugMode    bool
}

var (
	// Regexp used to validate player names.
	playernameRegexp = regexp.MustCompile(`^[\p{Latin}0-9_.-]+$`)
)

type Server struct {
	// Contains all ongoing games, mapped by their ID.
	ongoingGames    map[string]*GameHandle
	ongoingGamesMut sync.Mutex

	// Contains all logged in players, mapped by their (cookie) playerId.
	loggedInPlayers    map[PlayerId]*Player
	loggedInPlayersMut sync.Mutex

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

func NewServer(cfg *ServerConfig) *Server {
	s := &Server{
		ongoingGames:    make(map[string]*GameHandle),
		loggedInPlayers: make(map[PlayerId]*Player),
		config:          cfg,
		counters:        make(map[string]*Counter),
		distrib:         make(map[string]*Distribution),
		started:         time.Now(),
	}
	s.InitCounters()
	return s
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
	singlePlayer bool              // If true, only player 1 is human, the rest are computer-controlled.
	controlEvent chan ControlEvent // The channel to communicate with the game coordinating goroutine.
	done         chan tok          // Closed by the game master goroutine when it is done.
}

// Player has JSON annotations for serialization to disk.
// It is not used in the public API.
type Player struct {
	Id         PlayerId  `json:"id"`
	Name       string    `json:"name"`
	LastActive time.Time `json:"lastActive"`
}

func (s *Server) lookupPlayer(playerId PlayerId) (Player, bool) {
	s.loggedInPlayersMut.Lock()
	defer s.loggedInPlayersMut.Unlock()

	p, ok := s.loggedInPlayers[playerId]
	if !ok {
		return Player{}, false
	}
	p.LastActive = time.Now()
	return *p, true
}

func (s *Server) loginPlayer(playerId PlayerId, name string) bool {
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
	player    Player
	replyChan chan chan ServerEvent
}

type ControlEventUnregister struct {
	playerId PlayerId
}

type ControlEventMove struct {
	playerId PlayerId
	MoveRequest
	confidence float64 // In [0..1], can be populated by CPU players to express their confidence in winning.
}

type ControlEventReset struct {
	playerId PlayerId
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

func (s *Server) readFile(filename string) ([]byte, error) {
	s.IncCounter("/storage/files/readfile")
	return os.ReadFile(path.Join(s.config.DocumentRoot, filename))
}

// Generates a random 128-bit hex string representing a player ID.
func generatePlayerId() PlayerId {
	p := make([]byte, 16)
	crand.Read(p)
	return PlayerId(hex.EncodeToString(p))
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

// Looks up the game ID from the URL path.
func gameIdFromPath(path string) string {
	pathSegs := strings.Split(path, "/")
	l := len(pathSegs)
	if l >= 2 && pathSegs[1] == "hexz" {
		return pathSegs[l-1]
	}
	return ""
}

func (s *Server) startNewGame(host string, gameType GameType, singlePlayer bool) (*GameHandle, error) {
	// Try a few times to find an unused game Id, else give up.
	// (I don't like forever loops... 100 attempts is plenty.)
	var game *GameHandle
	for i := 0; i < 100; i++ {
		id := generateGameId()
		s.ongoingGamesMut.Lock()
		if _, ok := s.ongoingGames[id]; !ok {
			game = &GameHandle{
				id:           id,
				started:      time.Now(),
				gameType:     gameType,
				host:         host,
				singlePlayer: singlePlayer,
				controlEvent: make(chan ControlEvent),
				done:         make(chan tok),
			}
			s.ongoingGames[id] = game
		}
		s.ongoingGamesMut.Unlock()
		if game != nil {
			m := NewGameMaster(s, game)
			go m.Run()
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
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusBadRequest)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Invalid form", http.StatusBadRequest)
		return
	}
	name := r.Form.Get("name")
	name = strings.TrimSpace(name)
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
	s.IncCounter("/requests/login/success")
	http.SetCookie(w, makePlayerCookie(playerId, s.config.LoginTtl))
	http.Redirect(w, r, "/hexz", http.StatusSeeOther)
}

func (s *Server) handleHexz(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
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
	// Prolong cookie ttl.
	http.SetCookie(w, makePlayerCookie(p.Id, s.config.LoginTtl))
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
		http.Error(w, err.Error(), http.StatusPreconditionFailed)
	}
	s.IncCounter("/games/started")
	http.Redirect(w, r, fmt.Sprintf("/hexz/%s", game.id), http.StatusSeeOther)
}

func (s *Server) validatePostRequest(r *http.Request) (Player, error) {
	if r.Method != http.MethodPost {
		return Player{}, fmt.Errorf("invalid method")
	}
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		return Player{}, fmt.Errorf("invalid method")
	}
	return p, nil
}

func (s *Server) lookupPlayerFromCookie(r *http.Request) (Player, error) {
	cookie, err := r.Cookie(playerIdCookieName)
	if err != nil {
		return Player{}, fmt.Errorf("missing cookie")
	}
	p, ok := s.lookupPlayer(PlayerId(cookie.Value))
	if !ok {
		return Player{}, fmt.Errorf("player not found")
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
	game.sendEvent(ControlEventMove{playerId: player.Id, MoveRequest: req})
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
	s.IncCounter("/requests/sse/incoming")
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
	s.IncCounter("/requests/sse/accepted")
	// Headers to establish server-sent events (SSE) communication.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	for {
		select {
		case ev, ok := <-serverEventChan:
			s.IncCounter("/requests/sse/events")
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
			// log.Printf("Sending %d bytes over SSE", buf.Len())
			fmt.Fprintf(w, "data: %s\n\n", buf.String())
			if f, canFlush := w.(http.Flusher); canFlush {
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

// /hexz/gamez: async request by clients to obtain a list of games to join.
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
	p, err := s.lookupPlayerFromCookie(r)
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
	// Prolong cookie ttl.
	http.SetCookie(w, makePlayerCookie(p.Id, s.config.LoginTtl))
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
				b.Lower = math.Inf(-1)
				b.Upper = d.upperBounds[j]
			} else if j == len(d.counts)-1 {
				b.Lower = d.upperBounds[j-1]
				b.Upper = math.Inf(1)
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
		http.Error(w, "Serialization error", http.StatusInternalServerError)
		panic(fmt.Sprintf("Cannot serialize my own structs?! %s", err))
	}
}

func (s *Server) defaultHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	if isFavicon(r.URL.Path) {
		s.IncCounter("/requests/favicon")
		ico, err := s.readFile(path.Join("images", path.Base(r.URL.Path)))
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

func (s *Server) saveUserDatabase(players []Player) {
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
	s.IncCounter("/storage/userdb/saved")
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
		del := []*Player{}
		for _, p := range s.loggedInPlayers {
			if p.LastActive.Before(logoutThresh) {
				del = append(del, p)
			} else if p.LastActive.After(lastIteration) {
				activity = true
			}
		}
		for _, p := range del {
			delete(s.loggedInPlayers, p.Id)
		}
		s.loggedInPlayersMut.Unlock()
		// Do I/O outside the mutex.
		for _, p := range del {
			log.Printf("Logged out player %s(%s)", p.Name, p.Id)
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
			s.saveUserDatabase(players)
		}
		lastIteration = now
	}
}

func (s *Server) loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// t := time.Now().Format("2006-01-02 15:04:05.999Z07:00")
		s.IncCounter("/requests/total")
		log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL.String())
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
				http.Error(w, "Forbidden", http.StatusForbidden)
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
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}
			s.IncCounter("/auth/granted/basic_auth")
			h(w, r)
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
	mux.Handle("/statusz", s.basicAuthHandlerFunc(s.handleStatusz))
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
