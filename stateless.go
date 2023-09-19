package hexz

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
)

// This file contains the implementation of the stateless hexz game server.
// It can be used in "serverless" contexts (e.g. Cloud Run) where the server
// is only guaranteed to run while it is handling a request.
//
// See server.go for the stateful implementation.

type StatelessServer struct {
	config      *ServerConfig
	playerStore PlayerStore
	gameStore   GameStore
}

func NewStatelessServer(config *ServerConfig) (*StatelessServer, error) {
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     config.RedisAddr,
		LoginTTL: config.LoginTTL,
		GameTTL:  config.InactivityTimeout,
	})
	if err != nil {
		return nil, err
	}

	if err != nil {
		return nil, err
	}

	return &StatelessServer{
		playerStore: &RemotePlayerStore{rc},
		config:      config,
		gameStore:   rc,
	}, nil
}

func (s *StatelessServer) loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		infoLog.Printf("Incoming request: %s %s %s", r.RemoteAddr, r.Method, r.URL.String())
		h.ServeHTTP(w, r)
	})
}

func (s *StatelessServer) readStaticResource(filename string) ([]byte, error) {
	if strings.Contains(filename, "..") {
		return nil, fmt.Errorf("refusing to read %q", filename)
	}
	return os.ReadFile(path.Join(s.config.DocumentRoot, filename))
}

func (s *StatelessServer) lookupPlayerFromCookie(r *http.Request) (Player, error) {
	cookie, err := r.Cookie(playerIdCookieName)
	if err != nil {
		return Player{}, fmt.Errorf("missing cookie")
	}
	return s.playerStore.Lookup(r.Context(), PlayerId(cookie.Value))
}

func (s *StatelessServer) startNewGame(ctx context.Context, p *Player, gameType GameType, singlePlayer bool) (string, error) {
	engineState, err := NewGameEngine(gameType).Encode()
	if err != nil {
		return "", err
	}
	// Try to find an unused gameId.
	for i := 0; i < 100; i++ {
		gameState := &pb.GameState{
			GameId:      GenerateGameId(),
			Players:     []*pb.Player{}, // Players are registed in handleSSE.
			EngineState: engineState,
		}
		// Keep the game in Redis for 24 hours max.
		var ok bool
		ok, err = s.gameStore.StoreNewGame(ctx, gameState)
		if err != nil {
			return "", err
		}
		if ok {
			return gameState.GameId, nil
		}
	}
	return "", fmt.Errorf("cannot find unused gameId")
}

// Sends the contents of filename to the ResponseWriter.
func (s *StatelessServer) serveHtmlFile(w http.ResponseWriter, filename string) {
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

func (s *StatelessServer) handleLoginRequest(w http.ResponseWriter, r *http.Request) {
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
	http.SetCookie(w, makePlayerCookie(playerId, s.config.LoginTTL))
	http.Redirect(w, r, "/hexz", http.StatusSeeOther)
}

func (s *StatelessServer) handleNewGame(w http.ResponseWriter, r *http.Request) {
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
	gameId, err := s.startNewGame(r.Context(), &p, GameType(typeParam), singlePlayer)
	if err != nil {
		errorLog.Printf("cannot start new game: %s\n", err)
		http.Error(w, "", http.StatusPreconditionFailed)
	}
	http.Redirect(w, r, fmt.Sprintf("/hexz/%s", gameId), http.StatusSeeOther)
}

func (s *StatelessServer) handleHexz(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		s.serveHtmlFile(w, loginHtmlFilename)
		return
	}
	// Prolong cookie ttl.
	http.SetCookie(w, makePlayerCookie(p.Id, s.config.LoginTTL))
	s.serveHtmlFile(w, newGameHtmlFilename)
}

func (s *StatelessServer) handleGame(w http.ResponseWriter, r *http.Request) {
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
	if _, err := s.gameStore.LookupGame(r.Context(), gameId); err != nil {
		// Game does not exist: offer to start a new game.
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	// Game exists, serve HTML and prolong cookie ttl.
	http.SetCookie(w, makePlayerCookie(p.Id, s.config.LoginTTL))
	s.serveHtmlFile(w, gameHtmlFilename)
}

const (
	sseEventPlayerJoined = "player joined"
)

func (s *StatelessServer) handleSSE(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "Player not logged in", http.StatusPreconditionFailed)
		return
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	gameState, err := s.gameStore.LookupGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	ge, err := DecodeGameEngine(gameState)
	if err != nil {
		errorLog.Print("Cannot decode game state: ", err)
		http.Error(w, "invalid game state", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	// If there is a slot in the game left, add this player.
	if len(gameState.Players) < ge.NumPlayers() {
		gameState.Players = append(gameState.Players, &pb.Player{
			Id:   string(p.Id),
			Name: p.Name,
		})
		if err := s.gameStore.UpdateGame(r.Context(), gameState); err != nil {
			errorLog.Printf("Cannot store updated game state: %s", err)
			return
		}
		// Tell others we've joined.
		s.gameStore.Publish(r.Context(), gameId, sseEventPlayerJoined+":"+p.Name)
	}
	// Send initial ServerEvent to the player.
	err = sendSSEEvent(w, ServerEvent{
		Timestamp:     time.Now(),
		Board:         ge.Board().ViewFor(gameState.PlayerNum(string(p.Id))),
		Role:          gameState.PlayerNum(string(p.Id)),
		PlayerNames:   gameState.PlayerNames(),
		Announcements: []string{fmt.Sprintf("Welcome %s!", p.Name)},
		GameInfo: &ServerEventGameInfo{
			ValidCellTypes: ge.ValidCellTypes(),
			GameType:       ge.GameType(),
		},
	})
	if err != nil {
		errorLog.Printf("Cannot send initial ServerEvent: %s", err)
		return
	}
	// Process events from Redis.
	eventCh := make(chan string)
	go s.gameStore.Subscribe(r.Context(), gameId, eventCh)
	for {
		select {
		case e, ok := <-eventCh:
			if !ok {
				errorLog.Printf("Redis pubsub closed for player %s", p.Name)
				return
			}
			ev, msg, _ := strings.Cut(e, ":")
			switch ev {
			case sseEventPlayerJoined:
				infoLog.Printf("[%s] A new player joined: %s", p.Name, msg)
				gameState, err := s.gameStore.LookupGame(r.Context(), gameId)
				if err != nil {
					errorLog.Printf("Cannot lookup game state for game %s: %s", gameId, err)
					return
				}
				ge, err = DecodeGameEngine(gameState)
				if err != nil {
					errorLog.Printf("Cannot decode game state for game %s: %s", gameId, err)
					return
				}
				pNum := gameState.PlayerNum(string(p.Id))
				err = sendSSEEvent(w, ServerEvent{
					Timestamp:     time.Now(),
					Board:         ge.Board().ViewFor(pNum),
					Role:          pNum,
					PlayerNames:   gameState.PlayerNames(),
					Announcements: []string{"New player joined!"},
					GameInfo: &ServerEventGameInfo{
						ValidCellTypes: ge.ValidCellTypes(),
						GameType:       ge.GameType(),
					},
				})
				if err != nil {
					errorLog.Printf("Cannot send ServerEvent: %s", err)
					return
				}
			default:
				infoLog.Printf("[%s] Received event: %s", p.Name, e)
			}
		case <-r.Context().Done():
			infoLog.Printf("SSE connection closed for player %s", p.Name)
			return
		}
	}
}

func (s *StatelessServer) defaultHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/hexz", http.StatusSeeOther)
		return
	}
	if isFavicon(r.URL.Path) {
		ico, err := s.readStaticResource(path.Join("images", path.Base(r.URL.Path)))
		if err != nil {
			http.Error(w, "favicon not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write(ico)
		return
	}
	http.Error(w, "", http.StatusNotFound)
}

func (s *StatelessServer) createMux() *http.ServeMux {
	// TODO: the handler functions are copy&pasted from server.go. We should
	// refactor them into a common place.

	mux := &http.ServeMux{}
	// Static resources (images, JavaScript, ...) live under DocumentRoot.
	mux.Handle("/hexz/static/", http.StripPrefix("/hexz/static/",
		http.FileServer(http.Dir(s.config.DocumentRoot))))
	// POST method API
	mux.HandleFunc("/hexz/login", postHandlerFunc(s.handleLoginRequest))
	mux.HandleFunc("/hexz/new", postHandlerFunc(s.handleNewGame))
	// mux.HandleFunc("/hexz/move/", postHandlerFunc(s.handleMove))
	// mux.HandleFunc("/hexz/reset/", postHandlerFunc(s.handleReset))
	// mux.HandleFunc("/hexz/undo/", postHandlerFunc(s.handleUndo))
	// mux.HandleFunc("/hexz/redo/", postHandlerFunc(s.handleRedo))
	// Server-sent Event handling
	mux.HandleFunc("/hexz/sse/", s.handleSSE)

	// GET method API
	mux.HandleFunc("/hexz", s.handleHexz)
	// mux.HandleFunc("/hexz/gamez", s.handleGamez)
	// mux.HandleFunc("/hexz/view/", s.handleView)
	// mux.HandleFunc("/hexz/history/", s.handleHistory)
	// mux.HandleFunc("/hexz/moves/", s.handleValidMoves)
	mux.HandleFunc("/hexz/", s.handleGame) // /hexz/<GameId>
	// Technical services
	// mux.Handle("/statusz", s.basicAuthHandlerFunc(s.handleStatusz))

	mux.HandleFunc("/", s.defaultHandler)

	return mux
}

func (s *StatelessServer) Serve() {
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

	infoLog.Printf("Stateless server listening on %s", addr)

	errorLog.Fatal(srv.ListenAndServe())
}
