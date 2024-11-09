package hexz

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/lpar/gzipped/v2"

	pb "github.com/dnswlt/hexz/hexzpb"
	"github.com/dnswlt/hexz/hlog"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

// This file contains the implementation of the stateless hexz game server.
// It can be used in "serverless" contexts (e.g. Cloud Run) where the server
// is only guaranteed to run while it is handling a request.
//
// See server.go for the stateful implementation.

type StatelessServer struct {
	config      *ServerConfig
	playerStore PlayerStore
	dbStore     DatabaseStore
	gameStore   GameStore
}

func NewStatelessServer(config *ServerConfig, playerStore PlayerStore, gameStore GameStore, dbStore DatabaseStore) (*StatelessServer, error) {
	return &StatelessServer{
		config:      config,
		playerStore: playerStore,
		gameStore:   gameStore,
		dbStore:     dbStore,
	}, nil
}

func (s *StatelessServer) loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.config.DebugMode {
			hlog.Infof("Incoming request: %s %s %s", r.RemoteAddr, r.Method, r.URL.String())
		}
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

// Stores a new game in the game store and returns the new game ID.
func (s *StatelessServer) startNewGame(ctx context.Context, p *Player, gameType GameType, singlePlayer bool) (string, error) {
	engineState, err := NewGameEngine(gameType).Encode()
	if err != nil {
		return "", err
	}
	players := []*pb.Player{{Id: string(p.Id), Name: p.Name}}
	if singlePlayer {
		players = append(players, &pb.Player{Id: "CPU", Name: "CPU"})
	}
	// Try to find an unused gameId. This loop should usually exit after the first iteration.
	var gameState *pb.GameState
	for i := 0; i < 100; i++ {
		gs := &pb.GameState{
			GameInfo: &pb.GameInfo{
				Id:        GenerateGameId(),
				Host:      p.Name,
				Started:   tpb.Now(),
				Type:      string(gameType),
				CpuPlayer: singlePlayer,
			},
			Players:     players, // More players are registed in handleSSE.
			EngineState: engineState,
		}
		if ok, err := s.gameStore.StoreNewGame(ctx, gs); err != nil {
			return "", err
		} else if ok {
			gameState = gs
			break
		}
	}
	if gameState == nil {
		return "", fmt.Errorf("cannot find unused gameId")
	}
	if s.dbStore != nil {
		if err := s.dbStore.StoreGame(ctx, string(p.Id), gameState); err != nil {
			hlog.Errorf("Cannot store game %s in database: %s", gameState.GameInfo.Id, err)
		}
	}
	return gameState.GameInfo.Id, nil
}

// Sends the contents of filename to the ResponseWriter.
func (s *StatelessServer) serveHtmlFile(w http.ResponseWriter, filename string) {
	if path.Ext(filename) != ".html" || strings.Contains(filename, "..") || strings.Contains(filename, "/") {
		// It's a programming error to call this method with non-html files.
		hlog.Fatalf("Not a valid HTML file: %q\n", filename)
	}
	p := path.Join(s.config.DocumentRoot, filename)
	html, err := os.ReadFile(p)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			// We should only call this method for existing html files.
			hlog.Fatalf("Cannot read %s: %s", p, err.Error())
		} else {
			// Whatever might cause us to fail reading our own files...
			hlog.Errorf("Could not read existing file %s: %s", p, err.Error())
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
		hlog.Infof("Rejected login for player %s: %s", name, err)
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
		hlog.Errorf("Cannot start new game: %s\n", err)
		http.Error(w, "", http.StatusPreconditionFailed)
		return
	}
	http.Redirect(w, r, fmt.Sprintf("/hexz/%s", gameId), http.StatusSeeOther)
}

func (s *StatelessServer) handleReset(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "unknown player", http.StatusForbidden)
		return
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "invalid game ID", http.StatusBadRequest)
		return
	}
	gameState, ge, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "game does not exist", http.StatusNotFound)
		return
	}
	dec := json.NewDecoder(r.Body)
	var req ResetRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "unmarshal error", http.StatusBadRequest)
		return
	}
	if gameState.PlayerNum(string(p.Id)) <= 0 {
		http.Error(w, "only players can reset a game", http.StatusForbidden)
		return
	}
	ge.Reset()
	state, _ := ge.Encode()
	gameState.EngineState = state
	if err := s.gameStore.UpdateGame(r.Context(), gameState); err != nil {
		http.Error(w, "cannot update game", http.StatusInternalServerError)
		hlog.Errorf("Cannot update game %s: %s", gameId, err)
		return
	}
	// Inform other players.
	s.gameStore.Publish(r.Context(), gameId, sseEventGameUpdated)
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

func (s *StatelessServer) handleGamez(w http.ResponseWriter, r *http.Request) {
	gameInfos, err := s.gameStore.ListRecentGames(r.Context(), 10)
	if err != nil {
		http.Error(w, "list recent games", http.StatusInternalServerError)
		hlog.Errorf("Cannot list recent games: %s", err)
		return
	}
	resp := make([]*GameInfo, len(gameInfos))
	for i, g := range gameInfos {
		resp[i] = &GameInfo{
			Id:       g.Id,
			Host:     g.Host,
			Started:  g.Started.AsTime(),
			GameType: GameType(g.Type),
		}
	}
	json, err := json.Marshal(resp)
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		hlog.Errorf("JSON marshal error: %s", err)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(json)
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

func (s *StatelessServer) handleWASMStats(w http.ResponseWriter, r *http.Request) {
	_, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "missing player cookie", http.StatusBadRequest)
		return
	}
	_, err = gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		hlog.Errorf("Cannot read request body: %s", err)
		http.Error(w, "", http.StatusInternalServerError)
	}
	var req WASMStatsRequest
	if err = json.Unmarshal(body, &req); err != nil {
		http.Error(w, "unmarshal error", http.StatusBadRequest)
		return
	}
	if s.dbStore != nil {
		s.dbStore.InsertStats(r.Context(), &req)
	}
	hlog.Infof("CPU stats: %s", string(body))
}

// Download the full game state as an encoded protobuf. This is used to run a CPU player in
// WASM in the user's browser.
func (s *StatelessServer) handleState(w http.ResponseWriter, r *http.Request) {
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
	gameState, err := s.gameStore.LookupGame(r.Context(), gameId)
	if err != nil {
		// Game does not exist: offer to start a new game.
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	if gameState.PlayerNum(string(p.Id)) == 0 {
		http.Error(w, "Only players can request the game state", http.StatusForbidden)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	// Game state can change at any time, so don't cache it.
	w.Header().Set("Cache-Control", "no-cache")
	var enc *json.Encoder
	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
		w.Header().Set("Content-Encoding", "gzip")
		gw := gzip.NewWriter(w)
		defer gw.Close()
		enc = json.NewEncoder(gw)
	} else {
		enc = json.NewEncoder(w)
	}
	encodedGameState, err := proto.Marshal(gameState)
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		hlog.Errorf("Cannot marshal GameState: %s", err.Error())
	}
	enc.Encode(GameStateResponse{
		GameId:           gameId,
		EncodedGameState: encodedGameState,
	})
}

const (
	sseEventPlayerJoined = "player.joined"
	sseEventGameUpdated  = "game.updated"
)

func (s *StatelessServer) loadGame(ctx context.Context, gameId string) (*pb.GameState, GameEngine, error) {
	gameState, err := s.gameStore.LookupGame(ctx, gameId)
	if err != nil {
		return nil, nil, err
	}
	ge, err := DecodeGameEngine(gameState.EngineState)
	if err != nil {
		hlog.Errorf("Cannot decode game engine for game %s: %s", gameId, err)
		return nil, nil, err
	}
	return gameState, ge, nil
}

func (s *StatelessServer) handleMove(w http.ResponseWriter, r *http.Request) {
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
	gameState, ge, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	// Get move request.
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
	// Is it the player's turn?
	pNum := gameState.PlayerNum(string(p.Id))
	if ge.Board().Turn != pNum && !gameState.GetGameInfo().GetCpuPlayer() {
		http.Error(w, "player cannot make a move", http.StatusPreconditionFailed)
		return
	} else if pNum == 1 && gameState.GetGameInfo().GetCpuPlayer() && ge.Board().Turn == 2 {
		pNum = 2 // Pretend to be the CPU player.
	}

	if !ge.MakeMove(GameEngineMove{
		PlayerNum: pNum,
		Move:      req.Move,
		Row:       req.Row,
		Col:       req.Col,
		CellType:  req.Type,
	}) {
		http.Error(w, "invalid move", http.StatusBadRequest)
		return
	}
	// Store new game state and notify other players.
	enc, _ := ge.Encode()
	gameState.EngineState = enc
	if err := s.gameStore.UpdateGame(r.Context(), gameState); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %s", gameId, err)
		return
	}
	if s.dbStore != nil {
		if err := s.dbStore.InsertHistory(r.Context(), "move", gameId, gameState); err != nil {
			hlog.Errorf("Cannot add history entry for game %s in database: %s", gameState.GameInfo.Id, err)
		}
	}
	s.gameStore.Publish(r.Context(), gameId, sseEventGameUpdated)
}

func (s *StatelessServer) handleUndo(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	if s.dbStore == nil {
		http.Error(w, "Undo not supported", http.StatusNotImplemented)
		return
	}
	currentGameState, _, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	if currentGameState.PlayerNum(string(p.Id)) == 0 {
		http.Error(w, "Only players can undo a move", http.StatusForbidden)
		return
	}
	prevGameState, err := s.dbStore.PreviousGameState(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No previous game state", http.StatusNotFound)
		return
	}
	if err := s.gameStore.UpdateGame(r.Context(), prevGameState); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %s", gameId, err)
		return
	}
	if err := s.dbStore.InsertHistory(r.Context(), "undo", gameId, nil); err != nil {
		hlog.Errorf("Cannot add history entry for game %s in database: %s", gameId, err)
	}
	s.gameStore.Publish(r.Context(), gameId, sseEventGameUpdated)
}

func (s *StatelessServer) handleRedo(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	gameId, err := gameIdFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	if s.dbStore == nil {
		http.Error(w, "Redo not supported", http.StatusNotImplemented)
		return
	}
	currentGameState, _, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	if currentGameState.PlayerNum(string(p.Id)) == 0 {
		http.Error(w, "Only players can redo a move", http.StatusForbidden)
		return
	}
	nextGameState, err := s.dbStore.NextGameState(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No next game state", http.StatusNotFound)
		return
	}
	if err := s.gameStore.UpdateGame(r.Context(), nextGameState); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %s", gameId, err)
		return
	}
	if err := s.dbStore.InsertHistory(r.Context(), "redo", gameId, nil); err != nil {
		hlog.Errorf("Cannot add history entry for game %s in database: %s", gameId, err)
	}
	s.gameStore.Publish(r.Context(), gameId, sseEventGameUpdated)
}

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
	gameState, ge, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	pNum := gameState.PlayerNum(string(p.Id))
	// If there is a slot in the game left, add this player.
	if pNum == 0 && len(gameState.Players) < ge.NumPlayers() {
		gameState.Players = append(gameState.Players, &pb.Player{
			Id:   string(p.Id),
			Name: p.Name,
		})
		if err := s.gameStore.UpdateGame(r.Context(), gameState); err != nil {
			hlog.Errorf("Cannot store updated game state: %s", err)
			return
		}
		if s.dbStore != nil {
			if err := s.dbStore.InsertHistory(r.Context(), "join", gameId, gameState); err != nil {
				hlog.Errorf("Cannot add history entry for game %s in database: %s", gameState.GameInfo.Id, err)
			}
		}
		pNum = gameState.PlayerNum(string(p.Id))
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
			ValidCellTypes:      ge.ValidCellTypes(),
			GameType:            ge.GameType(),
			ClientSideCPUPlayer: gameState.GameInfo.CpuPlayer,
		},
		DisableUndo: s.config.DisableUndo,
	})
	if err != nil {
		hlog.Errorf("Cannot send initial ServerEvent: %s", err)
		return
	}
	// Process events from Redis.
	eventCh := make(chan string)
	go s.gameStore.Subscribe(r.Context(), gameId, eventCh)
	for {
		select {
		case e, ok := <-eventCh:
			if !ok {
				hlog.Errorf("Pubsub closed for player %s", p.Name)
				return
			}
			ev, msg, _ := strings.Cut(e, ":") // Event format: <eventType>:<msg>
			switch ev {
			case sseEventPlayerJoined:
				hlog.Infof("[%s/%s] A new player joined: %s", gameId, p.Name, msg)
				gameState, ge, err := s.loadGame(r.Context(), gameId)
				if err != nil {
					hlog.Errorf("Cannot load ongoing game %s: %s", gameId, err)
					return
				}
				err = sendSSEEvent(w, ServerEvent{
					Timestamp:     time.Now(),
					Board:         ge.Board().ViewFor(pNum),
					Role:          pNum,
					PlayerNames:   gameState.PlayerNames(),
					Announcements: []string{"New player " + msg + " joined!"},
				})
				if err != nil {
					hlog.Errorf("Cannot send ServerEvent: %s", err)
					return
				}
			case sseEventGameUpdated:
				gameState, ge, err := s.loadGame(r.Context(), gameId)
				if err != nil {
					hlog.Errorf("Cannot load ongoing game %s: %s", gameId, err)
					return
				}
				var winner int
				var announcements []string
				if ge.IsDone() {
					winner = ge.Winner()
					if winner > 0 {
						announcements = append(announcements,
							fmt.Sprintf("&#127942; &#127942; &#127942; %s won &#127942; &#127942; &#127942;",
								gameState.PlayerNames()[winner-1]))
					} else {
						announcements = append(announcements, "The game is a draw!")
					}
				}
				err = sendSSEEvent(w, ServerEvent{
					Timestamp:     time.Now(),
					Board:         ge.Board().ViewFor(pNum),
					Role:          pNum,
					PlayerNames:   gameState.PlayerNames(),
					Winner:        winner,
					Announcements: announcements,
				})
				if err != nil {
					hlog.Errorf("Cannot send ServerEvent: %s", err)
					return
				}
			default:
				hlog.Infof("[%s/%s] Received unknown event: %s", gameId, p.Name, e)
			}
		case <-r.Context().Done():
			hlog.Infof("SSE connection closed for player %s", p.Name)
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
	// TODO: Several generic handler functions are copy&pasted from server.go. We should
	// refactor them into a common place.

	mux := &http.ServeMux{}
	// Static resources (images, JavaScript, ...) live under DocumentRoot.
	mux.Handle("/hexz/static/", http.StripPrefix("/hexz/static/",
		gzipped.FileServer(gzipped.Dir(s.config.DocumentRoot))))
	// POST method API
	mux.HandleFunc("/hexz/login", postHandlerFunc(s.handleLoginRequest))
	mux.HandleFunc("/hexz/new", postHandlerFunc(s.handleNewGame))
	mux.HandleFunc("/hexz/move/", postHandlerFunc(s.handleMove))
	mux.HandleFunc("/hexz/reset/", postHandlerFunc(s.handleReset))
	// Methods for CPU player.
	mux.HandleFunc("/hexz/state/", s.handleState)
	mux.HandleFunc("/hexz/wasmstats/", postHandlerFunc(s.handleWASMStats))
	mux.HandleFunc("/hexz/undo/", postHandlerFunc(s.handleUndo))
	mux.HandleFunc("/hexz/redo/", postHandlerFunc(s.handleRedo))
	// Server-sent Event handling
	mux.HandleFunc("/hexz/sse/", s.handleSSE)

	// GET method API
	mux.HandleFunc("/hexz", s.handleHexz)
	mux.HandleFunc("/hexz/gamez", s.handleGamez)
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
		hlog.Fatalf("Cannot load game HTML: %s", err)
	}

	hlog.Infof("Stateless server listening on %s", addr)

	hlog.Fatalf("%s", srv.ListenAndServe())
}
