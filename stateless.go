package hexz

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
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

// StatelessServer is the handle to the "stateless" server implementation.
// Instances should be created using a builder. See NewStatelessServerBuilder.
type StatelessServer struct {
	config          *ServerConfig
	renderer        *Renderer
	playerStore     PlayerStore
	dbStore         DatabaseStore
	gameStore       GameStore
	remoteCPUClient pb.CPUPlayerServiceClient // Only non-nil if a remote CPU addr was configured.
}

type StatelessServerBuilder struct {
	s *StatelessServer
}

func NewStatelessServerBuilder(config *ServerConfig, playerStore PlayerStore, gameStore GameStore, renderer *Renderer) *StatelessServerBuilder {
	return &StatelessServerBuilder{
		s: &StatelessServer{
			config:      config,
			playerStore: playerStore,
			gameStore:   gameStore,
			renderer:    renderer,
		},
	}
}

func (b *StatelessServerBuilder) WithDatabaseStore(dbStore DatabaseStore) *StatelessServerBuilder {
	b.s.dbStore = dbStore
	return b
}

func (b *StatelessServerBuilder) WithCPUPlayerServiceClient(client pb.CPUPlayerServiceClient) *StatelessServerBuilder {
	b.s.remoteCPUClient = client
	return b
}

func (b *StatelessServerBuilder) Build() *StatelessServer {
	s := b.s
	b.s = nil
	return s
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

// prefix adds the configured URL prefix to the given urlPath.
// To run a server under any configured URL path prefix, every
// client-facing URL should be built using this method.
func (s *StatelessServer) prefix(urlPath string) string {
	return urlJoinPath(s.config.URLPathPrefix, urlPath)
}

func (s *StatelessServer) makePlayerCookie(playerId PlayerId, ttl time.Duration) *http.Cookie {
	return &http.Cookie{
		Name:     playerIdCookieName,
		Value:    string(playerId),
		Path:     s.prefix(""),
		MaxAge:   int(ttl.Seconds()),
		HttpOnly: true,  // Don't let JS access the cookie
		Secure:   false, // also allow plain http
		SameSite: http.SameSiteLaxMode,
	}
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
	cpuPlayer := pb.CPUPlayerMode_NONE
	if singlePlayer {
		cpuPlayer = s.config.CPUPlayerMode
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
				CpuPlayer: cpuPlayer,
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
	w.Header().Set("Content-Type", "text/html")
	s.renderer.Render(w, filename, map[string]any{
		"URLPathPrefix": s.config.URLPathPrefix,
	})
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
	http.SetCookie(w, s.makePlayerCookie(playerId, s.config.LoginTTL))
	http.Redirect(w, r, s.prefix(""), http.StatusSeeOther)
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
	http.Redirect(w, r, s.prefix(gameId), http.StatusSeeOther)
}

func (s *StatelessServer) handleReset(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "unknown player", http.StatusForbidden)
		return
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
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
	s.gameStore.Publish(r.Context(), gameId, &pb.GameStorePubsubEvent{
		GameId: gameId,
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	})
}

func (s *StatelessServer) handleHexz(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		s.serveHtmlFile(w, loginHtmlFilename)
		return
	}
	// Prolong cookie ttl.
	http.SetCookie(w, s.makePlayerCookie(p.Id, s.config.LoginTTL))
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
		http.Redirect(w, r, s.prefix(""), http.StatusSeeOther)
		return
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	if _, err := s.gameStore.LookupGame(r.Context(), gameId); err != nil {
		// Game does not exist: offer to start a new game.
		http.Redirect(w, r, s.prefix(""), http.StatusSeeOther)
		return
	}
	// Game exists, serve HTML and prolong cookie ttl.
	http.SetCookie(w, s.makePlayerCookie(p.Id, s.config.LoginTTL))
	s.serveHtmlFile(w, gameHtmlFilename)
}

func (s *StatelessServer) handleWASMStats(w http.ResponseWriter, r *http.Request) {
	_, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "missing player cookie", http.StatusBadRequest)
		return
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
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
		http.Redirect(w, r, s.prefix(""), http.StatusSeeOther)
		return
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
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

func (s *StatelessServer) storeGameAndNotify(ctx context.Context, gameState *pb.GameState) error {
	gameId := gameState.GetGameInfo().GetId()
	if err := s.gameStore.UpdateGame(ctx, gameState); err != nil {
		return fmt.Errorf("failed to save game state: %v", err)
	}
	if s.dbStore != nil {
		if err := s.dbStore.InsertHistory(ctx, "move", gameId, gameState); err != nil {
			hlog.Errorf("Cannot add history entry for game %s in database: %s", gameState.GameInfo.Id, err)
		}
	}
	s.gameStore.Publish(ctx, gameId, &pb.GameStorePubsubEvent{
		GameId: gameId,
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	})
	return nil
}

func (s *StatelessServer) goMakeCPUMove(ge GameEngine, gameState *pb.GameState) {
	flagz, ok := ge.(*GameEngineFlagz)
	if !ok {
		hlog.Errorf("Cannot make move for game type %v", ge.GameType())
		return
	}
	// Asynchronously request a CPU move in a 1P game, if necessary.
	var cpuPlayer CPUPlayer
	switch gameState.GameInfo.CpuPlayer {
	case pb.CPUPlayerMode_LOCAL_CPU:
		cpuPlayer = NewLocalCPUPlayer(PlayerId(gameState.Players[1].Id), s.config.CpuThinkTime, 0)
	case pb.CPUPlayerMode_REMOTE_CPU:
		cpuPlayer = NewRemoteCPUPlayer(s.remoteCPUClient, PlayerId(gameState.Players[1].Id), s.config.CpuThinkTime, 0)
	default:
		hlog.Errorf("Async CPU move requested for CPU player type %v", gameState.GameInfo.CpuPlayer)
		return
	}
	go func() {
		gameId := gameState.GetGameInfo().GetId()
		hlog.Infof("Requesting CPU move (%T) for game %s", cpuPlayer, gameId)
		ctx, cancel := context.WithTimeout(context.Background(), s.config.CpuThinkTime*2)
		defer cancel()
		move, _, err := cpuPlayer.SuggestMove(ctx, flagz)
		if err != nil {
			hlog.Errorf("SuggestMove failed: %v", err)
			return
		}
		if !ge.MakeMove(*move) {
			hlog.Errorf("failed to make a CPU move for game %s", gameId)
			return
		}
		enc, _ := ge.Encode()
		gameState.EngineState = enc
		if err := s.storeGameAndNotify(ctx, gameState); err != nil {
			hlog.Errorf("failed to store game after CPU move: %v", err)
		}
	}()
}

func isCPUTurn(turn int, cpuPlayerMode pb.CPUPlayerMode_Enum) bool {
	return turn == 2 && (cpuPlayerMode == pb.CPUPlayerMode_LOCAL_CPU ||
		cpuPlayerMode == pb.CPUPlayerMode_REMOTE_CPU)
}

func (s *StatelessServer) handleMove(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "Player not logged in", http.StatusPreconditionFailed)
		return
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
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
	isWASM := gameState.GetGameInfo().GetCpuPlayer() == pb.CPUPlayerMode_WASM
	if pNum == 1 && isWASM && ge.Board().Turn == 2 {
		// TODO: fix this mess. Clients should explicitly tell us that this is a WASM move for P2.
		pNum = 2 // WASM move: pretend to be the CPU player.
	} else if ge.Board().Turn != pNum {
		http.Error(w, "player cannot make a move", http.StatusPreconditionFailed)
		return
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
	if err := s.storeGameAndNotify(r.Context(), gameState); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %v", gameId, err)
		return
	}

	if isCPUTurn(ge.Board().Turn, gameState.GameInfo.CpuPlayer) {
		s.goMakeCPUMove(ge, gameState)
	}
}

func (s *StatelessServer) handleUndo(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
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
	s.gameStore.Publish(r.Context(), gameId, &pb.GameStorePubsubEvent{
		GameId: gameId,
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	})
}

func (s *StatelessServer) handleRedo(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
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
	s.gameStore.Publish(r.Context(), gameId, &pb.GameStorePubsubEvent{
		GameId: gameId,
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	})
}

func (s *StatelessServer) handleSSE(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "Player not logged in", http.StatusPreconditionFailed)
		return
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
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
	// Disable proxy buffering, in case we're behind a reverse proxy.
	w.Header().Set("X-Accel-Buffering", "no")
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
		s.gameStore.Publish(r.Context(), gameId, &pb.GameStorePubsubEvent{
			GameId: gameId,
			Event: &pb.GameStorePubsubEvent_PlayerJoined_{
				PlayerJoined: &pb.GameStorePubsubEvent_PlayerJoined{
					PlayerName: p.Name,
				},
			},
		})
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
			ClientSideCPUPlayer: gameState.GameInfo.CpuPlayer == pb.CPUPlayerMode_WASM,
		},
		DisableUndo: s.config.DisableUndo,
	})
	if err != nil {
		hlog.Errorf("Cannot send initial ServerEvent: %s", err)
		return
	}
	// Process events from the game store.
	eventCh := s.gameStore.Subscribe(r.Context(), gameId)
	for e := range eventCh {
		switch event := e.Event.(type) {
		case *pb.GameStorePubsubEvent_PlayerJoined_:
			playerName := event.PlayerJoined.GetPlayerName()
			hlog.Infof("[%s/%s] A new player joined: %s", gameId, p.Name, playerName)
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
				Announcements: []string{"New player " + playerName + " joined!"},
			})
			if err != nil {
				hlog.Errorf("Cannot send ServerEvent: %s", err)
				return
			}
		case *pb.GameStorePubsubEvent_GameUpdated_:
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
			hlog.Infof("[%s/%s] Received unknown event: %T", gameId, p.Name, e.Event)
		}
	}
	hlog.Infof("SSE connection closed for player %s", p.Name)
}

func (s *StatelessServer) createMux() *http.ServeMux {
	// TODO: Several generic handler functions are copy&pasted from server.go. We should
	// refactor them into a common place.

	mux := &http.ServeMux{}
	handle := func(pattern string, handler http.Handler) {
		mux.Handle(s.prefix(pattern), handler)
	}
	handleFunc := func(pattern string, handler func(http.ResponseWriter, *http.Request)) {
		mux.HandleFunc(s.prefix(pattern), handler)
	}
	// Static resources (images, JavaScript, ...) live under DocumentRoot.
	// Use gzipped.FileServer to deliver the WASM module compressed with
	// Content-Encoding: gzip
	// (other resources, too, but for them it doesn't matter).
	handle("/static/", http.StripPrefix(s.prefix("/static/"),
		gzipped.FileServer(gzipped.Dir(s.config.DocumentRoot))))
	// POST method API
	handleFunc("/login", postHandlerFunc(s.handleLoginRequest))
	handleFunc("/new", postHandlerFunc(s.handleNewGame))
	handleFunc("/move/{gameId}", postHandlerFunc(s.handleMove))
	handleFunc("/reset/{gameId}", postHandlerFunc(s.handleReset))
	// Methods for CPU player.
	handleFunc("/state/{gameId}", s.handleState)
	handleFunc("/wasmstats/{gameId}", postHandlerFunc(s.handleWASMStats))
	handleFunc("/undo/{gameId}", postHandlerFunc(s.handleUndo))
	handleFunc("/redo/{gameId}", postHandlerFunc(s.handleRedo))
	// Server-sent Event handling
	handleFunc("/sse/{gameId}", s.handleSSE)

	// GET method API
	handleFunc("", s.handleHexz)
	handleFunc("/gamez", s.handleGamez)
	// mux.HandleFunc("/hexz/view/", s.handleView)
	// mux.HandleFunc("/hexz/history/", s.handleHistory)
	// mux.HandleFunc("/hexz/moves/", s.handleValidMoves)
	handleFunc("/{gameId}", s.handleGame)
	// Technical services
	// mux.Handle("/statusz", s.basicAuthHandlerFunc(s.handleStatusz))

	return mux
}

func (s *StatelessServer) Serve() {
	addr := fmt.Sprintf("%s:%d", s.config.ServerHost, s.config.ServerPort)
	mux := s.createMux()
	srv := &http.Server{
		Addr:    addr,
		Handler: s.loggingHandler(mux),
	}

	// Quick sanity check that we have access to the game resource files.
	if _, err := s.readStaticResource("js/game.js"); err != nil {
		hlog.Fatalf("Cannot load game HTML: %s", err)
	}

	hlog.Infof("Stateless server listening on %s", addr)

	if s.config.TlsCertChain != "" && s.config.TlsPrivKey != "" {
		hlog.Fatalf("%v", srv.ListenAndServeTLS(s.config.TlsCertChain, s.config.TlsPrivKey))
	}
	hlog.Fatalf("%v", srv.ListenAndServe())
}
