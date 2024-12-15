package hexz

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	crand "crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/lpar/gzipped/v2"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/internal/hexzmem"
	"github.com/dnswlt/hexz/internal/hexzsql"
	"github.com/dnswlt/hexz/internal/hlog"
	"github.com/dnswlt/hexz/internal/xrand"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

type ServerConfig struct {
	ServerHost string
	ServerPort int
	// Path prefix that all URLs for this server have.
	// Usually "/hexz/", but when running behind a reverse proxy it might differ.
	URLPathPrefix      string
	DocumentRoot       string                // Path to static resource files.
	GameHistoryRoot    string                // Path to game history files.
	RemoteCPUPlayerURL string                // Base URL of the remote CPU player server. If emtpy, a local CPU player is used.
	CPUPlayerMode      pb.CPUPlayerMode_Enum // Type of CPU player to use.
	RedisAddr          string                // Address of the Redis server. If empty, local storage is used.
	PostgresURL        string                // URL of the PostgreSQL server. If empty, no persistent storage is used.
	InactivityTimeout  time.Duration         // Time after which a game is ended due to inactivity.
	PlayerRemoveDelay  time.Duration         // Time to wait before removing an unregistered player from the game.
	LoginTTL           time.Duration
	CpuThinkTime       time.Duration
	CpuMaxFlags        int
	AuthTokenSha256    string // Used in http Basic authentication for /statusz. Must be a SHA256 checksum.
	TlsCertChain       string
	TlsPrivKey         string
	DebugMode          bool
	// The VCS (typically: git) revision that the binary was built at.
	// Can be used in .js / .wasm URLs as a query parameter to avoid
	// the usual browser caching problems.
	VCSRevision string
}

const (
	playerIdCookieName = "playerId"
)

var (
	errInvalidGameID = fmt.Errorf("invalid game ID")
	errMissingCookie = fmt.Errorf("missing cookie")
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
	playerStore     hexzmem.PlayerStore
	dbStore         hexzsql.DatabaseStore
	gameStore       hexzmem.GameStore
	remoteCPUClient pb.CPUPlayerServiceClient // Only non-nil if a remote CPU addr was configured.
}

type StatelessServerBuilder struct {
	s *StatelessServer
}

func NewStatelessServerBuilder(config *ServerConfig, playerStore hexzmem.PlayerStore, gameStore hexzmem.GameStore, renderer *Renderer) *StatelessServerBuilder {
	return &StatelessServerBuilder{
		s: &StatelessServer{
			config:      config,
			playerStore: playerStore,
			gameStore:   gameStore,
			renderer:    renderer,
		},
	}
}

func (b *StatelessServerBuilder) WithDatabaseStore(dbStore hexzsql.DatabaseStore) *StatelessServerBuilder {
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

// Generates a random 128-bit hex string representing a player ID.
func generatePlayerId() api.PlayerId {
	p := make([]byte, 16)
	crand.Read(p)
	return api.PlayerId(hex.EncodeToString(p))
}

func validatePlayerName(name string) error {
	if len(name) < 3 || len(name) > 20 {
		return fmt.Errorf("player names must be 3..20 characters long")
	}
	if name[0] == ' ' || name[len(name)-1] == ' ' {
		return fmt.Errorf("player name starts or ends with a space")
	}
	letters := 0
	var prev rune
	for _, r := range name {
		if unicode.Is(unicode.Latin, r) {
			letters++
		} else if r == ' ' {
			if prev == ' ' {
				return fmt.Errorf("player name %q contains consecutive spaces", name)
			}
		} else if !strings.ContainsRune("_.-", r) && !unicode.IsDigit(r) {
			return fmt.Errorf("invalid character '%c' in player name %q", r, name)
		}
		prev = r
	}
	if letters == 0 {
		return fmt.Errorf("player name must contain at least one latin letter")
	}
	return nil
}

func isValidGameId(gameId string) bool {
	if len(gameId) != 6 {
		return false
	}
	for i := 0; i < len(gameId); i++ {
		if !(gameId[i] >= 'A' && gameId[i] <= 'Z') {
			return false
		}
	}
	return true
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

// Joins the prefix and urlPath into a single URL path.
// If either is empty, the other value is returned (even if it's empty as well).
// Otherwise, the two path segments are joined by a single '/'.
func urlJoinPath(prefix, urlPath string) string {
	if prefix == "" {
		return urlPath
	}
	if urlPath == "" {
		return prefix
	}
	// Neither is empty. Join, and ensure to have a single '/' in between.
	prefix = strings.TrimSuffix(prefix, "/")
	urlPath = strings.TrimPrefix(urlPath, "/")
	return prefix + "/" + urlPath
}

func (s *StatelessServer) loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.config.DebugMode {
			var proxyHeaders []string
			if xRealIP := r.Header.Get("X-Real-IP"); xRealIP != "" {
				proxyHeaders = append(proxyHeaders, "X-Real-IP: "+xRealIP)
			}
			if xForwardedFor := r.Header.Get("X-Forwarded-For"); xForwardedFor != "" {
				proxyHeaders = append(proxyHeaders, "X-Forwarded-For: "+xForwardedFor)
			}
			var proxyInfo string
			if len(proxyHeaders) > 0 {
				proxyInfo = fmt.Sprintf(" (via proxy: %s)", strings.Join(proxyHeaders, "; "))
			}
			hlog.Infof("Incoming request: %s %s %s%s", r.RemoteAddr, r.Method, r.URL.String(), proxyInfo)
		}
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

func (s *StatelessServer) makePlayerCookie(playerId api.PlayerId, ttl time.Duration) *http.Cookie {
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

func (s *StatelessServer) deletePlayerCookie() *http.Cookie {
	return &http.Cookie{
		Name:     playerIdCookieName,
		Value:    "",
		Path:     s.prefix(""),
		MaxAge:   -1,    // Delete immediately
		HttpOnly: true,  // Don't let JS access the cookie
		Secure:   false, // also allow plain http
		SameSite: http.SameSiteLaxMode,
	}
}

func (s *StatelessServer) lookupPlayerFromCookie(r *http.Request) (api.Player, error) {
	cookie, err := r.Cookie(playerIdCookieName)
	if err != nil {
		return api.Player{}, errMissingCookie
	}
	return s.playerStore.Lookup(r.Context(), api.PlayerId(cookie.Value))
}

func (s *StatelessServer) storeNewGameFromExisting(ctx context.Context, inputGameState *pb.GameState) (*GameRepr, error) {
	if len(inputGameState.Players) == 0 {
		return nil, fmt.Errorf("cannot start new game from existing: input game has no players or no GameInfo")
	}
	if inputGameState.GameInfo == nil {
		return nil, fmt.Errorf("cannot start new game from existing: input game has no GameInfo")
	}
	if !validGameType(inputGameState.GameInfo.Type) {
		return nil, fmt.Errorf("cannot start new game from existing: invalid game type: %s", inputGameState.GameInfo.Type)
	}
	// Copy input game state and only modify the copy.
	gameState := proto.Clone(inputGameState).(*pb.GameState)

	// Reset game ID, but keep pubsub ID, so the same topic gets reused. The game ID will be set in StoreNewGame below.
	gameState.GameInfo.Id = ""
	gameState.GameInfo.Started = tpb.Now()
	engineState := NewGameEngine(api.GameType(gameState.GameInfo.Type)).Proto()

	// Reset game engine and undo/redo states.
	gameState.EngineState = engineState
	gameState.UndoRedoState = &pb.GameState_UndoRedoState{
		InitialState: engineState,
	}
	if err := s.gameStore.StoreNewGame(ctx, gameState); err != nil {
		return nil, err
	}
	// Game was successfully stored in gameStore. Add to DB as well.
	if s.dbStore != nil {
		if err := s.dbStore.StoreGame(ctx, gameState.Players[0].Id, gameState); err != nil {
			hlog.Errorf("Cannot store game %s in database: %s", gameState.GameInfo.Id, err)
		}
	}
	return NewGameRepr(gameState), nil
}

// Stores a new game in the game store and returns the new game ID.
func (s *StatelessServer) startNewGame(r *http.Request, p *api.Player, gameType api.GameType, singlePlayer bool) (*GameRepr, error) {
	engineState := NewGameEngine(gameType).Proto()
	players := []*pb.Player{{Id: string(p.Id), Name: p.Name}}
	allJoined := false
	if singlePlayer {
		players = append(players, &pb.Player{Id: "CPU", Name: "CPU"})
		allJoined = true
	}
	cpuPlayerMode := pb.CPUPlayerMode_NONE
	if singlePlayer {
		cpuPlayerMode = s.config.CPUPlayerMode
	}
	gameInfo := &pb.GameInfo{
		Host:          p.Name,
		Started:       tpb.Now(),
		Type:          string(gameType),
		CpuPlayerMode: cpuPlayerMode,
		Settings: &pb.GameInfo_Settings{
			CpuThinkTimeMillis: s.config.CpuThinkTime.Milliseconds(),
		},
	}
	gameState := &pb.GameState{
		GameInfo:         gameInfo,
		Players:          players,
		AllPlayersJoined: allJoined,
		EngineState:      engineState,
		UndoRedoState: &pb.GameState_UndoRedoState{
			InitialState: engineState,
		},
	}
	if err := s.gameStore.StoreNewGame(r.Context(), gameState); err != nil {
		return nil, err
	}
	// Game was successfully stored in gameStore.
	if s.dbStore != nil {
		if err := s.dbStore.StoreGame(r.Context(), string(p.Id), gameState); err != nil {
			hlog.Errorf("Cannot store game %s in database: %s", gameState.GameInfo.Id, err)
		}
	}
	return NewGameRepr(gameState), nil
}

// Sends the contents of filename to the ResponseWriter.
func (s *StatelessServer) serveHtmlTemplate(w http.ResponseWriter, filename string) {
	s.serveHtmlTemplateParams(w, filename, nil)
}

func (s *StatelessServer) defaultTemplateParams() map[string]any {
	vcsRevision := s.config.VCSRevision
	if len(s.config.VCSRevision) > 12 {
		// Avoid overly long revision IDs in cache-busting URLs
		// (But avoid truncating timestamps, which are 10 digits.)
		vcsRevision = vcsRevision[:12]
	}
	return map[string]any{
		"URLPathPrefix": s.config.URLPathPrefix,
		"VCSRevision":   vcsRevision,
	}
}

func (s *StatelessServer) serveHtmlTemplateParams(w http.ResponseWriter, filename string, params map[string]any) {
	templateParams := s.defaultTemplateParams()
	for k, v := range params {
		templateParams[k] = v
	}
	var data bytes.Buffer
	err := s.renderer.Render(&data, filename, templateParams)
	if err != nil {
		http.Error(w, "failed to render template", http.StatusInternalServerError)
		var keys []string
		for k := range params {
			keys = append(keys, k)
		}
		hlog.Errorf("Error rendering template %s with parameters %v: %v", filename, keys, err)
		return
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(data.Bytes())
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
	if err := validatePlayerName(name); err != nil {
		s.serveHtmlTemplateParams(w, loginHtmlFilename, map[string]any{
			"ErrorMessage": fmt.Sprintf("Invalid username: %v", err),
		})
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

func (s *StatelessServer) handleLogoutRequest(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "Player not logged in", http.StatusPreconditionFailed)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Invalid form", http.StatusBadRequest)
		return
	}
	if err := s.playerStore.Logout(r.Context(), p.Id); err != nil {
		hlog.Infof("Rejected logout for player %s: %s", p.Name, err)
		http.Error(w, "Cannot log out right now", http.StatusPreconditionFailed)
		return
	}
	http.SetCookie(w, s.deletePlayerCookie())
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
	gameType := api.GameType(typeParam)
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
	g, err := s.startNewGame(r, &p, api.GameType(typeParam), singlePlayer)
	if err != nil {
		hlog.Errorf("Cannot start new game: %s\n", err)
		http.Error(w, "", http.StatusPreconditionFailed)
		return
	}
	http.Redirect(w, r, s.prefix(g.GameID()), http.StatusSeeOther)
}

func (s *StatelessServer) handleReset(w http.ResponseWriter, r *http.Request) {
	q, err := readActiveGameRequest[ResetRequest](s, w, r)
	if err != nil {
		return
	}
	g := q.gameRepr
	if g.PlayerNum(string(q.player.Id)) <= 0 {
		http.Error(w, "only players can reset a game", http.StatusForbidden)
		return
	}

	newG, err := s.storeNewGameFromExisting(r.Context(), g.State())
	if err != nil {
		hlog.Errorf("Reset: failed to store new game: %v", err)
		http.Error(w, "could not store new game", http.StatusInternalServerError)
		return
	}
	s.gameStore.Publish(r.Context(), newG.PubsubID(), &pb.GameStorePubsubEvent{
		GameId: q.gameId,
		Event: &pb.GameStorePubsubEvent_NewGameStarted_{
			NewGameStarted: &pb.GameStorePubsubEvent_NewGameStarted{
				GameId: newG.GameID(),
			},
		},
	})
}

func (s *StatelessServer) handleHexz(w http.ResponseWriter, r *http.Request) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		s.serveHtmlTemplate(w, loginHtmlFilename)
		return
	}
	// Prolong cookie ttl.
	http.SetCookie(w, s.makePlayerCookie(p.Id, s.config.LoginTTL))
	s.serveHtmlTemplateParams(w, newGameHtmlFilename, map[string]any{
		"PlayerName": p.Name,
	})
}

func (s *StatelessServer) serveGameInfos(w http.ResponseWriter, gameInfos []*hexzmem.GameInfo) {
	resp := make([]*GameInfo, len(gameInfos))
	for i, g := range gameInfos {
		resp[i] = &GameInfo{
			Id:       g.Id,
			Host:     g.Host,
			Started:  g.Started,
			GameType: g.GameType,
		}
	}
	json, err := json.Marshal(resp)
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		hlog.Errorf("JSON marshal error: %s", err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(json)
}

func (s *StatelessServer) handleActiveGames(w http.ResponseWriter, r *http.Request) {
	gameInfos, err := s.gameStore.ListActiveGames(r.Context(), 10)
	if err != nil {
		http.Error(w, "list active games", http.StatusInternalServerError)
		hlog.Errorf("Cannot list active games: %s", err)
		return
	}
	s.serveGameInfos(w, gameInfos)
}

func (s *StatelessServer) handleOpenGames(w http.ResponseWriter, r *http.Request) {
	gameInfos, err := s.gameStore.ListOpenGames(r.Context(), 10)
	if err != nil {
		http.Error(w, "list open games", http.StatusInternalServerError)
		hlog.Errorf("Cannot list open games: %s", err)
		return
	}
	s.serveGameInfos(w, gameInfos)
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
	g, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		// Game does not exist: offer to start a new game.
		http.Redirect(w, r, s.prefix(""), http.StatusSeeOther)
		return
	}
	// Game exists, serve HTML and prolong cookie ttl.
	http.SetCookie(w, s.makePlayerCookie(p.Id, s.config.LoginTTL))
	params := map[string]any{}
	if g.isCPUGame() {
		params["CPUThinkTimeOptions"] = cpuThinkTimeOptions(s.config.CpuThinkTime)
	}
	s.serveHtmlTemplateParams(w, gameHtmlFilename, params)
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
	var req api.WASMStatsRequest
	if err = json.Unmarshal(body, &req); err != nil {
		http.Error(w, "unmarshal error", http.StatusBadRequest)
		return
	}
	if s.dbStore != nil {
		s.dbStore.InsertStats(r.Context(), &req)
	}
	hlog.Infof("Received CPU stats for game %s: iterations=%d elapsed=%.3f heapAllocMiB=%.3f",
		req.GameId, req.Stats.Iterations, req.Stats.Elapsed.Seconds(), req.Stats.HeapAllocMiB)
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
	g, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		// Game does not exist: offer to start a new game.
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	if g.PlayerNum(string(p.Id)) == 0 {
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
	encodedGameState, err := proto.Marshal(g.State())
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		hlog.Errorf("Cannot marshal GameState: %s", err.Error())
	}
	enc.Encode(GameStateResponse{
		GameId:           gameId,
		EncodedGameState: encodedGameState,
	})
}

func (s *StatelessServer) loadGame(ctx context.Context, gameId string) (*GameRepr, error) {
	gameState, err := s.gameStore.LookupGame(ctx, gameId)
	if err != nil {
		return nil, err
	}
	return NewGameRepr(gameState), nil
}

func (s *StatelessServer) storeGameAndNotify(ctx context.Context, entryType string, g *GameRepr) error {
	if err := s.gameStore.UpdateGame(ctx, g.State()); err != nil {
		return fmt.Errorf("failed to save game state: %v", err)
	}
	gameId := g.State().GetGameInfo().GetId()
	if s.dbStore != nil {
		if err := s.dbStore.InsertHistory(ctx, entryType, gameId, g.State(), boardStatus(g.Engine())); err != nil {
			hlog.Errorf("Cannot add history entry for entry type %s, game %s in database: %s", entryType, gameId, err)
			return err
		}
	}
	return s.gameStore.Publish(ctx, g.State().GetGameInfo().PubsubChannelId, &pb.GameStorePubsubEvent{
		GameId: gameId,
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	})
}

func (s *StatelessServer) goMakeCPUMove(g *GameRepr) {
	if typ := g.State().GetGameInfo().Type; typ != string(gameTypeFlagz) {
		hlog.Errorf("Cannot make move for game %v of type %v", g.GameID(), typ)
		return
	}
	cpuThinkTime := s.config.CpuThinkTime
	if t := g.State().GetGameInfo().GetSettings().CpuThinkTimeMillis; t > 0 {
		cpuThinkTime = time.Duration(t) * time.Millisecond
	}
	// Asynchronously request a CPU move in a 1P game, if necessary.
	var cpuPlayer CPUPlayer
	switch g.State().GameInfo.CpuPlayerMode {
	case pb.CPUPlayerMode_EMBEDDED_CPU:
		cpuPlayer = NewLocalCPUPlayer(api.PlayerId(g.State().Players[1].Id), cpuThinkTime, 0)
	case pb.CPUPlayerMode_REMOTE_CPU:
		cpuPlayer = NewRemoteCPUPlayer(s.remoteCPUClient, api.PlayerId(g.State().Players[1].Id), cpuThinkTime, 0)
	default:
		hlog.Errorf("Async CPU move requested for CPU player type %v", g.State().GameInfo.CpuPlayerMode)
		return
	}
	go func(gameId string, turn int) {
		// Play 1..N moves while it is CPU's turn and the game is not over.
		for {
			// Always load the game from the store, it might have been modified concurrently (e.g. reset)
			g, err := s.loadGame(context.Background(), gameId)
			if err != nil {
				hlog.Errorf("MakeCPUMove: Could not load game: %v", err)
				return
			}
			flagz := g.Engine().(*GameEngineFlagz)
			if flagz.IsDone() || flagz.Board().Turn != turn {
				return
			}
			hlog.Infof("Requesting CPU move (%T) for game %s, move %d", cpuPlayer, gameId, flagz.Board().Move)
			ctx, cancel := context.WithTimeout(context.Background(), max(cpuThinkTime*2, 1*time.Second))
			move, _, err := cpuPlayer.SuggestMove(ctx, flagz)
			cancel()
			if err != nil {
				hlog.Errorf("SuggestMove failed: %v", err)
				return
			}
			if err := g.MakeMove(*move); err != nil {
				hlog.Errorf("failed to make a CPU move for game %s: %v", gameId, err)
				return
			}
			ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
			err = s.storeGameAndNotify(ctx, "move", g)
			cancel()
			if err != nil {
				hlog.Errorf("failed to store game %s after CPU move: %v", gameId, err)
			}
		}
	}(g.GameID(), g.Engine().Board().Turn)
}

type GameRequest[R any] struct {
	player   api.Player
	gameId   string
	request  *R
	gameRepr *GameRepr
}

func readActiveGameRequest[R any](s *StatelessServer, w http.ResponseWriter, r *http.Request) (*GameRequest[R], error) {
	p, err := s.lookupPlayerFromCookie(r)
	if err != nil {
		http.Error(w, "Player not logged in", http.StatusPreconditionFailed)
		return nil, err
	}
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return nil, errInvalidGameID
	}
	// Parse request as JSON.
	dec := json.NewDecoder(r.Body)
	request := new(R)
	if err := dec.Decode(request); err != nil {
		http.Error(w, "invalid request data", http.StatusBadRequest)
		return nil, err
	}
	// Load game.
	g, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return nil, err
	}
	return &GameRequest[R]{
		player:   p,
		gameId:   gameId,
		request:  request,
		gameRepr: g,
	}, nil
}

func (s *StatelessServer) handleMove(w http.ResponseWriter, r *http.Request) {
	q, err := readActiveGameRequest[MoveRequest](s, w, r)
	if err != nil {
		return
	}
	if !q.request.Type.valid() {
		http.Error(w, "Invalid cell type", http.StatusBadRequest)
		return
	}
	g := q.gameRepr
	// Is it the player's turn?
	pNum := g.PlayerNum(string(q.player.Id))
	isWASM := g.State().GetGameInfo().CpuPlayerMode == pb.CPUPlayerMode_WASM
	if pNum == 1 && isWASM && g.Engine().Board().Turn == 2 {
		// TODO: fix this mess. Clients should explicitly tell us that this is a WASM move for P2.
		pNum = 2 // WASM move: pretend to be the CPU player.
	} else if g.Engine().Board().Turn != pNum {
		http.Error(w, "player cannot make a move", http.StatusPreconditionFailed)
		return
	}
	if err := g.MakeMove(GameEngineMove{
		PlayerNum: pNum,
		Move:      q.request.Move,
		Row:       q.request.Row,
		Col:       q.request.Col,
		CellType:  q.request.Type,
	}); err != nil {
		http.Error(w, "invalid move", http.StatusBadRequest)
		return
	}

	// Store new game state and notify other players.
	if err := s.storeGameAndNotify(r.Context(), "move", g); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %v", q.gameId, err)
		return
	}

	if g.isCPUTurn() {
		s.goMakeCPUMove(g)
	}
}

func (s *StatelessServer) handleValidMoves(w http.ResponseWriter, r *http.Request) {
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	g, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}

	engine, ok := g.Engine().(*GameEngineFlagz)
	if !ok {
		http.Error(w, "invalid game type", http.StatusPreconditionFailed)
	}
	validMoves := engine.ValidMoves()
	moves := make([]*MoveRequest, len(validMoves))
	for i, m := range validMoves {
		moves[i] = &MoveRequest{
			Move: m.Move,
			Row:  m.Row,
			Col:  m.Col,
			Type: m.CellType,
		}
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	err = enc.Encode(moves)
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		hlog.Fatalf("Cannot marshal valid moves: %v", err)
	}
}

func (s *StatelessServer) handleUndoRedo(w http.ResponseWriter, r *http.Request) {
	q, err := readActiveGameRequest[UndoRedoRequest](s, w, r)
	if err != nil {
		return
	}

	action := q.request.Action
	if action != "undo" && action != "redo" {
		http.Error(w, "invalid action", http.StatusBadRequest)
		return
	}
	g := q.gameRepr
	if g.PlayerNum(string(q.player.Id)) == 0 {
		http.Error(w, "Only players can undo a move", http.StatusForbidden)
		return
	}
	if q.request.CurrentMove != g.Engine().Board().Move {
		http.Error(w, fmt.Sprintf("wrong move number: got %d, want %d", q.request.CurrentMove, g.Engine().Board().Move), http.StatusBadRequest)
		return
	}
	if action == "undo" {
		err = g.Undo()
	} else {
		err = g.Redo()
	}
	if err != nil {
		if errors.Is(err, errUndoRedoEmpty) {
			http.Error(w, "Cannot undo/redo: no previous state", http.StatusPreconditionFailed)
			return
		}
		hlog.Errorf("%s failed: %v", action, err)
		http.Error(w, "Failed to undo/redo", http.StatusInternalServerError)
		return
	}
	if err := s.gameStore.UpdateGame(r.Context(), g.State()); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %s", q.gameId, err)
		return
	}
	s.storeGameAndNotify(r.Context(), action, g)
}

func (s *StatelessServer) handleGameSettings(w http.ResponseWriter, r *http.Request) {
	q, err := readActiveGameRequest[GameSettingsRequest](s, w, r)
	if err != nil {
		return
	}
	g := q.gameRepr
	if g.PlayerNum(string(q.player.Id)) == 0 {
		http.Error(w, "Only players can update game settings", http.StatusForbidden)
		return
	}
	settings := g.State().GetGameInfo().Settings
	if settings == nil {
		http.Error(w, "game has no settings", http.StatusNotImplemented)
		return
	}
	if q.request.GameId != q.gameId {
		http.Error(w, "wrong game ID in request", http.StatusBadRequest)
		return
	}
	req := q.request
	if req.CPUThinkTimeMillis > 0 && req.CPUThinkTimeMillis <= s.config.CpuThinkTime.Milliseconds() {
		hlog.Infof("Updating CPU think time to %dms for game %v", req.CPUThinkTimeMillis, q.gameId)
		settings.CpuThinkTimeMillis = req.CPUThinkTimeMillis
	} else if req.CPUThinkTimeMillis != 0 {
		hlog.Infof("Ignoring request to set CpuThinkTimeMillis to %dms for game %v", req.CPUThinkTimeMillis, q.gameId)
	}
	if err := s.gameStore.UpdateGame(r.Context(), g.State()); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %s", q.gameId, err)
		return
	}
	if s.dbStore != nil {
		if err := s.dbStore.InsertHistory(r.Context(), "settings", q.gameId, g.State(), nil); err != nil {
			hlog.Errorf("Cannot add history entry for game %s in database: %s", q.gameId, err)
		}
	}

}

func (s *StatelessServer) handleView(w http.ResponseWriter, r *http.Request) {
	// Return the HTML for viewing a game.
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	seqNum := r.PathValue("seqNum")
	if _, err := strconv.Atoi(seqNum); err != nil {
		http.Error(w, "invalid seqNum", http.StatusBadRequest)
		return
	}
	s.serveHtmlTemplate(w, viewHtmlFilename)
}

func replayForGameHistoryResponse(gameState *pb.GameState) (*GameHistoryResponse, error) {
	if gameState.GetUndoRedoState().InitialState == nil {
		return nil, fmt.Errorf("no initial state in GameState")
	}
	ge, err := DecodeGameEngine(gameState.UndoRedoState.InitialState)
	if err != nil {
		return nil, fmt.Errorf("cannot decode game engine: %v", err)
	}
	entries := []*GameHistoryResponseEntry{
		{
			Timestamp:  gameState.GetModified().AsTime(),
			EntryType:  "reset",
			Board:      ge.Board().ViewFor(ge.Board().Turn),
			MoveScores: nil,
		},
	}
	for i, move := range gameState.UndoRedoState.Moves {
		geMove := GameEngineMove{}
		geMove.FromProto(move)
		if err := ge.MakeMoveError(geMove); err != nil {
			return nil, fmt.Errorf("error making move #%d: %v: %v", i, move, err)
		}
		entries = append(entries, &GameHistoryResponseEntry{
			Timestamp: gameState.GetModified().AsTime(),
			EntryType: "move",
			Move: &MoveRequest{
				Move: geMove.Move,
				Row:  geMove.Row,
				Col:  geMove.Col,
				Type: geMove.CellType,
			},
			Board:      ge.Board().ViewFor(geMove.PlayerNum),
			MoveScores: nil,
		})
	}
	playerNames := make([]string, len(gameState.Players))
	for i, p := range gameState.Players {
		playerNames[i] = p.Name
	}
	return &GameHistoryResponse{
		GameId:      gameState.GetGameInfo().Id,
		PlayerNames: playerNames,
		GameType:    api.GameType(gameState.GetGameInfo().Type),
		Entries:     entries,
	}, nil
}

func (s *StatelessServer) handleHistoryList(w http.ResponseWriter, r *http.Request) {
	if s.dbStore == nil {
		http.Error(w, "no database connected", http.StatusPreconditionFailed)
		return
	}
	validOffset := func(offset int) bool {
		return offset >= 0 && offset < 1000000
	}
	offsetStr := r.URL.Query().Get("offset")
	offset := 0
	if o, err := strconv.Atoi(offsetStr); err == nil && validOffset(o) {
		offset = o
	}
	limit := 20
	games, err := s.dbStore.ListRecentGames(r.Context(), offset, limit+1)
	if err != nil {
		hlog.Errorf("failed to list games: %v", err)
		http.Error(w, "failed to list games", http.StatusInternalServerError)
		return
	}
	moreGames := len(games) > limit
	if moreGames {
		// The last game is a marker showing that there are more games.
		games = games[:limit]
	}
	prevOffset := offset - limit
	if prevOffset < 0 {
		prevOffset = 0
	}
	s.serveHtmlTemplateParams(w, historyHtmlFilename, map[string]any{
		"Games":      games,
		"Offset":     offset,
		"HasPrev":    offset > 0,
		"PrevOffset": prevOffset,
		"HasNext":    moreGames,
		"NextOffset": offset + limit,
	})
}

func (s *StatelessServer) handleHistory(w http.ResponseWriter, r *http.Request) {
	gameId := r.PathValue("gameId")
	if !isValidGameId(gameId) {
		http.Error(w, "Invalid game ID", http.StatusBadRequest)
		return
	}
	if s.dbStore == nil {
		http.Error(w, "no database connected", http.StatusPreconditionFailed)
		return
	}
	gameState, err := s.dbStore.LoadGame(r.Context(), gameId)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.Error(w, "game not found", http.StatusNotFound)
			return
		}
		hlog.Errorf("Failed to load game %s from DB: %v", gameId, err)
		http.Error(w, "cannot load game", http.StatusInternalServerError)
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
	resp, err := replayForGameHistoryResponse(gameState)
	if err != nil {
		http.Error(w, "", http.StatusInternalServerError)
		hlog.Errorf("Failed to generate history response: %v", err)
		return
	}
	err = enc.Encode(resp)
	if err != nil {
		http.Error(w, "", http.StatusInternalServerError)
		hlog.Fatalf("Failed to marshal history response: %s", err)
	}
}

func readLines(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var lines []string

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// Skip emtpy lines and comment lines starting with '#'.
		if line != "" && line[0] != '#' {
			lines = append(lines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return lines, nil
}

func (s *StatelessServer) handleLoginNames(w http.ResponseWriter, r *http.Request) {
	animals, err := readLines(path.Join(s.config.DocumentRoot, "data", "animals.txt"))
	if err != nil {
		http.Error(w, "no suggestions available", http.StatusNotFound)
		return
	}
	adjectives, err := readLines(path.Join(s.config.DocumentRoot, "data", "adjectives.txt"))
	if err != nil {
		http.Error(w, "no suggestions available", http.StatusNotFound)
		return
	}
	if len(animals) == 0 || len(adjectives) == 0 {
		http.Error(w, "no suggestions available", http.StatusNotFound)
		return
	}
	n := 100 // Return a few suggestions, so clients don't have to request them one by one.
	var names []string
	for i := 0; i < n; i++ {
		adj := adjectives[xrand.Intn(len(adjectives))]
		animal := animals[xrand.Intn(len(animals))]
		name := adj + " " + animal // Add spaces so names can be wrapped.
		if len(name) <= 20 {
			// Ignore invalid names
			names = append(names, name)
		}
	}
	if len(names) == 0 {
		// this should never happen: all combinations are too long!?
		http.Error(w, "no suggestions generated", http.StatusNotFound)
		hlog.Errorf("No valid login suggestions generated. Data corruption?")
		return
	}
	json, err := json.Marshal(LoginNamesResponse{
		Names: names,
	})
	if err != nil {
		http.Error(w, "marshal error", http.StatusInternalServerError)
		hlog.Errorf("JSON marshal error: %s", err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(json)
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
	g, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	// Disable proxy buffering, in case we're behind a reverse proxy.
	w.Header().Set("X-Accel-Buffering", "no")
	pNum := g.PlayerNum(string(p.Id))
	// If there is a slot in the game left, add this player.
	if pNum == 0 && !g.AllPlayersJoined() {
		g.AddPlayer(&pb.Player{
			Id:   string(p.Id),
			Name: p.Name,
		})
		if err := s.gameStore.UpdateGame(r.Context(), g.State()); err != nil {
			hlog.Errorf("Cannot store updated game state: %s", err)
			return
		}
		if s.dbStore != nil {
			if err := s.dbStore.InsertHistory(r.Context(), "join", gameId, g.State(), nil); err != nil {
				hlog.Errorf("Cannot add history entry for game %s in database: %s", gameId, err)
			}
		}
		pNum = g.PlayerNum(string(p.Id))
		// Tell others we've joined.
		s.gameStore.Publish(r.Context(), g.PubsubID(), &pb.GameStorePubsubEvent{
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
		Board:         g.Engine().Board().ViewFor(g.PlayerNum(string(p.Id))),
		Role:          g.PlayerNum(string(p.Id)),
		PlayerNames:   g.PlayerNames(),
		Announcements: []string{fmt.Sprintf("Welcome %s!", p.Name)},
		GameInfo: &ServerEventGameInfo{
			ValidCellTypes:      g.Engine().ValidCellTypes(),
			GameType:            g.Engine().GameType(),
			ClientSideCPUPlayer: g.State().GetGameInfo().CpuPlayerMode == pb.CPUPlayerMode_WASM,
		},
	})
	if err != nil {
		hlog.Errorf("Cannot send initial ServerEvent: %s", err)
		return
	}
	// Process events from the game store.
	eventCh := s.gameStore.Subscribe(r.Context(), g.PubsubID())
	for e := range eventCh {
		if e.GameId != gameId {
			// This should only happen if events from Redis arrive out of order, and
			// we see an old event from a previous game (before a reset).
			hlog.Errorf("[%s/%s] Ignoring event for game %s", gameId, p.Name, e.GameId)
			continue
		}
		switch event := e.Event.(type) {
		case *pb.GameStorePubsubEvent_PlayerJoined_:
			playerName := event.PlayerJoined.GetPlayerName()
			hlog.Infof("[%s/%s] A new player joined: %s", gameId, p.Name, playerName)
			g, err := s.loadGame(r.Context(), gameId)
			if err != nil {
				hlog.Errorf("Cannot load ongoing game %s: %s", gameId, err)
				return
			}
			err = sendSSEEvent(w, ServerEvent{
				Timestamp:     time.Now(),
				Board:         g.Engine().Board().ViewFor(pNum),
				Role:          pNum,
				PlayerNames:   g.PlayerNames(),
				Announcements: []string{"New player " + playerName + " joined!"},
			})
			if err != nil {
				hlog.Errorf("Cannot send ServerEvent: %s", err)
				return
			}
		case *pb.GameStorePubsubEvent_GameUpdated_:
			g, err := s.loadGame(r.Context(), gameId)
			if err != nil {
				hlog.Errorf("Cannot load ongoing game %s: %s", gameId, err)
				return
			}
			var winner int
			var announcements []string
			if g.Engine().IsDone() {
				winner = g.Engine().Winner()
				if winner > 0 {
					announcements = append(announcements,
						fmt.Sprintf("&#127942; &#127942; &#127942; %s won &#127942; &#127942; &#127942;",
							g.PlayerNames()[winner-1]))
				} else {
					announcements = append(announcements, "The game is a draw!")
				}
			}
			err = sendSSEEvent(w, ServerEvent{
				Timestamp:     time.Now(),
				Board:         g.Engine().Board().ViewFor(pNum),
				Role:          pNum,
				PlayerNames:   g.PlayerNames(),
				Winner:        winner,
				Announcements: announcements,
			})
			if err != nil {
				hlog.Errorf("Cannot send ServerEvent: %s", err)
				return
			}
		case *pb.GameStorePubsubEvent_NewGameStarted_:
			gameId = event.NewGameStarted.GameId
			g, err := s.loadGame(r.Context(), gameId)
			if err != nil {
				hlog.Errorf("Cannot load game %s from NewGameStarted event: %v", gameId, err)
				return
			}
			err = sendSSEEvent(w, ServerEvent{
				Timestamp:     time.Now(),
				NewGameID:     gameId,
				Board:         g.Engine().Board().ViewFor(pNum),
				Role:          pNum,
				PlayerNames:   g.PlayerNames(),
				Announcements: []string{fmt.Sprintf("New game %s has started!", gameId)},
			})
			if err != nil {
				hlog.Errorf("Cannot send ServerEvent: %s", err)
				return
			}
		default:
			hlog.Errorf("[%s/%s] Received unknown event: %T", gameId, p.Name, e.Event)
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
	handleFunc("/logout", postHandlerFunc(s.handleLogoutRequest))
	handleFunc("/new", postHandlerFunc(s.handleNewGame))
	handleFunc("/move/{gameId}", postHandlerFunc(s.handleMove))
	handleFunc("/reset/{gameId}", postHandlerFunc(s.handleReset))
	// Methods for CPU player.
	handleFunc("/state/{gameId}", s.handleState)
	handleFunc("/wasmstats/{gameId}", postHandlerFunc(s.handleWASMStats))
	handleFunc("/undo/{gameId}", postHandlerFunc(s.handleUndoRedo))
	handleFunc("/redo/{gameId}", postHandlerFunc(s.handleUndoRedo))
	handleFunc("/gamesettings/{gameId}", postHandlerFunc(s.handleGameSettings))
	// Server-sent Event handling
	handleFunc("/sse/{gameId}", s.handleSSE)

	// GET method API
	handleFunc("", s.handleHexz)
	handleFunc("/opengames", s.handleOpenGames)
	handleFunc("/activegames", s.handleActiveGames)
	handleFunc("/view/{gameId}", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, r.URL.Path+"/0", http.StatusTemporaryRedirect)
	})
	handleFunc("/view/{gameId}/{seqNum}", s.handleView)
	handleFunc("/history", s.handleHistoryList)
	handleFunc("/history/{gameId}", s.handleHistory)
	handleFunc("/moves/{gameId}", s.handleValidMoves)

	handleFunc("/loginnames", s.handleLoginNames)

	// Must come last to avoid capturing other paths:
	handleFunc("/{gameId}", s.handleGame)
	// Technical services
	// mux.Handle("/statusz", s.basicAuthHandlerFunc(s.handleStatusz))

	// If we're not behind a reverse proxy and serve the root URL ourselves:
	// Redirect to prefix path
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, s.prefix(""), http.StatusTemporaryRedirect)
	})
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
