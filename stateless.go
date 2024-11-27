package hexz

import (
	"compress/gzip"
	"context"
	crand "crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/lpar/gzipped/v2"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/internal/hlog"
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
	LoginDatabasePath  string                // Path to the file where the player DB is stored. If empty, no persistent storage is used.
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
	DisableUndo        bool   // If true, Undo/Redo is enabled for all games
	TlsCertChain       string
	TlsPrivKey         string
	DebugMode          bool
}

var (
	// Regexp used to validate player names.
	playernameRegexp = regexp.MustCompile(`^[\p{Latin}0-9_.-]+$`)
)

const (
	playerIdCookieName = "playerId"
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

// A random UUID used to identify players. Also used in cookies.
type PlayerId string

// Generates a random 128-bit hex string representing a player ID.
func generatePlayerId() PlayerId {
	p := make([]byte, 16)
	crand.Read(p)
	return PlayerId(hex.EncodeToString(p))
}

// Player has JSON annotations for serialization to disk.
// It is not used in the public API.
type Player struct {
	Id         PlayerId  `json:"id"`
	Name       string    `json:"name"`
	LastActive time.Time `json:"lastActive"`
}

func isValidPlayerName(name string) bool {
	return len(name) >= 3 && len(name) <= 20 && playernameRegexp.MatchString(name)
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
			hlog.Infof("Incoming request: %s %s %s", r.RemoteAddr, r.Method, r.URL.String())
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
func (s *StatelessServer) startNewGame(ctx context.Context, p *Player, gameType api.GameType, singlePlayer bool) (string, error) {
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
			UndoRedoState: &pb.GameState_UndoRedoState{
				InitialState: engineState,
			},
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
	s.serveHtmlFileParams(w, filename, nil)
}

func (s *StatelessServer) serveHtmlFileParams(w http.ResponseWriter, filename string, params map[string]any) {
	w.Header().Set("Content-Type", "text/html")
	templateParams := map[string]any{
		"URLPathPrefix": s.config.URLPathPrefix,
	}
	for k, v := range params {
		templateParams[k] = v
	}
	s.renderer.Render(w, filename, templateParams)
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
	gameId, err := s.startNewGame(r.Context(), &p, api.GameType(typeParam), singlePlayer)
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
	g, err := s.loadGame(r.Context(), gameId)
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
	if g.PlayerNum(string(p.Id)) <= 0 {
		http.Error(w, "only players can reset a game", http.StatusForbidden)
		return
	}
	if err := g.Reset(); err != nil {
		http.Error(w, "cannot reset game", http.StatusInternalServerError)
		hlog.Errorf("Cannot reset game %s: %v", gameId, err)
		return
	}
	if err := s.gameStore.UpdateGame(r.Context(), g.State()); err != nil {
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
			GameType: api.GameType(g.Type),
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
	var req api.WASMStatsRequest
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

func (s *StatelessServer) storeGameAndNotify(ctx context.Context, g *GameRepr) error {
	gameId := g.State().GetGameInfo().GetId()
	if err := s.gameStore.UpdateGame(ctx, g.State()); err != nil {
		return fmt.Errorf("failed to save game state: %v", err)
	}
	if s.dbStore != nil {
		if err := s.dbStore.InsertHistory(ctx, "move", gameId, g.State()); err != nil {
			hlog.Errorf("Cannot add history entry for game %s in database: %s", gameId, err)
		}
	}
	s.gameStore.Publish(ctx, gameId, &pb.GameStorePubsubEvent{
		GameId: gameId,
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	})
	return nil
}

func (s *StatelessServer) goMakeCPUMove(g *GameRepr) {
	_, ok := g.Engine().(*GameEngineFlagz)
	if !ok {
		hlog.Errorf("Cannot make move for game type %v", g.Engine().GameType())
		return
	}
	// Asynchronously request a CPU move in a 1P game, if necessary.
	var cpuPlayer CPUPlayer
	switch g.State().GameInfo.CpuPlayer {
	case pb.CPUPlayerMode_EMBEDDED_CPU:
		cpuPlayer = NewLocalCPUPlayer(PlayerId(g.State().Players[1].Id), s.config.CpuThinkTime, 0)
	case pb.CPUPlayerMode_REMOTE_CPU:
		cpuPlayer = NewRemoteCPUPlayer(s.remoteCPUClient, PlayerId(g.State().Players[1].Id), s.config.CpuThinkTime, 0)
	default:
		hlog.Errorf("Async CPU move requested for CPU player type %v", g.State().GameInfo.CpuPlayer)
		return
	}
	turn := g.Engine().Board().Turn
	go func() {
		// Play 1..N moves while it is CPU's turn and the game is not over.
		flagz := g.Engine().(*GameEngineFlagz)
		for !flagz.IsDone() && flagz.Board().Turn == turn {
			gameId := g.State().GetGameInfo().GetId()
			hlog.Infof("Requesting CPU move (%T) for game %s, move %d", cpuPlayer, gameId, flagz.Board().Move)
			ctx, cancel := context.WithTimeout(context.Background(), s.config.CpuThinkTime*2)
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
			flagz = g.Engine().(*GameEngineFlagz)
			ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
			err = s.storeGameAndNotify(ctx, g)
			cancel()
			if err != nil {
				hlog.Errorf("failed to store game after CPU move: %v", err)
			}
		}
	}()
}

func isCPUTurn(turn int, cpuPlayerMode pb.CPUPlayerMode_Enum) bool {
	return turn == 2 && (cpuPlayerMode == pb.CPUPlayerMode_EMBEDDED_CPU ||
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
	g, err := s.loadGame(r.Context(), gameId)
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
	pNum := g.PlayerNum(string(p.Id))
	isWASM := g.State().GetGameInfo().GetCpuPlayer() == pb.CPUPlayerMode_WASM
	if pNum == 1 && isWASM && g.Engine().Board().Turn == 2 {
		// TODO: fix this mess. Clients should explicitly tell us that this is a WASM move for P2.
		pNum = 2 // WASM move: pretend to be the CPU player.
	} else if g.Engine().Board().Turn != pNum {
		http.Error(w, "player cannot make a move", http.StatusPreconditionFailed)
		return
	}
	if err := g.MakeMove(GameEngineMove{
		PlayerNum: pNum,
		Move:      req.Move,
		Row:       req.Row,
		Col:       req.Col,
		CellType:  req.Type,
	}); err != nil {
		http.Error(w, "invalid move", http.StatusBadRequest)
		return
	}

	// Store new game state and notify other players.
	if err := s.storeGameAndNotify(r.Context(), g); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %v", gameId, err)
		return
	}

	if !g.Engine().IsDone() && isCPUTurn(g.Engine().Board().Turn, g.State().GameInfo.CpuPlayer) {
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
	g, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	if g.PlayerNum(string(p.Id)) == 0 {
		http.Error(w, "Only players can undo a move", http.StatusForbidden)
		return
	}

	if err := g.Undo(); err != nil {
		if errors.Is(err, errUndoRedoEmpty) {
			http.Error(w, "Cannot undo: no previous state", http.StatusPreconditionFailed)
			return
		}
		hlog.Errorf("Undo failed: %v", err)
		http.Error(w, "Cannot undo (no previous state?)", http.StatusInternalServerError)
		return
	}
	if err := s.gameStore.UpdateGame(r.Context(), g.State()); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %s", gameId, err)
		return
	}
	if s.dbStore != nil {
		if err := s.dbStore.InsertHistory(r.Context(), "undo", gameId, nil); err != nil {
			hlog.Errorf("Cannot add history entry for game %s in database: %s", gameId, err)
		}
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
	g, err := s.loadGame(r.Context(), gameId)
	if err != nil {
		http.Error(w, "No such game", http.StatusNotFound)
		return
	}
	if g.PlayerNum(string(p.Id)) == 0 {
		http.Error(w, "Only players can redo a move", http.StatusForbidden)
		return
	}
	if err := g.Redo(); err != nil {
		if errors.Is(err, errUndoRedoEmpty) {
			http.Error(w, "No next game state", http.StatusNotFound)
			return
		}
		hlog.Errorf("Failed to redo move: %v", err)
		http.Error(w, "Failed to redo move", http.StatusInternalServerError)
		return
	}
	if err := s.gameStore.UpdateGame(r.Context(), g.State()); err != nil {
		http.Error(w, "failed to save game state", http.StatusInternalServerError)
		hlog.Errorf("Could not store game %s: %s", gameId, err)
		return
	}
	if s.dbStore != nil {
		if err := s.dbStore.InsertHistory(r.Context(), "redo", gameId, nil); err != nil {
			hlog.Errorf("Cannot add history entry for game %s in database: %s", gameId, err)
		}
	}
	s.gameStore.Publish(r.Context(), gameId, &pb.GameStorePubsubEvent{
		GameId: gameId,
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	})
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
	s.serveHtmlFile(w, viewHtmlFilename)
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
		geMove.DecodeProto(move)
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
	s.serveHtmlFileParams(w, historyHtmlFilename, map[string]any{
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
			if err := s.dbStore.InsertHistory(r.Context(), "join", gameId, g.State()); err != nil {
				hlog.Errorf("Cannot add history entry for game %s in database: %s", gameId, err)
			}
		}
		pNum = g.PlayerNum(string(p.Id))
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
		Board:         g.Engine().Board().ViewFor(g.PlayerNum(string(p.Id))),
		Role:          g.PlayerNum(string(p.Id)),
		PlayerNames:   g.PlayerNames(),
		Announcements: []string{fmt.Sprintf("Welcome %s!", p.Name)},
		GameInfo: &ServerEventGameInfo{
			ValidCellTypes:      g.Engine().ValidCellTypes(),
			GameType:            g.Engine().GameType(),
			ClientSideCPUPlayer: g.State().GameInfo.CpuPlayer == pb.CPUPlayerMode_WASM,
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
	handleFunc("/view/{gameId}", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, r.URL.Path+"/0", http.StatusTemporaryRedirect)
	})
	handleFunc("/view/{gameId}/{seqNum}", s.handleView)
	handleFunc("/history", s.handleHistoryList)
	handleFunc("/history/{gameId}", s.handleHistory)
	handleFunc("/moves/{gameId}", s.handleValidMoves)
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
