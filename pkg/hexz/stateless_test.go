package hexz

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/dnswlt/hexz/internal/hexzmem"
	"github.com/dnswlt/hexz/internal/hexzsql"
	"github.com/dnswlt/hexz/pkg/hexzpb"
)

const (
	testPlayerId   = "testId"
	testPlayerName = "tester"
)

var (
	// This is typically going to be postgres://hexz_test:hexz_test@localhost:5432/hexz_test
	postgresTestURL = flag.String("test-postgres-url", "", "URL to PostgreSQL DB for integration tests")
)

func testServerConfig(t *testing.T) *ServerConfig {
	historyRoot := t.TempDir()
	return &ServerConfig{
		ServerHost:        "localhost",
		ServerPort:        8999,
		URLPathPrefix:     "/hexz",
		DocumentRoot:      "./resources",
		GameHistoryRoot:   historyRoot,
		DebugMode:         true,
		LoginTTL:          24 * time.Hour, // By default, don't auto-log out players in tests.
		InactivityTimeout: 1 * time.Hour,
		CPUPlayerMode:     hexzpb.CPUPlayerMode_EMBEDDED_CPU,
	}
}

func newTestStatelessServer(t testing.TB, config *ServerConfig) (*StatelessServer, error) {
	t.Helper()
	templateDir := "../../resources/templates"
	renderer, err := NewRenderer(templateDir)
	if err != nil {
		absPath, _ := filepath.Abs(templateDir)
		return nil, fmt.Errorf("failed to created renderer in directory %s: %v", absPath, err)
	}
	playerStore, err := hexzmem.NewInMemoryPlayerStore(config.LoginTTL, config.LoginDatabasePath)
	if err != nil {
		return nil, err
	}
	gameStore := hexzmem.NewInMemoryGameStore(config.InactivityTimeout)
	b := NewStatelessServerBuilder(config, playerStore, gameStore, renderer)
	if config.PostgresURL != "" {
		var dbStore hexzsql.DatabaseStore
		dbStore, err := hexzsql.NewPostgresStore(context.Background(), config.PostgresURL)
		if err != nil {
			t.Fatalf("error connecting to postgres: %s", err)
		}
		b = b.WithDatabaseStore(dbStore)
	}
	return b.Build(), nil
}

func TestGenerateGameId(t *testing.T) {
	got := GenerateGameId()
	if !regexp.MustCompile(`^[A-Z]{6}$`).MatchString(got) {
		t.Errorf("Wrong gameId: %q", got)
	}
}

func TestValidPlayerName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"abc", true},
		{"abc.def", true},
		{"abc_def-123", true},
		{"1digit", true},
		{"HANS", true},
		{"Mørän", true},
		{"Jérôme", true},
		{"Strüßenbähn", true},
		{"123", true},
		{"_letter-or.digit", true},
		{"ab", false},      // Too short
		{"jens$", false},   // Invalid character
		{"dw@best", false}, // Invalid character
		{"", false},
		{"verylongusernamesarenotallowedalright", false},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if got := isValidPlayerName(test.name); got != test.want {
				t.Errorf("unexpected result %t for name %s", got, test.name)
			}
		})
	}
}

func TestHandleNewGame(t *testing.T) {
	cfg := testServerConfig(t)
	s, err := newTestStatelessServer(t, cfg)
	if err != nil {
		t.Fatal("Could not create server:", err)
	}
	if err := s.playerStore.Login(context.Background(), testPlayerId, testPlayerName); err != nil {
		t.Error("Cannot log in test player: ", err)
	}
	w := httptest.NewRecorder()
	// Create request with login form parameters.
	form := url.Values{}
	form.Add("type", string(gameTypeFlagz))
	form.Add("singlePlayer", "true")
	r := httptest.NewRequest(http.MethodPost, "/hexz/new", strings.NewReader(form.Encode()))
	r.AddCookie(s.makePlayerCookie(testPlayerId, 24*time.Hour))
	r.Header.Add("Content-Type", "application/x-www-form-urlencoded")

	s.handleNewGame(w, r)

	// Expect a redirect to /hexz/{gameId}
	resp := w.Result()
	want := http.StatusSeeOther
	if resp.StatusCode != want {
		msg, _ := io.ReadAll(resp.Body)
		t.Errorf("Want: %s, got: %s %q", http.StatusText(want), resp.Status, msg)
	}
	loc := resp.Header.Get("Location")
	if pattern := `/hexz/[A-Z]{6}`; !regexp.MustCompile(pattern).MatchString(loc) {
		t.Errorf("Wrong Location header: want: %s, got: %q", pattern, loc)
	}
	recentGames, err := s.gameStore.ListRecentGames(context.Background(), 100)
	if err != nil {
		t.Fatal("Could not get recent games:", err)
	}
	if len(recentGames) != 1 {
		t.Errorf("Ongoing games: %d, want: 1", len(recentGames))
	}
}

func TestURLJoinPath(t *testing.T) {
	tests := []struct {
		prefix string
		suffix string
		want   string
	}{
		{"", "", ""},
		{"", "/foo", "/foo"},
		{"", "foo", "foo"},
		{"/foo", "", "/foo"},
		{"foo", "", "foo"},
		{"/foo/", "/bar", "/foo/bar"},
		{"/foo/", "/bar/", "/foo/bar/"},
		{"/foo", "", "/foo"},
		{"/foo", "/", "/foo/"},
		{"", "/bar", "/bar"},
	}
	for _, tc := range tests {
		if got := urlJoinPath(tc.prefix, tc.suffix); got != tc.want {
			t.Errorf("Invalid prefix: got %q, want %q", got, tc.want)
		}
	}
}

// Process SSE ServerEvents and return boards with strictly monotonically increasing move numbers.
func receiveBoards(ctx context.Context, eventCh <-chan tcServerEvent) <-chan *BoardView {
	boardCh := make(chan *BoardView)
	go func() {
		defer close(boardCh)
		moveNum := -1
		for {
			select {
			case e := <-eventCh:
				if e.s.Board != nil {
					if e.s.Board.Move > moveNum {
						moveNum = e.s.Board.Move
						boardCh <- e.s.Board
					}
				}
				// Ignore errors and events without a new board.
			case <-ctx.Done():
				return
			}
		}
	}()
	return boardCh
}

func TestHistoryDatabase(t *testing.T) {
	if *postgresTestURL == "" {
		t.Skip("--test-postgres-url is not set, skipping database integration test")
	}
	cfg := testServerConfig(t)
	cfg.PostgresURL = *postgresTestURL
	cfg.CpuThinkTime = 1 * time.Millisecond // We want a fast test, not smart moves.
	srv, err := newTestStatelessServer(t, cfg)
	if err != nil {
		t.Fatal(err)
	}

	testServer := httptest.NewServer(srv.createMux())
	defer testServer.Close()

	c, err := newHexzTestClient(testServer.URL)
	if err != nil {
		t.Fatalf("could not create client: %s", err)
	}
	// Log in.
	if err := c.login("testuser"); err != nil {
		t.Fatal(err)
	}
	// Start a new single player game.
	gameId, err := c.newFlagzGame(true)
	if err != nil {
		t.Fatal(err)
	}
	// Receive SSE events.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eventCh, err := c.receiveEvents(ctx, testServer.URL+"/hexz/sse/"+gameId)
	if err != nil {
		t.Fatalf("cannot receive events for game %s: %s", gameId, err)
	}
	boardCh := receiveBoards(ctx, eventCh)
	finished := false
	maxMoves := numFieldsFirstRow * numBoardRows // upper bound for possible moves
	moves := 0
GameLoop:
	for ; !finished && moves < maxMoves; moves++ {
		// Receive boards until the game is finished or it's our turn again.
		for {
			board := <-boardCh
			if board.State == Finished {
				finished = true
				break GameLoop
			}
			if board.Turn == 1 {
				break
			}
			moves++ // Count move of P2
		}
		// Get valid moves.
		validMoves, err := c.validMoves(gameId)
		if err != nil {
			t.Errorf("could not get valid moves: %v", err)
			break
		}
		if len(validMoves) == 0 {
			t.Errorf("No valid move despite game not having finished")
			break
		}
		// Make move.
		if err := c.makeMove(gameId, validMoves[0]); err != nil {
			t.Errorf("could not make move: %v", err)
			break
		}
	}
	hist, err := c.history(gameId)
	if err != nil {
		t.Fatalf("could not get history: %v", err)
	}
	if hist.GameId != gameId {
		t.Errorf("wrong game ID in history: want %s, got %s", gameId, hist.GameId)
	}
	if len(hist.Entries) != moves+1 {
		// History entries should be the initial board and one per move.
		t.Errorf("Want %d history entries, got %d", moves, len(hist.Entries))
	}
}

func TestFlagzUndo(t *testing.T) {
	if testing.Short() {
		t.Skip("Don't run http tests in -short mode.")
	}
	cfg := testServerConfig(t)
	cfg.CpuThinkTime = 1 * time.Millisecond // We want a fast test, not smart moves.
	srv, err := newTestStatelessServer(t, cfg)
	if err != nil {
		t.Fatal(err)
	}
	testServer := httptest.NewServer(srv.createMux())
	defer testServer.Close()

	c, err := newHexzTestClient(testServer.URL)
	if err != nil {
		t.Fatalf("could not create client: %s", err)
	}
	// Log in.
	if err := c.login("testuser"); err != nil {
		t.Fatal(err)
	}
	// Start a new two player game (so the CPU doesn't interfere).
	gameId, err := c.newFlagzGame(false)
	if err != nil {
		t.Fatal(err)
	}
	// Receive SSE events.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eventCh, err := c.receiveEvents(ctx, testServer.URL+"/hexz/sse/"+gameId)
	if err != nil {
		t.Fatalf("cannot receive events for game %s: %s", gameId, err)
	}
	go func() {
		for range eventCh {
			// Consume and ignore all events
		}
	}()
	// Make one random move:
	validMoves, err := c.validMoves(gameId)
	if err != nil {
		t.Fatalf("could not get valid moves: %v", err)
	}
	if len(validMoves) < 2 {
		t.Fatalf("No valid moves despite game not having finished")
	}
	if err := c.makeMove(gameId, validMoves[0]); err != nil {
		t.Fatalf("could not make move: %v", err)
	}
	// Undo the move:
	if err := c.undo(gameId); err != nil {
		t.Fatalf("undo failed: %v", err)
	}
	// Redo the move:
	if err := c.redo(gameId); err != nil {
		t.Fatalf("undo failed: %v", err)
	}
	// Undo the move (again):
	if err := c.undo(gameId); err != nil {
		t.Fatalf("undo failed: %v", err)
	}
	// Now make a different first move:
	if err := c.makeMove(gameId, validMoves[1]); err != nil {
		t.Fatalf("could not make move: %v", err)
	}
}

// Longish test that starts a server, logs in a new player, starts a new
// single-player flagz game and plays it till the end using random moves.
func TestFlagzSinglePlayer(t *testing.T) {
	if testing.Short() {
		t.Skip("Don't run http tests in -short mode.")
	}
	cfg := testServerConfig(t)
	cfg.CpuThinkTime = 1 * time.Millisecond // We want a fast test, not smart moves.
	srv, err := newTestStatelessServer(t, cfg)
	if err != nil {
		t.Fatal(err)
	}

	testServer := httptest.NewServer(srv.createMux())
	defer testServer.Close()

	c, err := newHexzTestClient(testServer.URL)
	if err != nil {
		t.Fatalf("could not create client: %s", err)
	}
	// Log in.
	if err := c.login("testuser"); err != nil {
		t.Fatal(err)
	}
	// Start a new single player game.
	gameId, err := c.newFlagzGame(true)
	if err != nil {
		t.Fatal(err)
	}
	// Receive SSE events.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eventCh, err := c.receiveEvents(ctx, testServer.URL+"/hexz/sse/"+gameId)
	if err != nil {
		t.Fatalf("cannot receive events for game %s: %s", gameId, err)
	}
	boardCh := receiveBoards(ctx, eventCh)
	<-boardCh // Ignore first broadcast of the initial board.
	finished := false
	maxMoves := numFieldsFirstRow * numBoardRows // upper bound for possible moves
	moves := 0
	for ; !finished && moves < maxMoves; moves++ {
		// Get valid moves.
		validMoves, err := c.validMoves(gameId)
		if err != nil {
			t.Errorf("could not get valid moves: %v", err)
			break
		}
		if len(validMoves) == 0 {
			t.Errorf("No valid move despite game not having finished")
			break
		}
		// Make move.
		if err := c.makeMove(gameId, validMoves[0]); err != nil {
			t.Errorf("could not make move: %v", err)
			break
		}
		// Receive boards until the game is finished or it's our turn again.
		for {
			board := <-boardCh
			if board.State == Finished {
				finished = true
				break
			}
			if board.Turn == 1 {
				break
			}
		}
	}
	if !finished {
		t.Errorf("did not finish the game after %d moves", moves)
	}
}
