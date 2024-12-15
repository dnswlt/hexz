package hexz

import (
	"context"
	"flag"
	"fmt"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/dnswlt/hexz/internal/hexzmem"
	"github.com/dnswlt/hexz/internal/hexzsql"
	"github.com/dnswlt/hexz/pkg/hexzpb"
)

var (
	// This is typically going to be postgres://hexz_test:hexz_test@localhost:5432/hexz_test
	testPostgresURL = flag.String("test-postgres-url", "", "URL to PostgreSQL DB for integration tests")
	testRedisAddr   = flag.String("test-redis-addr", "", "Address of the (e.g. \"localhost:6379\") for integration tests")
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
		RedisAddr:         *testRedisAddr,
		PostgresURL:       *testPostgresURL,
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
	playerStore, err := hexzmem.NewInMemoryPlayerStore(config.LoginTTL)
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

func TestValidPlayerName(t *testing.T) {
	tests := []struct {
		name   string
		accept bool
	}{
		{"abc", true},
		{"abc.def", true},
		{"abc_def-123", true},
		{"1digit", true},
		{"HANS", true},
		{"Mørän", true},
		{"Jérôme", true},
		{"Strüßenbähn", true},
		{"My Best", true},
		{"My  Best", false}, // No consecutive spaces in the middle
		{"123", false},      // Need at least one latin character
		{"_letter-or.digit", true},
		{"ab", false},       // Too short
		{"jens$", false},    // Invalid character
		{"dw@best", false},  // Invalid character
		{" voodoo ", false}, // Spaces at the ends
		{"", false},
		{"verylongusernamesarenotallowedalright", false},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			if err := validatePlayerName(test.name); (err == nil) != test.accept {
				t.Errorf("unexpected error result for name %s: %v", test.name, err)
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
	testServer := httptest.NewServer(s.createMux())
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
	if _, err := c.newFlagzGame(true); err != nil {
		t.Fatal(err)
	}
}

func TestListActiveGames1P(t *testing.T) {
	if testing.Short() {
		t.Skip("Don't run http tests in -short mode.")
	}
	cfg := testServerConfig(t)
	s, err := newTestStatelessServer(t, cfg)
	if err != nil {
		t.Fatal("Could not create server:", err)
	}
	testServer := httptest.NewServer(s.createMux())
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
	if _, err := c.newFlagzGame(true); err != nil {
		t.Fatal(err)
	}
	// Should not be listed under open games: it's 1P.
	openGames, err := s.gameStore.ListOpenGames(context.Background(), 100)
	if err != nil {
		t.Fatal("Could not get active games:", err)
	}
	if len(openGames) != 0 {
		t.Errorf("Open games: %d, want: 0", len(openGames))
	}
	// Should be listed under active games immediately.
	activeGames, err := s.gameStore.ListActiveGames(context.Background(), 100)
	if err != nil {
		t.Fatal("Could not get active games:", err)
	}
	if len(activeGames) != 1 {
		t.Errorf("Open games: %d, want: 1", len(activeGames))
	}
}

func TestListActiveGames2P(t *testing.T) {
	// Starts a 2 player game. Expects the game to appear as an open game after creation,
	// and as an active game after the second player joined.
	if testing.Short() {
		t.Skip("Don't run http tests in -short mode.")
	}
	cfg := testServerConfig(t)
	s, err := newTestStatelessServer(t, cfg)
	if err != nil {
		t.Fatal("Could not create server:", err)
	}
	testServer := httptest.NewServer(s.createMux())
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
	if _, err := c.newFlagzGame(false); err != nil {
		t.Fatal(err)
	}
	// Should be listed under open games: it's 1P.
	openGames, err := c.openGames()
	if err != nil {
		t.Fatal("Could not get open games:", err)
	}
	if len(openGames) != 1 {
		t.Fatalf("Open games: %d, want: 1", len(openGames))
	}
	// Should not be listed under active games immediately.
	activeGames, err := c.activeGames()
	if err != nil {
		t.Fatal("Could not get active games:", err)
	}
	if len(activeGames) != 0 {
		t.Errorf("Active games: %d, want: 0", len(activeGames))
	}
	ctx1, cancel1 := context.WithCancel(context.Background())
	defer cancel1()
	_, err = c.receiveEvents(ctx1, testServer.URL+"/hexz/sse/"+openGames[0].Id)
	if err != nil {
		t.Fatalf("Could not receive events for P1: %v", err)
	}
	// Second player joins.
	c2, err := newHexzTestClient(testServer.URL)
	if err != nil {
		t.Fatalf("could not create 2nd client: %s", err)
	}
	if err := c2.login("testuser_2"); err != nil {
		t.Fatal(err)
	}
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	_, err = c2.receiveEvents(ctx2, testServer.URL+"/hexz/sse/"+openGames[0].Id)
	if err != nil {
		t.Fatalf("Could not receive events for P2: %v", err)
	}

	// Now the game should no longer be open.
	openGames, err = c.openGames()
	if err != nil {
		t.Fatal("Could not get open games:", err)
	}
	if len(openGames) != 0 {
		t.Fatalf("Open games: %d, want: 0", len(openGames))
	}
	// Should now be listed under active games.
	activeGames, err = c.activeGames()
	if err != nil {
		t.Fatal("Could not get active games:", err)
	}
	if len(activeGames) != 1 {
		t.Errorf("Active games: %d, want: 1", len(activeGames))
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
	if *testPostgresURL == "" {
		t.Skip("--test-postgres-url is not set, skipping database integration test")
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
	if err := c.undo(gameId, 1); err != nil {
		t.Fatalf("undo failed: %v", err)
	}
	// Redo the move:
	if err := c.redo(gameId, 0); err != nil {
		t.Fatalf("undo failed: %v", err)
	}
	// Undo the move (again):
	if err := c.undo(gameId, 1); err != nil {
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
