package hexz

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

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

func TestSha256HexDigest(t *testing.T) {
	got := sha256HexDigest("foo")
	want := "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
	if got != want {
		t.Errorf("Want: %q, got: %q", want, got)
	}
}

func serverConfigForTest(t *testing.T) *ServerConfig {
	historyRoot := t.TempDir()
	return &ServerConfig{
		ServerAddress:   "localhost",
		ServerPort:      8999,
		DocumentRoot:    "./resources",
		GameHistoryRoot: historyRoot,
		DebugMode:       true,
	}
}

const (
	testPlayerId   = "testId"
	testPlayerName = "tester"
)

func TestHandleNewGame(t *testing.T) {
	cfg := serverConfigForTest(t)
	s := NewServer(cfg)
	if !s.loginPlayer(testPlayerId, testPlayerName) {
		t.Errorf("Cannot log in test player")
	}
	w := httptest.NewRecorder()
	// Create request with login form parameters.
	form := url.Values{}
	form.Add("type", string(gameTypeFlagz))
	form.Add("singlePlayer", "true")
	r := httptest.NewRequest(http.MethodPost, "/hexz/new", strings.NewReader(form.Encode()))
	r.AddCookie(makePlayerCookie(testPlayerId, 24*time.Hour))
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
	if len(s.ongoingGames) != 1 {
		t.Errorf("Ongoing games: %d, want: 1", len(s.ongoingGames))
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
				if e.s.Board != nil && e.s.Board.Move > moveNum {
					moveNum = e.s.Board.Move
					boardCh <- e.s.Board
				}
				// Ignore errors and events without a new board.
			case <-ctx.Done():
				return
			}
		}
	}()
	return boardCh
}

// Longish test that starts a server, logs in a new player, starts a new
// single-player flagz game and plays it till the end using random moves.
func TestFlagzSinglePlayer(t *testing.T) {
	if testing.Short() {
		return // This test takes >1s...
	}
	cfg := serverConfigForTest(t)
	cfg.CpuThinkTime = 1 * time.Millisecond // We want a fast test, not smart moves.
	srv := NewServer(cfg)
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
		t.Fatal("cannot receive events:", err)
	}
	boardCh := receiveBoards(ctx, eventCh)
	<-boardCh // Ignore first broadcast of the initial board.
	finished := false
	maxMoves := numFieldsFirstRow * numBoardRows // upper bound for possible moves
	for i := 0; !finished && i < maxMoves; i++ {
		// Get valid moves.
		validMoves, err := c.validMoves(gameId)
		if err != nil {
			t.Error(err)
			break
		}
		if len(validMoves) == 0 {
			t.Errorf("No valid move despite game not having finished")
			break
		}
		// Make move.
		if err := c.makeMove(gameId, validMoves[0]); err != nil {
			t.Error(err)
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
		t.Errorf("did not finish the game after %d moves", maxMoves)
	}
}

// History should be written for each move and be available once the game
// is finished.
func TestFlagzSinglePlayerHistory(t *testing.T) {
	histDir := t.TempDir()
	if testing.Short() {
		return // This test takes >1s...
	}
	cfg := serverConfigForTest(t)
	cfg.CpuThinkTime = 1 * time.Millisecond // We want a fast test, not smart moves.
	cfg.GameHistoryRoot = histDir
	srv := NewServer(cfg)
	testServer := httptest.NewServer(srv.createMux())
	defer testServer.Close()

	c, err := newHexzTestClient(testServer.URL)
	if err != nil {
		t.Fatalf("could not create client: %s", err)
	}
	// Log in.
	testuser := "testuser"
	if err := c.login(testuser); err != nil {
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
		t.Fatal("cannot receive events:", err)
	}
	boardCh := receiveBoards(ctx, eventCh)
	<-boardCh // Ignore first broadcast of the initial board.
	finished := false
	maxMoves := numFieldsFirstRow * numBoardRows // upper bound for possible moves
	for i := 0; !finished && i < maxMoves; i++ {
		// Get valid moves.
		validMoves, err := c.validMoves(gameId)
		if err != nil {
			t.Error(err)
			break
		}
		if len(validMoves) == 0 {
			t.Errorf("No valid move despite game not having finished")
			break
		}
		// Make move.
		if err := c.makeMove(gameId, validMoves[0]); err != nil {
			t.Error(err)
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
		t.Fatalf("did not finish the game after %d moves", maxMoves)
	}
	// Finally: Read game history.
	hist, err := c.history(gameId)
	if err != nil {
		t.Fatal("failed to get game history: ", err)
	}
	if len(hist.Entries) < 50 {
		t.Errorf("wrong number of history entries: want >= 50, got %d", len(hist.Entries))
	}
	if hist.GameId != gameId {
		t.Errorf("wrong gameId in history: want %s, got %s", gameId, hist.GameId)
	}
	if hist.GameType != gameTypeFlagz {
		t.Errorf("wrong game type in history: want %s, got %s", gameTypeFlagz, hist.GameType)
	}
	if diff := cmp.Diff([]string{testuser, "Computer"}, hist.PlayerNames); diff != "" {
		t.Errorf("wrong player names in history: -want +got: %s", diff)
	}
}
