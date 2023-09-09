package hexz

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"
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

type SSEClient struct {
	client *http.Client
}

func newSSEClient() (*SSEClient, error) {
	jar, err := cookiejar.New(nil)
	if err != nil {
		return nil, err
	}
	return &SSEClient{
		client: &http.Client{
			Jar: jar, // Set cookie jar for passing along playerId
		},
	}, nil
}

// Establishes a SSE connection with the given URL and sends the "data:" part
// of each event through the returned channel. Clients should close the done channel
// to abort the SSE connection.
func (c *SSEClient) receiveEvents(url string, done <-chan tok) (<-chan string, error) {
	ch := make(chan string)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Accept", "text/event-stream")
	req.Header.Add("Connection", "keep-alive")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("received status %s: %s", http.StatusText(resp.StatusCode), msg)
	}
	go func() {
		// Close connection if we're done processing events.
		// Use a separate goroutine b/c the receiving one will
		// be blocked in the scanner, waiting for more data.
		<-done
		resp.Body.Close()
	}()
	go func() {
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" || line[0] == ':' {
				// SSE separates messages by empty lines. Lines starting with a colon are ignored.
				continue
			}
			dataPrefix := "data: "
			if strings.HasPrefix(line, dataPrefix) {
				select {
				case <-done:
					return
				case ch <- line[len(dataPrefix):]:
					break
				}
			}
			// Ignore all other lines for now.
		}
	}()
	return ch, nil
}

// Process SSE ServerEvents and return boards with strictly monotonically increasing move numbers.
func receiveBoards(eventCh <-chan string, done <-chan tok) <-chan *BoardView {
	boardCh := make(chan *BoardView)
	go func() {
		defer close(boardCh)
		moveNum := -1
		for {
			select {
			case e := <-eventCh:
				var se ServerEvent
				if err := json.Unmarshal([]byte(e), &se); err != nil {
					continue // Ignore everything else
				}
				if se.Board != nil && se.Board.Move > moveNum {
					moveNum = se.Board.Move
					boardCh <- se.Board
				}
			case <-done:
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

	c, err := newSSEClient()
	if err != nil {
		t.Fatalf("could not create SSEClient: %s", err)
	}
	// Log in.
	loginForm := url.Values{}
	loginForm.Add("name", "testsse")
	loginResp, err := c.client.PostForm(testServer.URL+"/hexz/login", loginForm)
	if err != nil {
		t.Fatalf("could not log in: %s", err)
	}
	if loginResp.StatusCode != http.StatusOK {
		t.Fatalf("could not log in: %s", loginResp.Status)
	}
	// Start a new single player game.
	newGameForm := url.Values{}
	newGameForm.Add("type", string(gameTypeFlagz))
	newGameForm.Add("singlePlayer", "true")
	newGameResp, err := c.client.PostForm(testServer.URL+"/hexz/new", newGameForm)
	if err != nil {
		t.Fatalf("could not start new game: %s", err)
	}
	if newGameResp.StatusCode != http.StatusOK {
		t.Fatalf("could not start new game: %s", newGameResp.Status)
	}
	gameId := gameIdFromPath(newGameResp.Request.URL.Path)

	// Receive SSE events.
	done := make(chan tok)
	defer close(done) // Used to tell client to stop receiving SSE events.
	eventCh, err := c.receiveEvents(testServer.URL+"/hexz/sse/"+gameId, done)
	if err != nil {
		t.Fatal("cannot receive events:", err)
	}
	boardCh := receiveBoards(eventCh, done)
	<-boardCh // Ignore first broadcast of the initial board.
	finished := false
	maxMoves := numFieldsFirstRow * numBoardRows // upper bound for possible moves
	for i := 0; i < maxMoves; i++ {
		// Request valid move.
		validMovesResp, err := c.client.Get(testServer.URL + "/hexz/moves/" + gameId)
		if err != nil {
			t.Fatal("cannot get valid moves: ", err)
		}
		if validMovesResp.StatusCode != http.StatusOK {
			t.Fatal("cannot get valid moves: ", validMovesResp.Status)
		}
		dec := json.NewDecoder(validMovesResp.Body)
		var validMoves []*MoveRequest
		if err := dec.Decode(&validMoves); err != nil {
			t.Fatal("cannot unmarshal MoveRequest: ", err)
		}
		// Make move.
		if len(validMoves) == 0 {
			break
		}
		data, _ := json.Marshal(validMoves[0])
		resp, err := c.client.Post(testServer.URL+"/hexz/move/"+gameId, "application/json", bytes.NewReader(data))
		if err != nil {
			t.Fatal("failed to make move: ", err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatal("failed to make move: ", resp.Status)
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
