package hexz

import (
	"testing"
)

func TestSha256HexDigest(t *testing.T) {
	got := sha256HexDigest("foo")
	want := "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
	if got != want {
		t.Errorf("Want: %q, got: %q", want, got)
	}
}

// // Process SSE ServerEvents and return boards with strictly monotonically increasing move numbers.
// func receiveBoards(ctx context.Context, eventCh <-chan tcServerEvent) <-chan *BoardView {
// 	boardCh := make(chan *BoardView)
// 	go func() {
// 		defer close(boardCh)
// 		moveNum := -1
// 		for {
// 			select {
// 			case e := <-eventCh:
// 				if e.s.Board != nil && e.s.Board.Move > moveNum {
// 					moveNum = e.s.Board.Move
// 					boardCh <- e.s.Board
// 				}
// 				// Ignore errors and events without a new board.
// 			case <-ctx.Done():
// 				return
// 			}
// 		}
// 	}()
// 	return boardCh
// }

// // Longish test that starts a server, logs in a new player, starts a new
// // single-player flagz game and plays it till the end using random moves.
// func TestFlagzSinglePlayer(t *testing.T) {
// 	if testing.Short() {
// 		t.Skip("Don't run http tests in -short mode.")
// 	}
// 	cfg := serverConfigForTest(t)
// 	cfg.CpuThinkTime = 1 * time.Millisecond // We want a fast test, not smart moves.
// 	srv, _ := NewServer(cfg)
// 	testServer := httptest.NewServer(srv.createMux())
// 	defer testServer.Close()

// 	c, err := newHexzTestClient(testServer.URL)
// 	if err != nil {
// 		t.Fatalf("could not create client: %s", err)
// 	}
// 	// Log in.
// 	if err := c.login("testuser"); err != nil {
// 		t.Fatal(err)
// 	}
// 	// Start a new single player game.
// 	gameId, err := c.newFlagzGame(true)
// 	if err != nil {
// 		t.Fatal(err)
// 	}
// 	// Receive SSE events.
// 	ctx, cancel := context.WithCancel(context.Background())
// 	defer cancel()
// 	eventCh, err := c.receiveEvents(ctx, testServer.URL+"/hexz/sse/"+gameId)
// 	if err != nil {
// 		t.Fatalf("cannot receive events for game %s: %s", gameId, err)
// 	}
// 	boardCh := receiveBoards(ctx, eventCh)
// 	<-boardCh // Ignore first broadcast of the initial board.
// 	finished := false
// 	maxMoves := numFieldsFirstRow * numBoardRows // upper bound for possible moves
// 	for i := 0; !finished && i < maxMoves; i++ {
// 		// Get valid moves.
// 		validMoves, err := c.validMoves(gameId)
// 		if err != nil {
// 			t.Error(err)
// 			break
// 		}
// 		if len(validMoves) == 0 {
// 			t.Errorf("No valid move despite game not having finished")
// 			break
// 		}
// 		// Make move.
// 		if err := c.makeMove(gameId, validMoves[0]); err != nil {
// 			t.Error(err)
// 			break
// 		}
// 		// Receive boards until the game is finished or it's our turn again.
// 		for {
// 			board := <-boardCh
// 			if board.State == Finished {
// 				finished = true
// 				break
// 			}
// 			if board.Turn == 1 {
// 				break
// 			}
// 		}
// 	}
// 	if !finished {
// 		t.Errorf("did not finish the game after %d moves", maxMoves)
// 	}
// }

// // History should be written for each move and be available once the game
// // is finished.
// func TestFlagzSinglePlayerHistory(t *testing.T) {
// 	histDir := t.TempDir()
// 	if testing.Short() {
// 		t.Skip("Don't run http tests in -short mode.")
// 	}
// 	cfg := serverConfigForTest(t)
// 	cfg.CpuThinkTime = 1 * time.Millisecond // We want a fast test, not smart moves.
// 	cfg.GameHistoryRoot = histDir
// 	srv, _ := NewServer(cfg)
// 	testServer := httptest.NewServer(srv.createMux())
// 	defer testServer.Close()

// 	c, err := newHexzTestClient(testServer.URL)
// 	if err != nil {
// 		t.Fatalf("could not create client: %s", err)
// 	}
// 	// Log in.
// 	testuser := "testuser"
// 	if err := c.login(testuser); err != nil {
// 		t.Fatal(err)
// 	}
// 	// Start a new single player game.
// 	gameId, err := c.newFlagzGame(true)
// 	if err != nil {
// 		t.Fatal(err)
// 	}
// 	// Receive SSE events.
// 	ctx, cancel := context.WithCancel(context.Background())
// 	defer cancel()
// 	eventCh, err := c.receiveEvents(ctx, testServer.URL+"/hexz/sse/"+gameId)
// 	if err != nil {
// 		t.Fatal("cannot receive events:", err)
// 	}
// 	boardCh := receiveBoards(ctx, eventCh)
// 	<-boardCh // Ignore first broadcast of the initial board.
// 	finished := false
// 	maxMoves := numFieldsFirstRow * numBoardRows // upper bound for possible moves
// 	nMoves := 0
// 	for i := 0; !finished && i < maxMoves; i++ {
// 		// Get valid moves.
// 		validMoves, err := c.validMoves(gameId)
// 		if err != nil {
// 			t.Error(err)
// 			break
// 		}
// 		if len(validMoves) == 0 {
// 			t.Errorf("No valid move despite game not having finished")
// 			break
// 		}
// 		// Make move.
// 		if err := c.makeMove(gameId, validMoves[0]); err != nil {
// 			t.Error(err)
// 			break
// 		}
// 		// Receive boards until the game is finished or it's our turn again.
// 		for {
// 			board := <-boardCh
// 			nMoves++
// 			if board.State == Finished {
// 				finished = true
// 				break
// 			}
// 			if board.Turn == 1 {
// 				break
// 			}
// 		}
// 	}
// 	if !finished {
// 		t.Fatalf("did not finish the game after %d moves", maxMoves)
// 	}
// 	// Finally: Read game history.
// 	hist, err := c.history(gameId)
// 	if err != nil {
// 		t.Fatal("failed to get game history: ", err)
// 	}
// 	if len(hist.Entries) != nMoves {
// 		t.Errorf("wrong number of history entries: want %d, got %d", nMoves, len(hist.Entries))
// 	}
// 	if hist.GameId != gameId {
// 		t.Errorf("wrong gameId in history: want %s, got %s", gameId, hist.GameId)
// 	}
// 	if hist.GameType != gameTypeFlagz {
// 		t.Errorf("wrong game type in history: want %s, got %s", gameTypeFlagz, hist.GameType)
// 	}
// 	if diff := cmp.Diff([]string{testuser, "CPU"}, hist.PlayerNames); diff != "" {
// 		t.Errorf("wrong player names in history: -want +got: %s", diff)
// 	}
// }
