package hexz

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestGameMasterRegisterUnregister(t *testing.T) {
	// This test registers two players for a Flagz game.
	// It then immediately unregisters a player.
	// Both players should receive two events:
	// P1: Welcome P1, Welcome P2
	// P2: Welcome P2, Player 1 left.
	ctx, cancel := context.WithCancel(context.Background())
	g := &GameHandle{
		id:           "game1",
		gameType:     gameTypeFlagz,
		host:         "testhost",
		singlePlayer: false,
		ctx:          ctx,
		controlEvent: make(chan ControlEvent),
	}
	cfg := &ServerConfig{
		PlayerRemoveDelay: time.Duration(1) * time.Microsecond,
	}
	s := NewServer(cfg)
	m := NewGameMaster(s, g)
	go m.Run(cancel)
	allEvents := make(chan ServerEvent)
	type eventStats struct {
		events int
		boards int // events with board
	}
	stats := make(chan [3]eventStats)
	// Fan-in receiver of all events.
	go func() {
		var s [3]eventStats
		for e := range allEvents {
			if e.Board != nil {
				s[e.Role].boards++
			}
			s[e.Role].events++
		}
		stats <- s
	}()
	var wg sync.WaitGroup
	playerIds := []string{"P1", "P2"}
	for i := 0; i < 2; i++ {
		wg.Add(1)
		replyChan := make(chan chan ServerEvent)
		// Create one goroutine per player and let them simply forward all events.
		go func() {
			eventCh := <-replyChan
			for e := range eventCh {
				allEvents <- e
			}
			wg.Done()
		}()
		g.controlEvent <- ControlEventRegister{
			player:    Player{Id: PlayerId(playerIds[i]), Name: playerIds[i]},
			replyChan: replyChan,
		}
	}
	// Unregister P1
	g.controlEvent <- ControlEventUnregister{
		playerId: PlayerId(playerIds[0]),
	}
	// Wait for both player goroutines to finish. They should finish, because
	// the GameMaster should close their event channels.
	wg.Wait()
	// Notify fan-in receiver goroutine that we're done.
	close(allEvents)
	st := <-stats

	if want := 2; st[1].events != want {
		t.Errorf("Want %d events, got %d", want, st[1].events)
	}
	if want := 2; st[2].events != want {
		t.Errorf("Want %d events, got %d", want, st[2].events)
	}
	if want := 2; st[1].boards != want {
		t.Errorf("Want %d events with board, got %d", want, st[1].boards)
	}
	if want := 2; st[2].boards != want {
		t.Errorf("Want %d events with board, got %d", want, st[2].boards)
	}
}

func TestGameMasterPlayFullGame(t *testing.T) {
	// This test registers two players for a Classic game.
	// It then plays all moves for a full game.
	// It finally checks that both players received an
	// event indicating the winner.
	ctx, cancel := context.WithCancel(context.Background())
	g := &GameHandle{
		id:           "game1",
		gameType:     gameTypeClassic,
		host:         "testhost",
		singlePlayer: false,
		controlEvent: make(chan ControlEvent),
		ctx:          ctx,
	}
	cfg := &ServerConfig{
		PlayerRemoveDelay: time.Duration(1) * time.Microsecond,
	}
	s := NewServer(cfg)
	m := NewGameMaster(s, g)
	go m.Run(cancel)
	allEvents := make(chan ServerEvent)
	type eventStats struct {
		winner    int
		lastState GameState
		lastEvent ServerEvent
	}
	stats := make(chan [3]eventStats)
	// Fan-in receiver of all events.
	go func() {
		var s [3]eventStats
		for e := range allEvents {
			if e.Winner > 0 {
				s[e.Role].winner = e.Winner
			}
			s[e.Role].lastState = e.Board.State
			s[e.Role].lastEvent = e
		}
		stats <- s
	}()
	var wg sync.WaitGroup
	playerIds := []string{"P1", "P2"}
	for i := 0; i < 2; i++ {
		wg.Add(1)
		replyChan := make(chan chan ServerEvent)
		// Create one goroutine per player and let them simply forward all events.
		go func() {
			eventCh := <-replyChan
			for e := range eventCh {
				allEvents <- e
			}
			wg.Done()
		}()
		g.controlEvent <- ControlEventRegister{
			player:    Player{Id: PlayerId(playerIds[i]), Name: playerIds[i]},
			replyChan: replyChan,
		}
	}
	// Play moves
	b := m.gameEngine.Board()
	n := 0
	for r := range b.Fields {
		for c := range b.Fields[r] {
			g.controlEvent <- ControlEventMove{
				playerId: PlayerId(playerIds[n%2]),
				moveRequest: &MoveRequest{
					Move: n,
					Row:  r,
					Col:  c,
					Type: cellNormal,
				},
			}
			n++
		}
	}
	cancel()

	// Wait for both player goroutines to finish. They should finish, because
	// the GameMaster should close their event channels.
	wg.Wait()
	// Notify fan-in receiver goroutine that we're done.
	close(allEvents)
	st := <-stats

	wantState := Finished
	if got := st[1].lastState; got != wantState {
		t.Errorf("Want final state %s, got %s", wantState, got)
	}

	wantWinner := 1
	if got := st[1].winner; got != wantWinner {
		t.Errorf("Want winner %d, got %d", wantWinner, got)
	}
	if got := st[2].winner; got != wantWinner {
		t.Errorf("Want winner %d, got %d", wantWinner, got)
	}
}

func TestRejoinGame(t *testing.T) {
	// This test checks that a player can rejoin a game if they reconnect
	// before the removal timeout expires.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Terminate the game master goroutine eventually.
	g := &GameHandle{
		id:           "game1",
		gameType:     gameTypeFlagz,
		host:         "testhost",
		singlePlayer: true,
		ctx:          ctx,
		controlEvent: make(chan ControlEvent),
	}
	cfg := &ServerConfig{
		PlayerRemoveDelay: 500 * time.Millisecond, // Should be plenty to allow for a rejoin.
	}
	s := NewServer(cfg)
	m := NewGameMaster(s, g)
	// Start the game.
	done := make(chan struct{}, 1)
	go func() {
		m.Run(cancel)
		done <- struct{}{}
	}()
	// Register player 1 (the only player).
	playerId := PlayerId("P1")
	replyChan := make(chan chan ServerEvent)
	g.controlEvent <- ControlEventRegister{
		player:    Player{Id: playerId, Name: "Horst"},
		replyChan: replyChan,
	}
	eventCh := <-replyChan
	// Expect and ignore the welcome event.
	<-eventCh
	// Unregister player 1. Should not trigger an event.
	g.controlEvent <- ControlEventUnregister{
		playerId: playerId,
	}
	// Re-register player 1 almost immediately.
	time.Sleep(1 * time.Millisecond)
	replyChan = make(chan chan ServerEvent)
	select {
	case g.controlEvent <- ControlEventRegister{
		player:    Player{Id: playerId, Name: "Horst"},
		replyChan: replyChan,
	}:
	case <-done:
		t.Fatal("GameMaster unexpectedly stopped.")
	}
	eventCh = <-replyChan
	<-eventCh
}

func TestRejoinGameTimeout(t *testing.T) {
	// This test checks that a player gets logged out of a game and the game ends
	// if they do not reconnect before the removal timeout.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Terminate the game master goroutine eventually.
	g := &GameHandle{
		id:           "game1",
		gameType:     gameTypeFlagz,
		host:         "testhost",
		singlePlayer: true,
		ctx:          ctx,
		controlEvent: make(chan ControlEvent),
	}
	cfg := &ServerConfig{
		PlayerRemoveDelay: 10 * time.Microsecond, // Log out player quickly for a fast test.
	}
	s := NewServer(cfg)
	m := NewGameMaster(s, g)
	// Start the game.
	done := make(chan struct{}, 1)
	go func() {
		m.Run(cancel)
		done <- struct{}{}
	}()
	// Register player 1 (the only player).
	playerId := PlayerId("P1")
	replyChan := make(chan chan ServerEvent)
	g.controlEvent <- ControlEventRegister{
		player:    Player{Id: playerId, Name: "Horst"},
		replyChan: replyChan,
	}
	eventCh := <-replyChan
	// Expect and ignore the welcome event.
	<-eventCh
	// Unregister player 1. Should not trigger an event.
	g.controlEvent <- ControlEventUnregister{
		playerId: playerId,
	}
	// Try to re-register player 1, but too late.
	time.Sleep(5 * time.Millisecond) // 5ms >> 10us
	replyChan = make(chan chan ServerEvent)
	select {
	case g.controlEvent <- ControlEventRegister{
		player:    Player{Id: playerId, Name: "Horst"},
		replyChan: replyChan,
	}:
		t.Fatal("Player 1 could unexpectedly re-register.")
	case <-done:
	}
}
