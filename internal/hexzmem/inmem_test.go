package hexzmem

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	"google.golang.org/protobuf/testing/protocmp"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

func TestInMemStoreGameNoGameInfo(t *testing.T) {
	gameTTL := 5 * time.Minute
	store := NewInMemoryGameStore(gameTTL)
	ctx := context.Background()
	state := &pb.GameState{}
	err := store.StoreNewGame(ctx, state)
	if err == nil {
		t.Errorf("Expected error saving game state without GameInfo, got none")
	}
}

func TestInMemStoreLoadGame(t *testing.T) {
	gameTTL := 5 * time.Minute
	store := NewInMemoryGameStore(gameTTL)
	ctx := context.Background()
	state := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Host: "horst",
		},
	}
	err := store.StoreNewGame(ctx, state)
	if err != nil {
		t.Error("Error storing new game")
	}
	if state.GameInfo.Id == "" {
		t.Error("GameInfo.Id not set after store")
	}
	if state.GameInfo.PubsubChannelId == "" {
		t.Error("GameInfo.PubsubChannelId not set after store")
	}
	if state.Modified == nil {
		t.Error(".Modified not set after store")
	}
	state2, err := store.LookupGame(ctx, state.GameInfo.Id)
	if err != nil {
		t.Fatal("LookupGame returned error: ", err)
	}
	if diff := cmp.Diff(state2, state, protocmp.Transform()); diff != "" {
		t.Errorf("Loaded game state differs from origina (-want, +got): %s", diff)
	}
}

func TestInMemPubsub(t *testing.T) {
	gameTTL := 5 * time.Minute
	store := NewInMemoryGameStore(gameTTL)
	ctx, cancel := context.WithCancel(context.Background())
	pubsubId := GeneratePubsubID()
	nSubscribers := 2
	results := make(chan int)
	var wgStart sync.WaitGroup
	var wgEnd sync.WaitGroup
	for i := 0; i < nSubscribers; i++ {
		wgStart.Add(1)
		wgEnd.Add(1)
		go func() {
			events := 0
			ch := store.Subscribe(ctx, pubsubId)
			wgStart.Done()
			for range ch {
				events++
			}
			results <- events
		}()
	}
	wgStart.Wait() // Wait until all subscribers have subscribed.
	event := &pb.GameStorePubsubEvent{
		GameId: "hello",
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	}
	if err := store.Publish(ctx, pubsubId, event); err != nil {
		t.Fatalf("Failed to publish event: %v", err)
	}
	// Since Publish in the InMemoryStore is synchronous, i.e. all intermediary
	// goroutines (one per subscriber) must have received the published message,
	// we can immediately cancel the context, which will close all subscriber's channels.
	cancel()
	n1 := <-results
	n2 := <-results
	wantN := 1
	if n1 != wantN || n2 != wantN {
		t.Errorf("Want %d events per subscriber, got %d and %d", wantN, n1, n2)
	}
}

func TestInMemListOpenGamesCleanup(t *testing.T) {
	gameTTL := 12 * time.Hour
	store := NewInMemoryGameStore(gameTTL)
	fakeClock := NewFakeClock(time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC))
	store.clock = fakeClock
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// Add one game every hour, 24 hours.
	for i := 0; i < 24; i++ {
		gameId := fmt.Sprintf("%d", i)
		gameState := &pb.GameState{
			GameInfo: &pb.GameInfo{
				Id:      gameId,
				Started: tpb.New(fakeClock.Now()),
			},
		}
		err := store.StoreNewGame(ctx, gameState)
		if err != nil {
			t.Fatal("Failed to store game: ", err)
		}
		fakeClock.Advance(1 * time.Hour)
	}
	// List open games. Only the last 11 should be returned.
	games, err := store.ListOpenGames(ctx, 24)
	if err != nil {
		t.Fatal("Failed to list recent games: ", err)
	}
	wantGames := 11
	if len(games) != wantGames {
		t.Errorf("Want %d games, got %d", wantGames, len(games))
	}
	// We only want 12 games in the /opengames set now in Redis.
	n := len(store.gameStatesSeq)
	if n != wantGames {
		t.Errorf("Want %d games in gameStatesSeq, got %d", wantGames, n)
	}
}

func TestInMemFlashStore(t *testing.T) {
	s := NewInMemFlashStore()
	flashID := uuid.New().String()
	for i := 0; i < 4; i++ {
		err := s.AddMessage(context.Background(), flashID, FlashMessage{
			Kind:    "Test",
			Created: time.Date(2024, 1, 1, 12, 1, 0, 0, time.UTC),
		})
		if err != nil {
			t.Fatalf("AddMessage failed: %v", err)
		}
	}
	messages, err := s.PopMessages(context.Background(), flashID)
	if err != nil {
		t.Fatalf("PopMessages failed: %v", err)
	}
	if len(messages) != 4 {
		t.Errorf("Wrong number of messages: want 4, got %d", len(messages))
	}
	// Should get 0 messages second time around.
	messages, err = s.PopMessages(context.Background(), flashID)
	if err != nil {
		t.Fatalf("PopMessages failed: %v", err)
	}
	if len(messages) != 0 {
		t.Errorf("Wrong number of messages: want 0, got %d", len(messages))
	}

}
