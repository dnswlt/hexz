package hexzmem

import (
	"context"
	"flag"
	"fmt"
	"regexp"
	"testing"
	"time"

	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

var (
	testRedisAddr = flag.String("test-redis-addr", "", "Address of Redis server used for integration tests")
)

type FakeClock struct {
	now time.Time
}

func (c *FakeClock) Now() time.Time          { return c.now }
func (c *FakeClock) SetNow(now time.Time)    { c.now = now }
func (c *FakeClock) Advance(d time.Duration) { c.now = c.now.Add(d) }
func NewFakeClock(now time.Time) *FakeClock  { return &FakeClock{now: now} }

func TestGenerateGameId(t *testing.T) {
	got := GenerateGameID()
	if !regexp.MustCompile(`^[A-Z]{6}$`).MatchString(got) {
		t.Errorf("Wrong gameId: %q", got)
	}
}

func TestStoreGameNoGameInfo(t *testing.T) {
	if *testRedisAddr == "" {
		t.Skip("Skipping integration test because -test-redis-addr is not set")
	}
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     *testRedisAddr,
		LoginTTL: 24 * time.Hour,
		GameTTL:  24 * time.Hour,
		DB:       1, // Use test DB
	})
	if err != nil {
		t.Fatal("Failed to connect to Redis: ", err)
	}
	defer rc.client.Close()
	ctx := context.Background()
	state := &pb.GameState{}
	err = rc.StoreNewGame(ctx, state)
	if err == nil {
		t.Errorf("Expected error saving game state without GameInfo, got none")
	}
}

func TestStoreLoadGame(t *testing.T) {
	if *testRedisAddr == "" {
		t.Skip("Skipping integration test because -test-redis-addr is not set")
	}
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     *testRedisAddr,
		LoginTTL: 24 * time.Hour,
		GameTTL:  24 * time.Hour,
		DB:       1, // Use test DB
	})
	if err != nil {
		t.Fatal("Failed to connect to Redis: ", err)
	}
	defer rc.client.Close()
	ctx := context.Background()
	state := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Host: "horst",
		},
	}
	err = rc.StoreNewGame(ctx, state)
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
	state2, err := rc.LookupGame(ctx, state.GameInfo.Id)
	if err != nil {
		t.Fatal("LookupGame returned error: ", err)
	}
	if diff := cmp.Diff(state2, state, protocmp.Transform()); diff != "" {
		t.Errorf("Loaded game state differs from origina (-want, +got): %s", diff)
	}
}

func TestRedisPubsub(t *testing.T) {
	if *testRedisAddr == "" {
		t.Skip("Skipping integration test because -test-redis-addr is not set")
	}
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     *testRedisAddr,
		LoginTTL: 24 * time.Hour,
		GameTTL:  24 * time.Hour,
		DB:       1, // Use test DB
	})
	if err != nil {
		t.Fatal("Failed to connect to Redis: ", err)
	}
	defer rc.client.Close()
	ctx, cancel := context.WithCancel(context.Background())
	pubsubId := GeneratePubsubID()
	nSubscribers := 2
	results := make(chan int)
	for i := 0; i < nSubscribers; i++ {
		go func() {
			events := 0
			ch := rc.Subscribe(ctx, pubsubId)
			for range ch {
				events++
			}
			results <- events
		}()
	}
	// Wait for all subscribers to be ready. We cannot synchronize this properly,
	// b/c even the Redis client's Subscribe method returns before the subscription might be active.
	time.Sleep(500 * time.Millisecond)
	event := &pb.GameStorePubsubEvent{
		GameId: "hello",
		Event:  &pb.GameStorePubsubEvent_GameUpdated_{},
	}
	if err := rc.Publish(ctx, pubsubId, event); err != nil {
		t.Fatalf("Failed to publish event: %v", err)
	}
	time.Sleep(500 * time.Millisecond)
	cancel()
	n1 := <-results
	n2 := <-results
	wantN := 1
	if n1 != wantN || n2 != wantN {
		t.Errorf("Want %d events per subscriber, got %d and %d", wantN, n1, n2)
	}
}

func TestRedisListOpenGamesCleanup(t *testing.T) {
	*testRedisAddr = "localhost:6379"
	// White-box test. Checks that old and deleted entries get purged.
	if *testRedisAddr == "" {
		t.Skip("Skipping integration test because -test-redis-addr is not set")
	}
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     *testRedisAddr,
		LoginTTL: 24 * time.Hour,
		GameTTL:  12 * time.Hour,
		DB:       1, // Use test DB
	})
	fakeClock := NewFakeClock(time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC))
	rc.clock = fakeClock
	if err != nil {
		t.Fatal("Failed to connect to Redis: ", err)
	}
	defer rc.client.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := rc.client.FlushDB(ctx).Err(); err != nil {
		t.Fatal("Failed to flush Redis DB: ", err)
	}
	// Add one game every hour, 24 hours.
	for i := 0; i < 24; i++ {
		gameId := fmt.Sprintf("%d", i)
		gameState := &pb.GameState{
			GameInfo: &pb.GameInfo{
				Id:      gameId,
				Started: tpb.New(fakeClock.Now()),
			},
		}
		err := rc.StoreNewGame(ctx, gameState)
		if err != nil {
			t.Fatal("Failed to store game: ", err)
		}
		fakeClock.Advance(1 * time.Hour)
	}
	// List open games. Only the last 11 should be returned.
	games, err := rc.ListOpenGames(ctx, 24)
	if err != nil {
		t.Fatal("Failed to list recent games: ", err)
	}
	wantGames := 11
	if len(games) != wantGames {
		t.Errorf("Want %d games, got %d", wantGames, len(games))
	}
	// We only want 12 games in the /opengames set now in Redis.
	n, _ := rc.client.ZCard(ctx, "/opengames").Result()
	if n != int64(wantGames) {
		t.Errorf("Want %d games in /opengames set, got %d", wantGames, n)
	}
}

func TestRedisListOpenGamesSkipActive(t *testing.T) {
	// White-box test.
	// Checks that games that have become active in the meantime are not returned as open games.
	if *testRedisAddr == "" {
		t.Skip("Skipping integration test because -test-redis-addr is not set")
	}
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     *testRedisAddr,
		LoginTTL: 24 * time.Hour,
		GameTTL:  12 * time.Hour,
		DB:       1, // Use test DB
	})
	fakeClock := NewFakeClock(time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC))
	rc.clock = fakeClock
	if err != nil {
		t.Fatal("Failed to connect to Redis: ", err)
	}
	defer rc.client.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := rc.client.FlushDB(ctx).Err(); err != nil {
		t.Fatal("Failed to flush Redis DB: ", err)
	}
	started := rc.clock.Now()
	gameState := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Id:      "game",
			Started: tpb.New(started),
		},
	}
	if err := rc.StoreNewGame(ctx, gameState); err != nil {
		t.Fatal("Failed to store game: ", err)
	}
	// Mark the game as active:
	gameState.AllPlayersJoined = true
	rc.UpdateGame(ctx, gameState)
	games, err := rc.ListOpenGames(ctx, 1)
	if err != nil {
		t.Fatal("Failed to list recent games: ", err)
	}
	if len(games) != 0 {
		t.Fatalf("Want 0 games, got %d", len(games))
	}
	n, _ := rc.client.ZCard(ctx, "/opengames").Result()
	if n != 0 {
		t.Errorf("Want %d games in /opengames set, got %d", 0, n)
	}
}

func TestRedisListOpenGamesSkipDeleted(t *testing.T) {
	// Checks that deleted games are not returned as open games.
	if *testRedisAddr == "" {
		t.Skip("Skipping integration test because -test-redis-addr is not set")
	}
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     *testRedisAddr,
		LoginTTL: 24 * time.Hour,
		GameTTL:  12 * time.Hour,
		DB:       1, // Use test DB
	})
	fakeClock := NewFakeClock(time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC))
	rc.clock = fakeClock
	if err != nil {
		t.Fatal("Failed to connect to Redis: ", err)
	}
	defer rc.client.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := rc.client.FlushDB(ctx).Err(); err != nil {
		t.Fatal("Failed to flush Redis DB: ", err)
	}
	started := rc.clock.Now()
	// Add one game every hour, 24 hours.
	var gameIds []string
	for i := 0; i < 24; i++ {
		gameId := fmt.Sprintf("%d", i)
		gameIds = append(gameIds, gameId)
		gameState := &pb.GameState{
			GameInfo: &pb.GameInfo{
				Id:      gameId,
				Started: tpb.New(started.Add(time.Duration(i) * time.Hour)),
			},
		}
		err := rc.StoreNewGame(ctx, gameState)
		if err != nil {
			t.Fatal("Failed to store game: ", err)
		}
		fakeClock.Advance(1 * time.Hour)
	}
	// Delete the last game that was added.
	if err := rc.DeleteGame(ctx, gameIds[len(gameIds)-1]); err != nil {
		t.Fatalf("Could not delete game %s: %v", gameIds[len(gameIds)-1], err)
	}
	// List open games. Only the last 10 should be returned.
	games, err := rc.ListOpenGames(ctx, 1)
	if err != nil {
		t.Fatal("Failed to list recent games: ", err)
	}
	if len(games) != 1 {
		t.Fatalf("Want 1 game, got %d", len(games))
	}
	if games[0].Id != gameIds[len(gameIds)-2] {
		t.Errorf("ListOpenGames returned wrong gameId: want %s, got %s", gameIds[len(gameIds)-2], games[0].Id)
	}
}
