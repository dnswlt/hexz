package hexzmem

import (
	"context"
	"flag"
	"fmt"
	"testing"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

var (
	testRedisAddr = flag.String("test-redis-addr", "", "Address of Redis server used for integration tests")
)

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
	gameId := "123"
	nSubscribers := 2
	results := make(chan int)
	for i := 0; i < nSubscribers; i++ {
		go func() {
			events := 0
			ch := make(chan string)
			go rc.Subscribe(ctx, gameId, ch)
			for range ch {
				events++
			}
			results <- events
		}()
	}
	// Wait for all subscribers to be ready. We cannot synchronize this properly,
	// b/c even the Redis client's Subscribe method returns before the subscription might be active.
	time.Sleep(500 * time.Millisecond)
	if err := rc.Publish(ctx, gameId, "hello"); err != nil {
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

func TestRedisListRecentGamesCleanup(t *testing.T) {
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
	defer cancel()
	if err := rc.client.FlushDB(ctx).Err(); err != nil {
		t.Fatal("Failed to flush Redis DB: ", err)
	}
	started := tpb.Now()
	for i := 0; i < 110; i++ {
		gameId := fmt.Sprintf("%d", i)
		gameState := &pb.GameState{
			GameInfo: &pb.GameInfo{
				Id:      gameId,
				Started: started,
			},
		}
		ok, err := rc.StoreNewGame(ctx, gameState)
		if err != nil {
			t.Fatal("Failed to store game: ", err)
		}
		if !ok {
			t.Fatal("Game already exists")
		}
	}
	games, err := rc.ListRecentGames(ctx, 10)
	if err != nil {
		t.Fatal("Failed to list recent games: ", err)
	}
	if len(games) != 10 {
		t.Errorf("Want %d games, got %d", 10, len(games))
	}
	// We only want 20 games in the list now (2*limit). White-box test.
	n, _ := rc.client.ZCard(ctx, "recentgames").Result()
	if n != 20 {
		t.Errorf("Want %d games in recent games list, got %d", 20, n)
	}
}

func TestRedisListRecentGamesTTL(t *testing.T) {
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
	defer cancel()
	if err := rc.client.FlushDB(ctx).Err(); err != nil {
		t.Fatal("Failed to flush Redis DB: ", err)
	}
	now := time.Now()
	started := []*tpb.Timestamp{
		tpb.New(now),
		tpb.New(now.Add(-1 * time.Minute)),
		tpb.New(now.Add(-48 * time.Hour)),
	}
	for i, st := range started {
		gameId := fmt.Sprintf("%d", i)
		gameState := &pb.GameState{
			GameInfo: &pb.GameInfo{
				Id:      gameId,
				Started: st,
			},
		}
		ok, err := rc.StoreNewGame(ctx, gameState)
		if err != nil {
			t.Fatal("Failed to store game: ", err)
		}
		if !ok {
			t.Fatal("Game already exists")
		}
	}
	// The third game should be ignored, since its TTL is expired.
	games, err := rc.ListRecentGames(ctx, 3)
	if err != nil {
		t.Fatal("Failed to list recent games: ", err)
	}
	if len(games) != 2 {
		t.Errorf("Want %d games, got %d", 2, len(games))
	}
	// Want them in descending order of .Started time:
	if games[0].Started.AsTime() != started[0].AsTime() {
		t.Errorf("Want game 0 to be %v, got %v", started[0], games[0].Started)
	}
	if games[1].Started.AsTime() != started[1].AsTime() {
		t.Errorf("Want game 0 to be %v, got %v", started[1], games[1].Started)
	}
}
