package hexz

import (
	"context"
	"flag"
	"testing"
	"time"
)

var (
	testRedisAddr = flag.String("test-redis-addr", "localhost:6379", "Address of Redis server used for integration tests")
)

func TestRedisPubsub(t *testing.T) {
	if *testRedisAddr == "" {
		t.Skip("Skipping integration test because -test-redis-addr is not set")
	}
	rc, err := NewRedisClient(&RedisClientConfig{
		Addr:     *testRedisAddr,
		LoginTTL: 24 * time.Hour,
		GameTTL:  24 * time.Hour,
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
