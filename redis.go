package hexz

import (
	"context"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
	"github.com/redis/go-redis/v9"
	"google.golang.org/protobuf/proto"
)

type RedisClient struct {
	client *redis.Client
}

func NewRedisClient(addr string) (*RedisClient, error) {
	rc := &RedisClient{
		redis.NewClient(&redis.Options{
			Addr: addr,
		}),
	}
	if err := rc.Ping(); err != nil {
		return nil, err
	}
	infoLog.Printf("Connected to Redis at %s", rc.client.Options().Addr)
	return rc, nil
}

func (c *RedisClient) Ping() error {
	return c.client.Ping(context.Background()).Err()
}

func (c *RedisClient) LookupPlayer(ctx context.Context, playerId PlayerId, loginTTL time.Duration) (Player, error) {
	val, err := c.client.GetEx(ctx, "login:"+string(playerId), loginTTL).Result()
	if err != nil {
		if err != redis.Nil {
			errorLog.Printf("Failed to look up player %q: %v", playerId, err)
		}
		return Player{}, err
	}
	return Player{
		Id:         playerId,
		Name:       val,
		LastActive: time.Now(),
	}, nil
}

func (c *RedisClient) LoginPlayer(ctx context.Context, playerId PlayerId, name string, loginTTL time.Duration) error {
	return c.client.SetEx(ctx, "login:"+string(playerId), name, loginTTL).Err()
}

func (c *RedisClient) StoreNewGame(ctx context.Context, gameId string, gameState *pb.GameState) (bool, error) {
	data, err := proto.Marshal(gameState)
	if err != nil {
		return false, err
	}
	// Store the game in Redis for 24 hours max.
	return c.client.SetNX(ctx, "game:"+gameId, data, 24*time.Hour).Result()
}

func (c *RedisClient) LookupGame(ctx context.Context, gameId string) (*pb.GameState, error) {
	data, err := c.client.Get(ctx, "game:"+gameId).Result()
	if err != nil {
		return nil, err
	}
	gameState := &pb.GameState{}
	if err := proto.Unmarshal([]byte(data), gameState); err != nil {
		return nil, err
	}
	return gameState, nil
}

func (c *RedisClient) SubscribeSSE(ctx context.Context, gameId string, ch chan<- string) {
	sub := c.client.Subscribe(ctx, "sse:"+gameId)
	defer sub.Close()
	defer close(ch)
	for msg := range sub.Channel() {
		ch <- msg.Payload
	}
}

// Sends a "notify" message to the SSE channel for the given game.
// Returns the number of subscribers that received the message.
func (c *RedisClient) NotifySSE(ctx context.Context, gameId string) (int64, error) {
	return c.client.Publish(ctx, "sse:"+gameId, "notify").Result()
}
