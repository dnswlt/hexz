package hexz

// Contains interfaces and implementations for storing game data remotely.

import (
	"context"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
	"github.com/redis/go-redis/v9"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

type GameStore interface {
	StoreNewGame(ctx context.Context, s *pb.GameState) (bool, error)
	LookupGame(ctx context.Context, gameId string) (*pb.GameState, error)
	UpdateGame(ctx context.Context, s *pb.GameState) error
	ListRecentGames(ctx context.Context, limit int) ([]*pb.GameInfo, error)

	Publish(ctx context.Context, gameId string, event string) error
	Subscribe(ctx context.Context, gameId string, ch chan<- string)
}

type RedisClient struct {
	client *redis.Client
	config *RedisClientConfig
}

type RedisClientConfig struct {
	Addr     string
	LoginTTL time.Duration
	GameTTL  time.Duration
}

func NewRedisClient(config *RedisClientConfig) (*RedisClient, error) {
	rc := &RedisClient{
		config: config,
		client: redis.NewClient(&redis.Options{
			Addr: config.Addr,
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

func (c *RedisClient) LookupPlayer(ctx context.Context, playerId PlayerId) (Player, error) {
	val, err := c.client.GetEx(ctx, "login:"+string(playerId), c.config.LoginTTL).Result()
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

func (c *RedisClient) LoginPlayer(ctx context.Context, playerId PlayerId, name string) error {
	return c.client.SetEx(ctx, "login:"+string(playerId), name, c.config.LoginTTL).Err()
}

// Stores the given game state in Redis, unless a game with the same ID already exists.
// This method updates the Modified fields of the game state.
func (c *RedisClient) StoreNewGame(ctx context.Context, s *pb.GameState) (bool, error) {
	now := tpb.Now()
	s.Modified = now
	data, err := proto.Marshal(s)
	if err != nil {
		return false, err
	}
	gameId := s.GameInfo.Id
	infoLog.Printf("Storing new game %q: %d bytes", gameId, len(data))
	ok, err := c.client.SetNX(ctx, "game:"+gameId, data, c.config.GameTTL).Result()
	if !ok || err != nil {
		return ok, err
	}
	mInfo, _ := proto.Marshal(s.GameInfo) // We can always marshal a GameInfo.
	// TODO: Clean up "recentgames" periodically.
	if err := c.client.ZAdd(ctx, "recentgames", redis.Z{Score: float64(now.Seconds), Member: mInfo}).Err(); err != nil {
		errorLog.Printf("Failed to add game %q to recent games: %v", gameId, err)
	}
	return true, nil
}

// Stores the given game state in Redis, overwriting any existing game with the same ID.
// This method updates the Seqnum and Modified fields of the game state.
func (c *RedisClient) UpdateGame(ctx context.Context, s *pb.GameState) error {
	s.Seqnum++
	s.Modified = tpb.Now()
	data, err := proto.Marshal(s)
	if err != nil {
		return err
	}
	return c.client.Set(ctx, "game:"+s.GameInfo.Id, data, c.config.GameTTL).Err()
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

func (c *RedisClient) ListRecentGames(ctx context.Context, limit int) ([]*pb.GameInfo, error) {
	r, err := c.client.ZRevRange(ctx, "recentgames", 0, int64(limit-1)).Result()
	if err != nil {
		return nil, err
	}
	games := make([]*pb.GameInfo, len(r))
	for i, m := range r {
		games[i] = &pb.GameInfo{}
		if err := proto.Unmarshal([]byte(m), games[i]); err != nil {
			return nil, err
		}
	}
	return games, nil
}

func (c *RedisClient) Subscribe(ctx context.Context, gameId string, ch chan<- string) {
	sub := c.client.Subscribe(ctx, "pubsub:"+gameId)
	defer sub.Close()
	defer close(ch)
	for {
		select {
		case msg, ok := <-sub.Channel():
			if !ok {
				return
			}
			ch <- msg.Payload
		case <-ctx.Done():
			// sub.Channel() does not seem to respond to context cancellation, so we do it externally.
			return
		}
	}
}

// Sends a message to the channel for the given game.
// Returns the number of subscribers that received the message.
func (c *RedisClient) Publish(ctx context.Context, gameId string, message string) error {
	return c.client.Publish(ctx, "pubsub:"+gameId, message).Err()
}
