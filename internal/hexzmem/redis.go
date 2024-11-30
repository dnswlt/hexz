package hexzmem

// Contains interfaces and implementations for storing game data remotely.

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/internal/hlog"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"github.com/redis/go-redis/v9"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

type RedisClient struct {
	client *redis.Client
	config *RedisClientConfig
}

type RedisClientConfig struct {
	Addr     string
	LoginTTL time.Duration
	GameTTL  time.Duration
	DB       int // Production should always use 0, 1 is for testing.
}

// RemotePlayerStore is an interface adapter that lets RedisClient implement PlayerStore.
type RemotePlayerStore struct {
	*RedisClient
}

func (s *RemotePlayerStore) Lookup(ctx context.Context, playerId api.PlayerId) (api.Player, error) {
	return s.LookupPlayer(ctx, playerId)
}

func (s *RemotePlayerStore) Login(ctx context.Context, playerId api.PlayerId, name string) error {
	return s.LoginPlayer(ctx, playerId, name)
}

func (s *RemotePlayerStore) Logout(ctx context.Context, playerId api.PlayerId) error {
	return s.LogoutPlayer(ctx, playerId)
}

func NewRedisClient(config *RedisClientConfig) (*RedisClient, error) {
	rc := &RedisClient{
		config: config,
		client: redis.NewClient(&redis.Options{
			Addr: config.Addr,
			DB:   config.DB,
		}),
	}
	if err := rc.Ping(); err != nil {
		return nil, err
	}
	return rc, nil
}

func (c *RedisClient) Ping() error {
	return c.client.Ping(context.Background()).Err()
}

// escape replaces all occurrences of % and / by their %<hex> escape values.
// Used to ensure path segments in a Redis key can be cleanly separated by "/".
func escape(keySegment string) string {
	keySegment = strings.ReplaceAll(keySegment, "%", "%25")
	return strings.ReplaceAll(keySegment, "/", "%2F")
}

// rjoin "redis joins" the dir and file parts of a redis key.
func rkey(path, key string) string {
	return strings.TrimRight(path, "/") + "/" + escape(key)
}

func (c *RedisClient) LookupPlayer(ctx context.Context, playerId api.PlayerId) (api.Player, error) {
	val, err := c.client.GetEx(ctx, rkey("/login", string(playerId)), c.config.LoginTTL).Result()
	if err != nil {
		return api.Player{}, err
	}
	return api.Player{
		Id:         playerId,
		Name:       val,
		LastActive: time.Now(),
	}, nil
}

func (c *RedisClient) LoginPlayer(ctx context.Context, playerId api.PlayerId, name string) error {
	return c.client.SetEx(ctx, rkey("/login", string(playerId)), name, c.config.LoginTTL).Err()
}

func (c *RedisClient) LogoutPlayer(ctx context.Context, playerId api.PlayerId) error {
	return c.client.Del(ctx, rkey("/login", string(playerId))).Err()
}

// Stores the given game state in Redis.
// Game ID and pubsub ID are taken from the game state's GameInfo.
// If any of them is not set, it is assigned an appropriate random value.
// This method also updates the state's Modified field.
func (c *RedisClient) StoreNewGame(ctx context.Context, state *pb.GameState) error {
	if state.GameInfo == nil {
		return fmt.Errorf("cannot store game state without GameInfo")
	}

	gameId := state.GameInfo.Id
	if gameId == "" {
		// Assign random game ID
		for {
			// Grab an unused game ID. Store only a "<reserved>" dummy value.
			gameId = GenerateGameID()
			set, err := c.client.SetNX(ctx, rkey("/game", gameId), "<reserved>", 5*time.Second).Result()
			if err != nil {
				return err
			}
			if set {
				break
			}
		}
		state.GameInfo.Id = gameId
	}
	state.Modified = tpb.Now()
	// Store pubsub ID mapping
	if state.GameInfo.PubsubChannelId == "" {
		// 16 bytes of random data should be enough to assume no collisions.
		state.GameInfo.PubsubChannelId = GeneratePubsubID()
	}
	// Store game
	data, err := proto.Marshal(state)
	if err != nil {
		return fmt.Errorf("failed to marshal GameState: %v", err)
	}
	err = c.client.Set(ctx, rkey("/game", gameId), data, c.config.GameTTL).Err()
	if err != nil {
		return fmt.Errorf("failed to store game in Redis: %v", err)
	}
	// Store in recentgames set.
	mInfo, _ := proto.Marshal(state.GameInfo) // We can always marshal a GameInfo.
	if err := c.client.ZAdd(ctx, "/recentgames", redis.Z{
		Score:  float64(state.GetGameInfo().GetStarted().GetSeconds()),
		Member: mInfo,
	}).Err(); err != nil {
		return fmt.Errorf("failed to add game %q to recent games: %v", gameId, err)
	}
	return nil
}

// Stores the given game state in Redis, overwriting any existing game with the same ID.
// This method updates the Seqnum and Modified fields of the game state.
func (c *RedisClient) UpdateGame(ctx context.Context, s *pb.GameState) error {
	if s.GetGameInfo().Id == "" {
		return fmt.Errorf("game state must have an ID for UpdateGame")
	}
	s.Modified = tpb.Now()
	data, err := proto.Marshal(s)
	if err != nil {
		return err
	}
	gameId := s.GameInfo.Id
	return c.client.Set(ctx, rkey("/game", gameId), data, c.config.GameTTL).Err()
}

func (c *RedisClient) LookupGame(ctx context.Context, gameId string) (*pb.GameState, error) {
	data, err := c.client.Get(ctx, rkey("/game", gameId)).Result()
	if err != nil {
		return nil, err
	}
	gameState := &pb.GameState{}
	if err := proto.Unmarshal([]byte(data), gameState); err != nil {
		return nil, err
	}
	return gameState, nil
}

func (c *RedisClient) DeleteGame(ctx context.Context, gameId string) error {
	if err := c.client.Del(ctx, rkey("/game", gameId)).Err(); err != nil {
		return err
	}
	if err := c.client.ZRem(ctx, "/recentgames", gameId).Err(); err != nil {
		hlog.Errorf("Failed to remove game %q from recentgames: %v", gameId, err)
	}
	return nil
}

func (c *RedisClient) ListRecentGames(ctx context.Context, limit int) ([]*pb.GameInfo, error) {
	r, err := c.client.ZRevRange(ctx, "/recentgames", 0, int64(limit-1)).Result()
	if err != nil {
		return nil, err
	}
	games := make([]*pb.GameInfo, 0, len(r))
	for _, m := range r {
		gi := &pb.GameInfo{}
		if err := proto.Unmarshal([]byte(m), gi); err != nil {
			return nil, err
		}
		s := time.Since(gi.GetStarted().AsTime())
		if s > c.config.GameTTL {
			continue
		}
		games = append(games, gi)
	}
	// Clean up recentgames if it gets too big. Use hard-coded numbers for now.
	// TODO: make this configurable.
	card, err := c.client.ZCard(ctx, "/recentgames").Result()
	if err != nil {
		hlog.Errorf("Failed to query ZCARD for /recentgames: %v", err)
		return games, err
	}
	minItems := 20
	if minItems < limit*2 {
		minItems = limit * 2 // Keep enough to list recent games, and have some buffer.
	}
	maxItems := 2 * minItems // Avoid removing single items at each call.
	if card > int64(maxItems) {
		if n, err := c.client.ZRemRangeByRank(ctx, "/recentgames", 0, card-int64(minItems)-1).Result(); err != nil {
			hlog.Errorf("Failed to remove old games from /recentgames: %v", err)
		} else {
			hlog.Infof("Removed %d old games from /recentgames", n)
		}
	}
	return games, nil
}

func (c *RedisClient) Subscribe(ctx context.Context, pubsubId string) <-chan *pb.GameStorePubsubEvent {
	ch := make(chan *pb.GameStorePubsubEvent)
	go func() {
		defer close(ch)
		sub := c.client.Subscribe(ctx, rkey("/pubsub", pubsubId))
		defer sub.Close()
		for {
			select {
			case msg, ok := <-sub.Channel():
				if !ok {
					return
				}
				event := &pb.GameStorePubsubEvent{}
				if err := proto.Unmarshal([]byte(msg.Payload), event); err != nil {
					hlog.Errorf("Received invalid event from Redis: %v", err)
					return
				}
				ch <- event
			case <-ctx.Done():
				// sub.Channel() does not seem to respond to context cancellation, so we do it externally.
				return
			}
		}
	}()
	return ch
}

// Sends a message to the channel for the given game.
// Returns the number of subscribers that received the message.
func (c *RedisClient) Publish(ctx context.Context, pubsubId string, event *pb.GameStorePubsubEvent) error {
	data, err := proto.Marshal(event)
	if err != nil {
		hlog.Fatalf("Cannot marshal event: %v", err)
	}
	return c.client.Publish(ctx, rkey("/pubsub", pubsubId), data).Err()
}
