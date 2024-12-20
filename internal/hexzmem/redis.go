package hexzmem

// Contains interfaces and implementations for storing game data remotely.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/internal/hlog"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"google.golang.org/protobuf/proto"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

type Clock interface {
	Now() time.Time
}

type RealClock struct{}

func (c *RealClock) Now() time.Time { return time.Now() }

type RedisClient struct {
	client *redis.Client
	config *RedisClientConfig
	clock  Clock
	// Last time a cleanup of the open & active game sets was run.
	// Used to periodically remove old games.
	lastCleanup time.Time
	mut         sync.Mutex
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

func (s *RemotePlayerStore) Lookup(ctx context.Context, playerId api.PlayerId) (Player, error) {
	return s.LookupPlayer(ctx, playerId)
}

func (s *RemotePlayerStore) Login(ctx context.Context, player Player) error {
	return s.LoginPlayer(ctx, player)
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
		clock: &RealClock{},
	}
	if err := rc.Ping(); err != nil {
		return nil, err
	}
	return rc, nil
}

func (c *RedisClient) Ping() error {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	return c.client.Ping(ctx).Err()
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

func (c *RedisClient) LookupPlayer(ctx context.Context, playerId api.PlayerId) (Player, error) {
	data, err := c.client.GetEx(ctx, rkey("/login", string(playerId)), c.config.LoginTTL).Bytes()
	if err != nil {
		return Player{}, err
	}
	var player Player
	err = json.Unmarshal(data, &player)
	if err != nil {
		return Player{}, fmt.Errorf("failed to unmarshal player: %v", err)
	}
	return player, nil
}

func (c *RedisClient) LoginPlayer(ctx context.Context, player Player) error {
	data, err := json.Marshal(player)
	if err != nil {
		return fmt.Errorf("cannot marshal player: %v", err)
	}
	return c.client.SetEx(ctx, rkey("/login", string(player.Id)), data, c.config.LoginTTL).Err()
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
	state.Modified = tpb.New(c.clock.Now())
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
	if state.AllPlayersJoined {
		// All players have joined (probably single player game).
		return c.storeActiveGame(ctx, state)
	}
	// Waiting for more players.
	return c.storeOpenGame(ctx, state)
}

func (c *RedisClient) storeOpenGame(ctx context.Context, s *pb.GameState) error {
	err := c.client.ZAdd(ctx, "/opengames", redis.Z{
		Score:  float64(s.Modified.Seconds),
		Member: s.GameInfo.Id,
	}).Err()
	if err != nil {
		return fmt.Errorf("failed to add game %q to open games: %v", s.GameInfo.Id, err)
	}
	return nil
}

func (c *RedisClient) storeActiveGame(ctx context.Context, s *pb.GameState) error {
	err := c.client.ZAdd(ctx, "/activegames", redis.Z{
		Score:  float64(s.Modified.Seconds),
		Member: s.GameInfo.Id,
	}).Err()
	if err != nil {
		return fmt.Errorf("failed to add game %q to active games: %v", s.GameInfo.Id, err)
	}
	return nil
}

func (c *RedisClient) refreshGameTTLs(ctx context.Context, s *pb.GameState) error {
	var err error
	if s.AllPlayersJoined {
		// Active game. Always ZADD, it might not yet exist.
		err = c.client.ZAdd(ctx, "/activegames", redis.Z{
			Score:  float64(s.Modified.Seconds),
			Member: s.GameInfo.Id,
		}).Err()
		if err != nil {
			return fmt.Errorf("failed to update TTL in /activegames for %s: %v", s.GameInfo.Id, err)
		}
		return nil
	}
	// Open game. Only ZADD XX (if exists), a new game is always stored on creation.
	err = c.client.ZAddXX(ctx, "/opengames", redis.Z{
		Score:  float64(s.Modified.Seconds),
		Member: s.GameInfo.Id,
	}).Err()
	if err != nil {
		return fmt.Errorf("failed to update TTL in /opengames for %s: %v", s.GameInfo.Id, err)
	}
	return nil
}

// Stores the given game state in Redis, overwriting any existing game with the same ID.
// This method updates the Modified field of the game state.
func (c *RedisClient) UpdateGame(ctx context.Context, s *pb.GameState) error {
	if s.GetGameInfo().Id == "" {
		return fmt.Errorf("game state must have an ID for UpdateGame")
	}
	s.Modified = tpb.New(c.clock.Now())
	data, err := proto.Marshal(s)
	if err != nil {
		return err
	}
	gameId := s.GameInfo.Id
	if err := c.client.Set(ctx, rkey("/game", gameId), data, c.config.GameTTL).Err(); err != nil {
		return fmt.Errorf("failed to update game %s: %v", gameId, err)
	}
	return c.refreshGameTTLs(ctx, s)
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
	if err := c.client.ZRem(ctx, "/opengames", gameId).Err(); err != nil {
		return fmt.Errorf("failed to remove game %q from /opengames: %v", gameId, err)
	}
	if err := c.client.ZRem(ctx, "/activegames", gameId).Err(); err != nil {
		return fmt.Errorf("failed to remove game %q from /activegames: %v", gameId, err)
	}
	return nil
}

// Cleans up games that are older than the gameTTL from the open & active game sets.
// This method is safe to be called frequently, it throttles internally to 1 cleanup / minute.
func (c *RedisClient) cleanupGameSets(ctx context.Context, force bool) error {
	c.mut.Lock()
	defer c.mut.Unlock()
	if !force && c.clock.Now().Sub(c.lastCleanup) < 1*time.Minute {
		// Clean up at most once per minute.
		return nil
	}
	now := c.clock.Now()
	c.lastCleanup = now
	err := c.client.ZRemRangeByScore(ctx, "/opengames", "-inf", strconv.FormatInt(now.Add(-c.config.GameTTL).Unix(), 10)).Err()
	if err != nil {
		return fmt.Errorf("failed to remove old games from /opengames: %v", err)
	}
	err = c.client.ZRemRangeByScore(ctx, "/activegames", "-inf", strconv.FormatInt(now.Add(-c.config.GameTTL).Unix(), 10)).Err()
	if err != nil {
		return fmt.Errorf("failed to remove old games from /activegames: %v", err)
	}
	return nil
}

func (c *RedisClient) ListOpenGames(ctx context.Context, limit int) ([]*GameInfo, error) {
	// Fetch 2x as many games as requested to increase chances of having enough in the face of deletions.
	rangeLimit := 2 * limit
	gameIds, err := c.client.ZRevRange(ctx, "/opengames", 0, int64(rangeLimit)).Result()
	if err != nil {
		return nil, err
	}
	games := make([]*GameInfo, 0, len(gameIds))
	foundOldEntry := false
	var removeIds []any
	// Iterate over retrieved game IDs, filtering out expired and deleted ones.
	// These are cleaned up below.
	for _, gameId := range gameIds {
		gameState, err := c.LookupGame(ctx, gameId)
		if err != nil {
			if errors.Is(err, redis.Nil) {
				// Game no longer exists: purge
				removeIds = append(removeIds, gameId)
				continue
			}
			return nil, fmt.Errorf("failed to lookup game ID %s: %v", gameId, err)
		}
		if gameState.AllPlayersJoined {
			// Game no longer accepts players: purge
			removeIds = append(removeIds, gameId)
			continue
		}
		if c.clock.Now().Sub(gameState.Modified.AsTime()) >= c.config.GameTTL {
			foundOldEntry = true
			break
		}
		games = append(games, &GameInfo{
			Id:       gameState.GameInfo.Id,
			Host:     gameState.GameInfo.Host,
			Started:  gameState.GameInfo.Started.AsTime(),
			GameType: api.GameType(gameState.GameInfo.Type),
		})
		if len(games) == limit {
			break // collected the requested number of games
		}
	}
	if len(removeIds) > 0 {
		if err := c.client.ZRem(ctx, "/opengames", removeIds...).Err(); err != nil {
			return nil, fmt.Errorf("failed to delete games %v from /opengames: %v", removeIds, err)
		}
		if len(games) < limit && !foundOldEntry && len(gameIds) == rangeLimit {
			// We found stale entries and there might be more valid entries: retry
			return c.ListOpenGames(ctx, limit)
		}
	}
	if err := c.cleanupGameSets(ctx, foundOldEntry); err != nil {
		return nil, err
	}
	return games, nil
}

func (c *RedisClient) ListActiveGames(ctx context.Context, limit int) ([]*GameInfo, error) {
	// Fetch 2x as many games as requested to increase chances of having enough in the face of deletions.
	rangeLimit := 2 * limit
	gameIds, err := c.client.ZRevRange(ctx, "/activegames", 0, int64(rangeLimit)).Result()
	if err != nil {
		return nil, err
	}
	games := make([]*GameInfo, 0, limit)
	foundOldEntry := false
	var removeIds []any
	for _, gameId := range gameIds {
		gameState, err := c.LookupGame(ctx, gameId)
		if err != nil {
			if errors.Is(err, redis.Nil) {
				// Game no longer exists: purge
				removeIds = append(removeIds, gameId)
				continue
			}
			return nil, fmt.Errorf("failed to lookup game ID %s: %v", gameId, err)
		}
		if c.clock.Now().Sub(gameState.Modified.AsTime()) >= c.config.GameTTL {
			foundOldEntry = true
			break
		}
		games = append(games, &GameInfo{
			Id:       gameState.GameInfo.Id,
			Host:     gameState.GameInfo.Host,
			Started:  gameState.GameInfo.Started.AsTime(),
			GameType: api.GameType(gameState.GameInfo.Type),
		})
		if len(games) == limit {
			break // collected the requested number of games
		}
	}
	if len(removeIds) > 0 {
		if err := c.client.ZRem(ctx, "/activegames", removeIds...).Err(); err != nil {
			return nil, fmt.Errorf("failed to delete games %v from /activegames: %v", removeIds, err)
		}
		if len(games) < limit && !foundOldEntry && len(gameIds) == rangeLimit {
			// We found stale entries and there might be more valid entries: retry
			return c.ListActiveGames(ctx, limit)
		}
	}
	if err := c.cleanupGameSets(ctx, foundOldEntry); err != nil {
		return nil, err
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

func (c *RedisClient) NewCSRFToken(ctx context.Context, ttl time.Duration) (string, error) {
	uid := uuid.New().String()
	err := c.client.Set(ctx, rkey("/csrf", uid), 1, ttl).Err()
	if err != nil {
		return "", fmt.Errorf("could not store token: %v", err)
	}
	return uid, nil
}

func (c *RedisClient) ConsumeCSRFToken(ctx context.Context, token string) (bool, error) {
	n, err := c.client.Del(ctx, rkey("/csrf", token)).Result()
	if err != nil {
		return false, fmt.Errorf("could not consume token: %v", err)
	}
	return n == 1, nil
}

func (c *RedisClient) TokenBucketGet(ctx context.Context, bucket string, tokens int, refillRate float64, capacity int) (bool, error) {
	if tokens <= 0 {
		return false, fmt.Errorf("tokens must be positive")
	}
	if refillRate <= 0 {
		return false, fmt.Errorf("refillRate must be positive")
	}
	if capacity <= 0 {
		return false, fmt.Errorf("capacity must be positive")
	}
	now := time.Now().Unix()
	// WTF, right?! Go, Lua, Redis.
	luaScript := `
		local key = KEYS[1]
		local tokens = tonumber(ARGV[1])
		local refillRate = tonumber(ARGV[2])
		local capacity = tonumber(ARGV[3])
		local now = tonumber(ARGV[4])

		-- Get current bucket state
		local bucket = redis.call("HMGET", key, "remaining_tokens", "last_refill_time")
		local remainingTokens = tonumber(bucket[1]) or capacity
		local lastRefillTime = tonumber(bucket[2]) or now

		-- Calculate tokens to refill
		local timeElapsed = now - lastRefillTime
		local tokensToAdd = math.floor(timeElapsed * refillRate)
		remainingTokens = math.min(capacity, remainingTokens + tokensToAdd)

		-- Update last refill time
		lastRefillTime = now

		-- TTL for the bucket: seconds it takes to fill entirely
		local ttl = math.ceil(capacity / refillRate)

		-- Check if request can be allowed
		if remainingTokens >= tokens then
			remainingTokens = remainingTokens - tokens
			redis.call("HMSET", key, "remaining_tokens", remainingTokens, "last_refill_time", lastRefillTime, "ttl", ttl)
			redis.call("EXPIRE", key, math.max(60, ttl))
			return 1 -- Allow request
		else
			-- Deny request
			redis.call("HMSET", key, "remaining_tokens", remainingTokens, "last_refill_time", lastRefillTime, "ttl", ttl)
			redis.call("EXPIRE", key, math.max(60, ttl))
			return 0 -- Deny request
		end
	`

	// Execute Lua script to atomically check and update the bucket.
	result, err := c.client.Eval(ctx, luaScript, []string{rkey("/tb", bucket)},
		tokens, refillRate, capacity, now).Result()
	if err != nil {
		return false, err
	}
	return result == int64(1), nil
}

func (c *RedisClient) AddMessage(ctx context.Context, flashID string, msg FlashMessage) error {
	data, err := json.Marshal(&msg)
	if err != nil {
		return fmt.Errorf("failed to marshal FlashMessage: %v", err)
	}
	key := rkey("/flash", flashID)
	err = c.client.ZAdd(ctx, key, redis.Z{
		Member: data,
		Score:  float64(msg.Created.Unix()),
	}).Err()
	if err != nil {
		return fmt.Errorf("failed to store FlashMessage: %v", err)
	}
	// Ensure flash messages that are never retrieved get removed quickly.
	c.client.Expire(ctx, key, 1*time.Minute)
	return nil
}

func (c *RedisClient) PopMessages(ctx context.Context, flashID string) ([]FlashMessage, error) {
	key := rkey("/flash", flashID)

	// Fetch all messages.
	msgs, err := c.client.ZRange(ctx, key, 0, -1).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to fetch FlashMessages: %v", err)
	}
	if len(msgs) == 0 {
		return nil, nil
	}
	// Delete all messages.
	if _, err := c.client.Del(ctx, key).Result(); err != nil {
		return nil, fmt.Errorf("failed to delete FlashMessages: %v", err)
	}
	result := make([]FlashMessage, len(msgs))
	for i, m := range msgs {
		if err := json.Unmarshal([]byte(m), &result[i]); err != nil {
			return nil, fmt.Errorf("failed to unmarshal FlashMessage: %v", err)
		}
	}

	return result, nil
}
