package hexzmem

import (
	"context"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

// Player contains the data that is stored in the PlayerStore for a given player.
// It has JSON annotations for serialization (to disk or in memory storage).
type Player struct {
	// A randomly generated unique ID.
	// This ID may be used in cookies.
	Id         api.PlayerId `json:"id"`
	Name       string       `json:"name"`
	LastActive time.Time    `json:"lastActive"`
	// Only set for registered users. For guests, this field is empty.
	UserID string `json:"userId"`
}

type PlayerStore interface {
	// Lookup looks up the given player by ID.
	Lookup(ctx context.Context, playerId api.PlayerId) (Player, error)
	// Login logs in the given player. If the player is already logged in,
	// the existing data will be overwritten with the new data.
	// LastActive will be updated internally and does not need to be set by the caller.
	Login(ctx context.Context, player Player) error
	// Deletes the player from the store.
	Logout(ctx context.Context, playerId api.PlayerId) error
}

// GameInfo contains essential summary info about a game.
// It is stored in serialized form for open and active games.
type GameInfo struct {
	Id       string
	Host     string
	Started  time.Time
	GameType api.GameType
}

// GameStore is an interface for local or remote game stores, e.g. Redis.
type GameStore interface {
	// StoreNewGame stores a new game under a randomly chosen new game ID.
	// The game ID will be returned, and also stored in state.GameInfo.Id.
	// If that field is already set in the input, an error is returned
	// and the game is not stored.
	StoreNewGame(ctx context.Context, state *pb.GameState) error
	// LookupGame looks up the current game state for the given gameId.
	LookupGame(ctx context.Context, gameId string) (*pb.GameState, error)
	// UpdateGame updates the game state for game ID s.GameInfo.Id.
	// Existing states are overwritten unconditionally.
	UpdateGame(ctx context.Context, state *pb.GameState) error
	// ListOpenGames lists games that are waiting for more players to join.
	ListOpenGames(ctx context.Context, limit int) ([]*GameInfo, error)
	// ListActiveGames lists the limit most recently played games.
	ListActiveGames(ctx context.Context, limit int) ([]*GameInfo, error)

	// Publish publishes the given game event on the GameStore's pubsub topic.
	Publish(ctx context.Context, pubsubId string, event *pb.GameStorePubsubEvent) error
	// Subscribe subscribes to game events for the given game ID. Events will be
	// sent to the returned channel. This method internaly spawns a goroutine
	// and returns immediately. The goroutine will close the returned channel
	// and terminate when the provided context ctx is cancelled.
	Subscribe(ctx context.Context, pubsubId string) <-chan *pb.GameStorePubsubEvent
}

// TokenStore is the interface for a store of tokens.
// Tokens can be used for CSRF prevention and rate limiting.
type TokenStore interface {
	// NewCSRFToken creates and returns a new token with the specified TTL.
	NewCSRFToken(ctx context.Context, playerId api.PlayerId, ttl time.Duration) (string, error)
	// ConsumeCSRFToken checks if the given CSRF token is known, and consumes it.
	ConsumeCSRFToken(ctx context.Context, playerId api.PlayerId, token string) (bool, error)

	// TokenBucketGet atomically gets the requested number of tokens from bucket.
	// Before removing any tokens, it refills the bucket according to the
	// given refillRate (in tokens per second) and the most recent refill time.
	// On the first call (ever or after the max TTL for any bucket), the bucket
	// is assumed to contain capacity tokens. The bucket will never contain
	// more than capacity tokens.
	TokenBucketGet(ctx context.Context, bucket string, tokens int, refillRate float64, capacity int) (bool, error)
}

type FlashMessage struct {
	Kind    string    `json:"kind"`
	Message string    `json:"message"`
	Created time.Time `json:"created"`
}

type FlashStore interface {
	// AddMessage adds a flash message for the given flashID.
	// Given the short-lived nature of flash messages, implementations should
	// ensure that messages associated with a flashID have a short TTL
	// (1 minute should be enough).
	AddMessage(ctx context.Context, flashID string, msg FlashMessage) error
	// PopMessages retrieves all flash messages for the flashID and removes them from the store.
	PopMessages(ctx context.Context, flashID string) ([]FlashMessage, error)
}
