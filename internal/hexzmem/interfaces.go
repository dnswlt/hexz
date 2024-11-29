package hexzmem

import (
	"context"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

type PlayerStore interface {
	// Lookup looks up the given player by ID.
	Lookup(ctx context.Context, playerId api.PlayerId) (api.Player, error)
	// Login logs in the given player. If the player is already logged in,
	// the existing data will be overwritten with the new data.
	Login(ctx context.Context, playerId api.PlayerId, name string) error
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
	// ListRecentGames lists the limit most recently played games.
	ListRecentGames(ctx context.Context, limit int) ([]*pb.GameInfo, error)

	// Publish publishes the given game event on the GameStore's pubsub topic.
	Publish(ctx context.Context, pubsubId string, event *pb.GameStorePubsubEvent) error
	// Subscribe subscribes to game events for the given game ID. Events will be
	// sent to the returned channel. This method internaly spawns a goroutine
	// and returns immediately. The goroutine will close the returned channel
	// and terminate when the provided context ctx is cancelled.
	Subscribe(ctx context.Context, pubsubId string) <-chan *pb.GameStorePubsubEvent
}
