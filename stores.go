package hexz

// Database support.
// While a stateless server will store the current game state in a memstore,
// history and stats can be stored in a database.
// Concrete implementations of the DatabaseStore interface are in subpackage hexzsql.

import (
	"context"

	pb "github.com/dnswlt/hexz/hexzpb"
)

type DatabaseStore interface {
	// Stores a game state in the database.
	StoreGame(ctx context.Context, hostId string, state *pb.GameState) error
	// Adds an entry to the game history.
	// state can be nil for "undo" and "redo" entries.
	InsertHistory(ctx context.Context, entryType string, gameId string, state *pb.GameState) error
	// Adds stats for a CPU move.
	InsertStats(ctx context.Context, stats *WASMStatsRequest) error
	// Loads the latest game state.
	LoadGame(ctx context.Context, gameId string) (*pb.GameState, error)
}

// GameStore is an interface for local or remote game stores, e.g. Redis.
type GameStore interface {
	// StoreNewGame stores a new game using the game ID s.GameInfo.Id.
	// Returns false if a game with that ID exists, and does not store anything.
	StoreNewGame(ctx context.Context, state *pb.GameState) (bool, error)
	// LookupGame looks up the current game state for the given gameId.
	LookupGame(ctx context.Context, gameId string) (*pb.GameState, error)
	// UpdateGame updates the game state for game ID s.GameInfo.Id.
	// Existing states are overwritten unconditionally.
	UpdateGame(ctx context.Context, state *pb.GameState) error
	// ListRecentGames lists the limit most recently played games.
	ListRecentGames(ctx context.Context, limit int) ([]*pb.GameInfo, error)
	// Publish publishes the given game event on the GameStore's pubsub topic.
	Publish(ctx context.Context, gameId string, event *pb.GameStorePubsubEvent) error
	// Subscribe subscribes to game events for the given game ID. Events will be
	// sent to the returned channel. This method internaly spawns a goroutine
	// and returns immediately. The goroutine will close the returned channel
	// and terminate when the provided context ctx is cancelled.
	Subscribe(ctx context.Context, gameId string) <-chan *pb.GameStorePubsubEvent
}

type PlayerStore interface {
	// Lookup looks up the given player by ID.
	Lookup(ctx context.Context, playerId PlayerId) (Player, error)
	// Login logs in the given player. If the player is already logged in,
	// the existing data will be overwritten with the new data.
	Login(ctx context.Context, playerId PlayerId, name string) error
}

// Local
