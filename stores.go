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
	// Adds an entry to the game history. state can be nil for "undo" and "redo" entries.
	InsertHistory(ctx context.Context, entryType string, gameId string, state *pb.GameState) error
	// Returns the previous game state from the database. Does not write any new history entries.
	// Clients should register the undo once it became effective by calling InsertHistory(ctx, "undo", ...).
	PreviousGameState(ctx context.Context, gameId string) (*pb.GameState, error)
	// Returns the next game state from the database. Does not write any new history entries.
	// This will only yield a non-error result if the previous event was an undo.
	// Clients should register the undo once it became effective by calling InsertHistory(ctx, "undo", ...).
	NextGameState(ctx context.Context, gameId string) (*pb.GameState, error)
	// Adds stats for a CPU move.
	InsertStats(ctx context.Context, stats *WASMStatsRequest) error
	// Loads a game state from the database.
	LoadGame(ctx context.Context, gameId string) (*pb.GameState, error)
}

// GameStore is an interface for local or remote game stores, e.g. Redis.
type GameStore interface {
	StoreNewGame(ctx context.Context, s *pb.GameState) (bool, error)
	LookupGame(ctx context.Context, gameId string) (*pb.GameState, error)
	UpdateGame(ctx context.Context, s *pb.GameState) error
	ListRecentGames(ctx context.Context, limit int) ([]*pb.GameInfo, error)

	Publish(ctx context.Context, gameId string, event string) error
	Subscribe(ctx context.Context, gameId string, ch chan<- string)
}

type PlayerStore interface {
	// Lookup looks up the given player by ID.
	Lookup(ctx context.Context, playerId PlayerId) (Player, error)
	// Login logs in the given player. If the player is already logged in,
	// the existing data will be overwritten with the new data.
	Login(ctx context.Context, playerId PlayerId, name string) error
}
