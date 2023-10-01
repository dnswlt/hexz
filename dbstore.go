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
	StoreGame(ctx context.Context, gameId string, state *pb.GameState) error
	// Adds stats for a CPU move.
	InsertStats(ctx context.Context, stats *WASMStatsRequest) error
	// Loads a game state from the database.
	LoadGame(ctx context.Context, gameId string) (*pb.GameState, error)
}
