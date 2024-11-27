package hexzsql

import (
	"context"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

type DatabaseStore interface {
	// Stores a game state in the database.
	StoreGame(ctx context.Context, hostId string, state *pb.GameState) error
	// Adds an entry to the game history.
	// state can be nil for "undo" and "redo" entries.
	InsertHistory(ctx context.Context, entryType string, gameId string, state *pb.GameState) error
	// Adds stats for a CPU move.
	InsertStats(ctx context.Context, stats *api.WASMStatsRequest) error
	// Loads the latest game state.
	LoadGame(ctx context.Context, gameId string) (*pb.GameState, error)
	// Lists the `limit` most recent games, skipping `offset` many (for paging).
	ListRecentGames(ctx context.Context, offset int, limit int) ([]*pb.GameInfo, error)
}
