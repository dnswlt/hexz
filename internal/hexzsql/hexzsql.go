package hexzsql

import (
	"context"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

type GameRow struct {
	Created       time.Time
	GameID        string
	Started       time.Time
	GameType      api.GameType
	CPUPlayerMode pb.CPUPlayerMode_Enum
	Host          string
	HostID        string
	Move          int
	ScoreP1       int
	ScoreP2       int
	Done          bool
}

type DatabaseStore interface {
	// Stores a game state in the database.
	StoreGame(ctx context.Context, hostId string, state *pb.GameState) error
	// Adds an entry to the game history.
	// If boardStatus is not nil, the game table will get updated with the status information.
	InsertHistory(ctx context.Context, entryType string, gameId string, state *pb.GameState, boardStatus *api.BoardStatus) error
	// Adds stats for a CPU move.
	InsertStats(ctx context.Context, stats *api.WASMStatsRequest) error
	// Loads the latest game state.
	LoadGame(ctx context.Context, gameId string) (*pb.GameState, error)
	// Lists the `limit` most recent games, skipping `offset` many (for paging).
	// Games without a single move are not included in the result.
	ListRecentGames(ctx context.Context, offset int, limit int) ([]*GameRow, error)
}
