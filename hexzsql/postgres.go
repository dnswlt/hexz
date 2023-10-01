package hexzsql

import (
	"context"
	"database/sql"
	"fmt"

	_ "github.com/jackc/pgx/v5/stdlib" // Needed to register pgx as a database/sql driver.

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
)

type PostgresStore struct {
	pool *sql.DB
}

func NewPostgresStore(ctx context.Context, database_url string) (*PostgresStore, error) {
	pool, err := sql.Open("pgx", database_url)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to PostgresSQL at %s: %w", database_url, err)
	}
	s := &PostgresStore{pool: pool}
	err = s.pool.PingContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to ping PostgresSQL at %s: %w", database_url, err)
	}
	return s, nil
}

func (s *PostgresStore) StoreGame(ctx context.Context, gameId string, gs *pb.GameState) error {
	_, err := s.pool.ExecContext(ctx, `
		INSERT INTO games (game_id, game_type, player1_name, player2_name) 
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (game_id) DO UPDATE SET game_type = $2, player1_name = $3, player2_name = $4`,
		gameId, string(gs.GameInfo.Type), "hans", "franz")
	return err
}

func (s *PostgresStore) InsertStats(ctx context.Context, stats *hexz.WASMStatsRequest) error {
	t := &stats.Stats
	u := &stats.UserInfo
	_, err := s.pool.ExecContext(ctx, `
		INSERT INTO wasm_stats (
			game_id,
			game_type,
			move_num,
			tree_size,
			max_depth,
			iterations,
			elapsed_seconds,
			total_alloc_mib,
			heap_alloc_mib,
			user_agent,
			lang,
			resolution_width,
			resolution_height,
			viewport_width,
			viewport_height,
			browser_window_width,
			browser_window_height,
			hardware_concurrency
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)`,
		stats.GameId, stats.GameType, stats.Move,
		t.TreeSize, t.MaxDepth, t.Iterations, t.Elapsed.Seconds(),
		t.TotalAllocMiB, t.HeapAllocMiB,
		u.UserAgent, u.Language, u.Resolution[0], u.Resolution[1],
		u.Viewport[0], u.Viewport[1], u.BrowserWindow[0], u.BrowserWindow[1],
		u.HardwareConcurrency)
	return err
}

func (s *PostgresStore) LoadGame(ctx context.Context, gameId string) (*pb.GameState, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *PostgresStore) Close() error {
	if s.pool != nil {
		s.pool.Close()
	}
	return nil
}
