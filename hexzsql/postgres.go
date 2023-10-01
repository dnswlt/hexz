package hexzsql

import (
	"bytes"
	"compress/gzip"
	"context"
	"database/sql"
	"fmt"
	"io"

	_ "github.com/jackc/pgx/v5/stdlib" // Needed to register pgx as a database/sql driver.
	"google.golang.org/protobuf/proto"

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

func (s *PostgresStore) StoreGame(ctx context.Context, hostId string, gs *pb.GameState) error {
	_, err := s.pool.ExecContext(ctx, `
		INSERT INTO games (game_id, game_type, host_id, host_name) 
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (game_id) DO UPDATE SET game_type = $2, host_id = $3, host_name = $4`,
		gs.GameInfo.Id, string(gs.GameInfo.Type), hostId, gs.GameInfo.Host)
	if err != nil {
		return fmt.Errorf("failed to store game: %w", err)
	}
	if err := s.InsertHistory(ctx, "reset", gs.GameInfo.Id, gs); err != nil {
		return fmt.Errorf("failed to store game history: %w", err)
	}
	return nil
}

func (s *PostgresStore) InsertHistory(ctx context.Context, entryType string, gameId string, gs *pb.GameState) error {
	var gameStateBytes []byte
	if gs != nil {
		enc, err := proto.Marshal(gs)
		if err != nil {
			return fmt.Errorf("failed to marshal game state: %w", err)
		}
		var buf bytes.Buffer
		gz := gzip.NewWriter(&buf)
		if _, err = gz.Write(enc); err != nil {
			return fmt.Errorf("failed to compress game state: %w", err)
		}
		gz.Close()
		gameStateBytes = buf.Bytes()
	}
	_, err := s.pool.ExecContext(ctx, `
		INSERT INTO game_history (game_id, game_state, entry_type)
		VALUES ($1, $2, $3)`,
		gameId, gameStateBytes, entryType)
	if err != nil {
		return fmt.Errorf("failed to store game history: %w", err)
	}
	return nil
}

const (
	// Don't allow more than this many history entries in total.
	// It is an arbitrary choice, the main goal is to avoid using too much
	// DB storage per game, e.g. caused by rogue clients.
	historyEntryLimit = 1000
)

// Used for undo/redo.
type historyEvent struct {
	seqnum    int
	entryType string
}

func (s *PostgresStore) readEventStacks(ctx context.Context, gameId string) (undoStack []historyEvent, redoStack []historyEvent, err error) {
	rows, err := s.pool.QueryContext(ctx, `
	SELECT 
		seqnum,
		entry_type
	FROM game_history
	WHERE game_id = $1
	ORDER BY seqnum`,
		gameId)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to query game history: %w", err)
	}
	count := 0
	for rows.Next() {
		count++
		if count == historyEntryLimit {
			return nil, nil, fmt.Errorf("too many history entries (limit is %d)", historyEntryLimit)
		}
		var seqnum int
		var entryType string
		if err := rows.Scan(&seqnum, &entryType); err != nil {
			return nil, nil, fmt.Errorf("failed to scan game history: %w", err)
		}
		switch entryType {
		case "move":
			undoStack = append(undoStack, historyEvent{seqnum, entryType})
			redoStack = []historyEvent{}
		case "undo":
			if len(undoStack) == 0 {
				return nil, nil, fmt.Errorf("cannot undo: inconsistent game history")
			}
			redoStack = append(redoStack, undoStack[len(undoStack)-1])
			undoStack = undoStack[:len(undoStack)-1]
		case "redo":
			if len(redoStack) == 0 {
				return nil, nil, fmt.Errorf("cannot redo: inconsistent game history")
			}
			undoStack = append(undoStack, redoStack[len(redoStack)-1])
			redoStack = redoStack[:len(redoStack)-1]
		case "reset":
			undoStack = []historyEvent{{seqnum, entryType}}
			redoStack = []historyEvent{}
		case "join":
			undoStack = append(undoStack, historyEvent{seqnum, entryType})
			redoStack = []historyEvent{}
		default:
			return nil, nil, fmt.Errorf("unknown entry type %q", entryType)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("failed to scan game history: %w", err)
	}
	return undoStack, redoStack, nil
}

func (s *PostgresStore) readGameFromHistory(ctx context.Context, gameId string, seqnum int) (*pb.GameState, error) {
	// Load the game state.
	var gameStateBytes []byte
	if err := s.pool.QueryRowContext(ctx, `
		SELECT game_state
		FROM game_history
		WHERE game_id = $1 AND seqnum = $2`,
		gameId, seqnum).Scan(&gameStateBytes); err != nil {
		return nil, fmt.Errorf("failed to query game history: %w", err)
	}
	gameState := &pb.GameState{}
	gz, err := gzip.NewReader(bytes.NewReader(gameStateBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gz.Close()
	protoBytes, err := io.ReadAll(gz)
	if err != nil {
		return nil, fmt.Errorf("failed to read compressed game state: %w", err)
	}
	if err := proto.Unmarshal(protoBytes, gameState); err != nil {
		return nil, fmt.Errorf("failed to unmarshal game state: %w", err)
	}
	return gameState, nil
}

func (s *PostgresStore) PreviousGameState(ctx context.Context, gameId string) (*pb.GameState, error) {
	undoStack, _, err := s.readEventStacks(ctx, gameId)
	if err != nil {
		return nil, err
	}
	if len(undoStack) <= 1 {
		// We cannot undo the first event, which is always a reset.
		return nil, fmt.Errorf("cannot undo: no moves to undo")
	}
	if undoStack[len(undoStack)-1].entryType != "move" {
		return nil, fmt.Errorf("only moves can be undone")
	}
	return s.readGameFromHistory(ctx, gameId, undoStack[len(undoStack)-2].seqnum)
}

func (s *PostgresStore) NextGameState(ctx context.Context, gameId string) (*pb.GameState, error) {
	_, redoStack, err := s.readEventStacks(ctx, gameId)
	if err != nil {
		return nil, err
	}
	if len(redoStack) == 0 {
		return nil, fmt.Errorf("cannot redo: no moves to redo")
	}
	return s.readGameFromHistory(ctx, gameId, redoStack[len(redoStack)-1].seqnum)
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
