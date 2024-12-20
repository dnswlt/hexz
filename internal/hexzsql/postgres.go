package hexzsql

import (
	"bytes"
	"compress/gzip"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/jackc/pgx/v5/pgconn"
	_ "github.com/jackc/pgx/v5/stdlib" // Needed to register pgx as a database/sql driver.
	"google.golang.org/protobuf/proto"

	"github.com/dnswlt/hexz/internal/api"
	"github.com/dnswlt/hexz/internal/hlog"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
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
		INSERT INTO games (
			started,
			game_id,
			game_type,
			cpu_player_mode,
			host_name,
			host_id,
			move,
			score_p1,
			score_p2,
			done
		) 
		VALUES ($1, $2, $3, $4, $5, $6, 0, 0, 0, FALSE)
		ON CONFLICT (game_id) DO UPDATE
			SET 
				started = $1,
				game_type = $3,
				cpu_player_mode = $4,
				host_name = $5,
				host_id = $6
		`,
		gs.GameInfo.Started.AsTime(),
		gs.GameInfo.Id,
		string(gs.GameInfo.Type),
		gs.GameInfo.CpuPlayerMode.String(),
		gs.GameInfo.Host,
		hostId)
	if err != nil {
		return fmt.Errorf("failed to store game: %v", err)
	}
	return s.InsertHistory(ctx, "reset", gs.GameInfo.Id, gs, nil)
}

func (s *PostgresStore) LoadGame(ctx context.Context, gameId string) (*pb.GameState, error) {
	var gameStateBytes []byte
	err := s.pool.QueryRowContext(ctx, `
		SELECT
			game_state
		FROM game_history
		WHERE game_id = $1 AND game_state IS NOT NULL
		ORDER BY seqnum DESC
		LIMIT 1
	`, gameId).Scan(&gameStateBytes)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("game %s not found in DB: %w", gameId, sql.ErrNoRows)
		}
		return nil, fmt.Errorf("failed to read game state from DB: %v", err)
	}
	hlog.Infof("Read game state of size %d bytes for game %s from DB.", len(gameStateBytes), gameId)
	gz, err := gzip.NewReader(bytes.NewReader(gameStateBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %v", err)
	}
	defer gz.Close()
	protoBytes, err := io.ReadAll(gz)
	if err != nil {
		return nil, fmt.Errorf("failed to read compressed game state: %v", err)
	}
	gameState := &pb.GameState{}
	if err := proto.Unmarshal(protoBytes, gameState); err != nil {
		return nil, fmt.Errorf("failed to unmarshal game state from DB: %v", err)
	}
	return gameState, nil
}

func (s *PostgresStore) listRecentGamesForPlayer(ctx context.Context, playerId string, offset int, limit int) ([]*GameRow, error) {
	queryTmpl := `
		SELECT
			created,
			game_id,
			started,
			game_type,
			cpu_player_mode,
			host_name,
			host_id,
			COALESCE(move, 0) AS move,
			COALESCE(score_p1, 0) AS score_p1,
			COALESCE(score_p2, 0) AS score_p2,
			COALESCE(done, FALSE) AS done
		FROM games
		WHERE %s
		ORDER BY started DESC
		LIMIT $1 OFFSET $2`
	args := []any{limit, offset}
	clauses := []string{"move > 0"}
	if playerId != "" {
		args = append(args, playerId)
		clauses = append(clauses, fmt.Sprintf("host_id=$%d", len(args)))
	}
	query := fmt.Sprintf(queryTmpl, strings.Join(clauses, " AND "))
	rows, err := s.pool.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to read games: %v", err)
	}
	defer rows.Close()
	result := []*GameRow{}
	for rows.Next() {
		var game GameRow
		var cpuPlayerMode string
		err := rows.Scan(
			&game.Created,
			&game.GameID,
			&game.Started,
			&game.GameType,
			&cpuPlayerMode,
			&game.Host,
			&game.HostID,
			&game.Move,
			&game.ScoreP1,
			&game.ScoreP2,
			&game.Done,
		)
		if err != nil {
			return nil, fmt.Errorf("error reading row: %v", err)
		}
		game.CPUPlayerMode = pb.CPUPlayerMode_Enum(pb.CPUPlayerMode_Enum_value[cpuPlayerMode])
		result = append(result, &game)
	}
	return result, nil
}

func (s *PostgresStore) ListRecentGames(ctx context.Context, offset int, limit int) ([]*GameRow, error) {
	return s.listRecentGamesForPlayer(ctx, "", offset, limit)
}

func (s *PostgresStore) InsertHistory(ctx context.Context, entryType string, gameId string, gs *pb.GameState, boardStatus *api.BoardStatus) error {
	var gameStateBytes []byte
	if gs != nil {
		var buf bytes.Buffer
		data, err := proto.Marshal(gs)
		if err != nil {
			return fmt.Errorf("failed to marshal game state: %v", err)
		}
		gz := gzip.NewWriter(&buf)
		if _, err := gz.Write(data); err != nil {
			return fmt.Errorf("failed to compress game state: %v", err)
		}
		gz.Close()
		gameStateBytes = buf.Bytes()
		hlog.Infof("Storing game ID %s in DB: game state is %d bytes", gameId, len(gameStateBytes))
	}
	_, err := s.pool.ExecContext(ctx, `
		INSERT INTO game_history (
			game_id,
			game_state,
			entry_type
		)
		VALUES ($1, $2, $3)`,
		gameId, gameStateBytes, entryType)
	if err != nil {
		return fmt.Errorf("failed to store game history: %v", err)
	}
	if boardStatus == nil {
		return nil
	}
	var score [2]int
	copy(score[:], boardStatus.Score) // In case boardStatus has only 0 or 1 scores.
	_, err = s.pool.ExecContext(ctx, `
		UPDATE games
		SET
			move = $2,
			score_p1 = $3,
			score_p2 = $4,
			done = $5
		WHERE game_id = $1`,
		gameId, boardStatus.Move, score[0], score[1], boardStatus.Done)
	if err != nil {
		return fmt.Errorf("failed to update game status: %v", err)
	}
	return nil
}

func (s *PostgresStore) InsertStats(ctx context.Context, stats *api.WASMStatsRequest) error {
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

func (s *PostgresStore) Close() error {
	if s.pool != nil {
		s.pool.Close()
	}
	return nil
}

func (s *PostgresStore) FindUser(ctx context.Context, email string) (*User, error) {
	query := `
		SELECT
			id,
			email,
			password_hash,
			player_name,
			account_status,
			COALESCE(verification_token, ''),
			COALESCE(reset_password_token, ''),
			token_expiry,
			created_at,
			updated_at
		FROM users
		WHERE email = $1
	`
	row := s.pool.QueryRowContext(ctx, query, email)
	user := &User{}
	var tokenExpiry sql.NullTime
	err := row.Scan(
		&user.ID,
		&user.Email,
		&user.PasswordHash,
		&user.PlayerName,
		&user.AccountStatus,
		&user.VerificationToken,
		&user.ResetPasswordToken,
		&tokenExpiry,
		&user.CreatedAt,
		&user.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, ErrUserNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("error reading row: %v", err)
	}
	if tokenExpiry.Valid {
		user.TokenExpiry = tokenExpiry.Time
	}
	return user, nil
}

func (s *PostgresStore) AddUser(ctx context.Context, user *User) error {
	if user.ID != "" {
		return fmt.Errorf("user already has an ID: %v", user.ID)
	}
	err := s.pool.QueryRowContext(ctx, `
		INSERT INTO users (
			email,
			password_hash,
			player_name,
			account_status,
			verification_token,
			reset_password_token,
			token_expiry
		) VALUES (
		  $1, $2, $3, $4, $5, $6, $7
		) RETURNING id`,
		user.Email,
		user.PasswordHash,
		user.PlayerName,
		user.AccountStatus,
		user.VerificationToken,
		user.ResetPasswordToken,
		user.TokenExpiry,
	).Scan(&user.ID)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			// 23505 - unique_violation
			// https://www.postgresql.org/docs/11/errcodes-appendix.html
			return ErrUserAlreadyExists
		}
		return fmt.Errorf("failed to insert user: %w", err)
	}
	return nil
}

func (s *PostgresStore) UpdateUser(ctx context.Context, user *User) error {
	if user.ID == "" {
		return fmt.Errorf("user ID must not be empty for update")
	}
	r, err := s.pool.ExecContext(ctx, `
		UPDATE users
		SET 
			email = $1,
			password_hash = $2,
			player_name = $3,
			account_status = $4,
			verification_token = $5,
			reset_password_token = $6,
			token_expiry = $7
		WHERE id = $8`,
		user.Email,
		user.PasswordHash,
		user.PlayerName,
		user.AccountStatus,
		user.VerificationToken,
		user.ResetPasswordToken,
		user.TokenExpiry,
		user.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update user: %v", err)
	}
	n, err := r.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get affected rows: %v", err)
	}
	if n != 1 {
		return fmt.Errorf("user update failed: %d rows updated", n)
	}
	return nil
}

func (s *PostgresStore) VerifyUser(ctx context.Context, verificationToken string) error {
	if verificationToken == "" {
		return fmt.Errorf("empty verification token")
	}
	r, err := s.pool.ExecContext(ctx, `
		UPDATE users
		SET 
			verification_token = NULL,
			token_expiry = NULL,
			account_status = $1
		WHERE 
			verification_token = $2
			AND account_status = $3
			AND token_expiry > now()`,
		AccountStatusActive,
		verificationToken,
		AccountStatusNew,
	)
	if err != nil {
		return fmt.Errorf("failed to update verification_token: %v", err)
	}
	n, err := r.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get affected rows: %v", err)
	}
	if n == 0 {
		return ErrInvalidToken
	}
	return nil
}
