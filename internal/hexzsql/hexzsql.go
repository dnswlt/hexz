package hexzsql

import (
	"context"
	"errors"
	"time"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
)

type DatabaseStore interface {
	GameStore
	UserStore
}

type GameStore interface {
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

var (
	ErrUserNotFound      = errors.New("user not found")
	ErrUserAlreadyExists = errors.New("user already exists")
	ErrInvalidToken      = errors.New("invalid or non-existing token")
)

type UserStore interface {
	// Looks up a single user by email address.
	// Returns ErrUserNotFound if no user with the given email exists.
	FindUser(ctx context.Context, email string) (*User, error)

	// Adds a new user to the database.
	// The ID field will be populated. An error is returned if
	// it is not empty.
	// CreatedAt and UpdatedAt may be empty, in which case they
	// will be set to "now" in the database.
	AddUser(ctx context.Context, user *User) error

	// Locates a user that has the given verification token.
	// If the token is not expired, the user's account status
	// is set to `active`.
	VerifyUser(ctx context.Context, verificationToken string) error
}

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

type AccountStatus int

const (
	AccountStatusNew     AccountStatus = 0
	AccountStatusActive  AccountStatus = 1
	AccountStatusBlocked AccountStatus = 2
	AccountStatusDeleted AccountStatus = 3
)

type User struct {
	ID                 string
	Email              string
	PasswordHash       string
	PlayerName         string
	AccountStatus      AccountStatus
	VerificationToken  string
	ResetPasswordToken string
	TokenExpiry        time.Time
	CreatedAt          time.Time
	UpdatedAt          time.Time
}
