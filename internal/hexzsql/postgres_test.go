package hexzsql

import (
	"context"
	"encoding/hex"
	"errors"
	"flag"
	"math"
	"regexp"
	"testing"
	"time"

	crand "crypto/rand"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	"github.com/google/uuid"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

var (
	// Set to sth like -test-postgres-url="postgres://hexz_test:hexz_test@nuc:5432/hexz_test"
	testPostgresURL = flag.String("test-postgres-url", "", "PostgresSQL URL for testing")
)

// Returns a unique game ID that can be used for DB integration tests
// to avoid collisions between tests.
func uniqueTestGameId() string {
	p := make([]byte, 16) // 128 bits of random goodness
	crand.Read(p)
	return time.Now().Format("20060102150405") + "-" + hex.EncodeToString(p)
}

func uniqueTestEmail() string {
	p := make([]byte, 4) // 32 bits of random goodness
	crand.Read(p)
	user := "user-" + time.Now().Format("20060102150405") + "-" + hex.EncodeToString(p)
	return user + "@test.example.com"
}

func TestPostgresStoreGame(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	gs := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Id:   "TestPostgresDatabase",
			Type: "TestType",
			Host: "test_host",
		},
	}
	if err := db.StoreGame(ctx, "test_host_id", gs); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresInsertHistory_LoadGame(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	gameId := uniqueTestGameId()
	hostId := "host_" + gameId
	gs := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Id:      gameId,
			Started: tpb.Now(),
			Type:    "TestType",
			Host:    "test_host",
		},
	}
	if err := db.StoreGame(ctx, hostId, gs); err != nil {
		t.Fatal(err)
	}
	bs := &api.BoardStatus{
		Move:  1,
		Score: []int{3, 1},
	}
	if err := db.InsertHistory(ctx, "move", gameId, gs, bs); err != nil {
		t.Fatal("InsertHistory failed: ", err)
	}
}

func TestPostgresInsertHistory_WithBoardStatus(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	gameId := uniqueTestGameId()
	hostId := "host_" + gameId
	gs := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Id:      gameId,
			Started: tpb.Now(),
			Type:    "TestType",
			Host:    "test_host",
		},
	}
	if err := db.StoreGame(ctx, hostId, gs); err != nil {
		t.Fatal(err)
	}
	bs := &api.BoardStatus{
		Move:  1,
		Score: []int{3, 1},
	}
	if err := db.InsertHistory(ctx, "move", gameId, gs, bs); err != nil {
		t.Fatal("InsertHistory failed: ", err)
	}
	rows, err := db.listRecentGamesForPlayer(ctx, hostId, 0, 1)
	if err != nil {
		t.Fatal("ListRecentGames failed: ", err)
	}
	if len(rows) != 1 {
		t.Fatalf("Expected 1 row, got %d", len(rows))
	}
	row := rows[0]
	if row.GameID != gameId {
		t.Errorf("Wrong game ID: want %q, got %s", gameId, row.GameID)
	}
	if row.ScoreP1 != 3 || row.ScoreP2 != 1 {
		t.Errorf("Wrong score: want 3-1, got %d-%d", row.ScoreP1, row.ScoreP2)
	}
}

func TestPostgresInsertStats(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	stats := &api.WASMStatsRequest{
		GameId:   "TestPostgresInsertStats",
		Move:     1,
		GameType: "Test",
		Stats: api.WASMStats{
			TreeSize:   42,
			Iterations: 1000,
		},
		UserInfo: api.UserInfo{
			UserAgent:  "Golang_Test",
			Language:   "en-US",
			Resolution: [2]int{800, 600},
		},
	}
	if err := db.InsertStats(ctx, stats); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresFindUser(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	email := uniqueTestEmail()
	uid := uuid.New()
	_, err = db.pool.ExecContext(ctx, `
		INSERT INTO users (
			id, 	
			email, 
			password_hash, 
			player_name, 
			is_verified
		) VALUES (
			$1,
			$2,
			'hashed_pwd',
			'John Doe',
			FALSE
		)
	`, uid, email)
	if err != nil {
		t.Fatalf("Failed to INSERT into table: %v", err)
	}
	user, err := db.FindUser(ctx, email)
	if err != nil {
		t.Fatalf("FindUser failed: %v", err)
	}
	if user.Email != email {
		t.Errorf("Wrong email: want %q, got %q", email, user.Email)
	}
	if user.PlayerName != "John Doe" {
		t.Errorf("DisplayName: Want \"John Doe\", got %v", user.PlayerName)
	}
	if user.PasswordHash != "hashed_pwd" {
		t.Errorf("PasswordHash: Want \"hashed_pwd\", got %v", user.PasswordHash)
	}
	now := time.Now()
	if math.Abs(float64(user.CreatedAt.Unix()-now.Unix())) > 300 {
		t.Errorf("CreatedAt unexpected: %v (want within 5m of %v)", user.CreatedAt, now)
	}
	if math.Abs(float64(user.UpdatedAt.Unix()-now.Unix())) > 300 {
		t.Errorf("UpdatedAt unexpected: %v (want within 5m of %v)", user.UpdatedAt, now)
	}
}

func TestPostgresAddFindUser(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	email := uniqueTestEmail()
	err = db.AddUser(ctx, &User{
		Email:        email,
		PasswordHash: "hashed_pwd",
		PlayerName:   "John Doe",
	})
	if err != nil {
		t.Fatalf("AddUser failed: %v", err)
	}
	user, err := db.FindUser(ctx, email)
	if err != nil {
		t.Fatalf("FindUser failed: %v", err)
	}
	uuidRegexp := regexp.MustCompile(`^[a-zA-Z0-9]{8}(-[a-zA-Z0-9]{4}){3}-[a-zA-Z0-9]{12}$`)
	if !uuidRegexp.MatchString(user.ID) {
		t.Errorf("User ID not a UUID: %q", user.ID)
	}
	if user.Email != email {
		t.Errorf("Wrong email: want %q, got %q", email, user.Email)
	}
	if user.PlayerName != "John Doe" {
		t.Errorf("DisplayName: Want \"John Doe\", got %v", user.PlayerName)
	}
	if user.PasswordHash != "hashed_pwd" {
		t.Errorf("PasswordHash: Want \"hashed_pwd\", got %v", user.PasswordHash)
	}
	if user.AccountStatus != AccountStatusNew {
		t.Errorf("AccountStatus: want new, got %q", user.AccountStatus)
	}
	now := time.Now()
	if math.Abs(float64(user.CreatedAt.Unix()-now.Unix())) > 300 {
		t.Errorf("CreatedAt unexpected: %v (want within 5m of %v)", user.CreatedAt, now)
	}
	if math.Abs(float64(user.UpdatedAt.Unix()-now.Unix())) > 300 {
		t.Errorf("UpdatedAt unexpected: %v (want within 5m of %v)", user.UpdatedAt, now)
	}
}

func TestPostgresAddAddUser(t *testing.T) {
	// Add user twice, should return "already exists" error.
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	user := &User{
		Email:        uniqueTestEmail(),
		PasswordHash: "hashed_pwd",
		PlayerName:   "John Doe",
	}
	err = db.AddUser(ctx, user)
	if err != nil {
		t.Fatalf("AddUser failed: %v", err)
	}
	user.ID = "" // Delete ID, pretend it's a fresh user with the same email.
	err = db.AddUser(ctx, user)
	if err != ErrUserAlreadyExists {
		t.Errorf("AddUser second time: want ErrUserAlreadyExists, got %v", err)
	}
}

func TestPostgresAddUpdateUser(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	email := uniqueTestEmail()
	err = db.AddUser(ctx, &User{
		Email:        email,
		PasswordHash: "hashed_pwd",
		PlayerName:   "John Doe",
	})
	if err != nil {
		t.Fatalf("AddUser failed: %v", err)
	}
	user, err := db.FindUser(ctx, email)
	if err != nil {
		t.Fatalf("FindUser failed: %v", err)
	}
	// Update some fields
	user.AccountStatus = AccountStatusBlocked
	user.PlayerName = "Franz Josef"
	// Update the UpdatedAt time, but expect this to be ignored.
	updatedAt := time.Date(1999, 12, 31, 12, 0, 0, 0, time.UTC)
	user.UpdatedAt = updatedAt

	err = db.UpdateUser(ctx, user)

	if err != nil {
		t.Fatalf("UpdateUser failed: %v", err)
	}
	user, err = db.FindUser(ctx, email)
	if err != nil {
		t.Fatalf("FindUser failed: %v", err)
	}

	if user.Email != email {
		t.Errorf("Wrong email: want %q, got %q", email, user.Email)
	}
	if user.PlayerName != "Franz Josef" {
		t.Errorf("DisplayName: Want \"John Doe\", got %v", user.PlayerName)
	}
	if user.PasswordHash != "hashed_pwd" {
		t.Errorf("PasswordHash: Want \"hashed_pwd\", got %v", user.PasswordHash)
	}
	if user.AccountStatus != AccountStatusBlocked {
		t.Errorf("AccountStatus: want blocked, got %v", user.AccountStatus)
	}
	if user.UpdatedAt.Equal(updatedAt) {
		t.Errorf("UpdatedAt should not be equal to what we set: got %v)", user.UpdatedAt)
	}
}

func TestPostgresVerifyUserInvalidToken(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	email := uniqueTestEmail()
	token := uuid.New().String()
	err = db.AddUser(ctx, &User{
		Email:             email,
		PasswordHash:      "hashed_pwd",
		PlayerName:        "John Doe",
		VerificationToken: token,
		TokenExpiry:       time.Now().Add(24 * time.Hour),
	})
	if err != nil {
		t.Fatalf("AddUser failed: %v", err)
	}
	err = db.VerifyUser(ctx, token+"_invalid")
	if !errors.Is(err, ErrVerificationFailed) {
		t.Errorf("VerifyUser with invalid token had unexpected error: %v", err)
	}
}

func TestPostgresVerifyUserExpiredToken(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	email := uniqueTestEmail()
	token := uuid.New().String()
	err = db.AddUser(ctx, &User{
		Email:             email,
		PasswordHash:      "hashed_pwd",
		PlayerName:        "John Doe",
		VerificationToken: token,
		// Token expired 24h ago:
		TokenExpiry: time.Now().Add(-24 * time.Hour),
	})
	if err != nil {
		t.Fatalf("AddUser failed: %v", err)
	}
	err = db.VerifyUser(ctx, token)
	if !errors.Is(err, ErrVerificationFailed) {
		t.Errorf("VerifyUser with invalid token had unexpected error: %v", err)
	}
}

func TestPostgresVerifyUserSuccess(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Flag -test-postgres-url is not set. Skipping DB integration test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	email := uniqueTestEmail()
	token := uuid.New().String()
	err = db.AddUser(ctx, &User{
		Email:             email,
		PasswordHash:      "hashed_pwd",
		PlayerName:        "John Doe",
		VerificationToken: token,
		TokenExpiry:       time.Now().Add(24 * time.Hour),
	})
	if err != nil {
		t.Fatalf("AddUser failed: %v", err)
	}
	err = db.VerifyUser(ctx, token)
	if err != nil {
		t.Errorf("VerifyUser failed: %v", err)
	}
	u, err := db.FindUser(ctx, email)
	if err != nil {
		t.Fatalf("FindUser failed: %v", err)
	}
	if u.AccountStatus != AccountStatusActive {
		t.Errorf("Expected account in status active(%d), got %d", AccountStatusActive, u.AccountStatus)
	}
	// It should fail the second time:
	err = db.VerifyUser(ctx, token)
	if !errors.Is(err, ErrVerificationFailed) {
		t.Errorf("VerifyUser did not fail as expected on the second attempt: %v", err)
	}
}
