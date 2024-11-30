package hexzsql

import (
	"context"
	"encoding/hex"
	"flag"
	"testing"
	"time"

	crand "crypto/rand"

	"github.com/dnswlt/hexz/internal/api"
	pb "github.com/dnswlt/hexz/pkg/hexzpb"
	tpb "google.golang.org/protobuf/types/known/timestamppb"
)

var (
	// Set to sth like -test-postgres-url="postgres://hexz_test:hexz_test@nuc:5432/hexz_test"
	testPostgresURL = flag.String("test-postgres-url", "postgres://hexz_test:hexz_test@nuc:5432/hexz_test", "PostgresSQL URL for testing")
)

// Returns a unique game ID that can be used for DB integration tests
// to avoid collisions between tests.
func uniqueTestGameId() string {
	p := make([]byte, 16) // 128 bits of random goodness
	crand.Read(p)
	return time.Now().Format("20060102150405") + "-" + hex.EncodeToString(p)
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
