package hexzsql

import (
	"context"
	"flag"
	"testing"

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
)

var (
	testPostgresURL = flag.String("test-postgres-url", "", "PostgresSQL URL for testing")
)

func TestPostgresDatabase(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Set --test-postgres-url to sth like \"postgres://hexz:hexz@${HOST}:5432/hexz\" to run this test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	gameId := "test_game_id"
	gs := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Type: "Test",
		},
	}
	if err := db.StoreGame(ctx, gameId, gs); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresInsertStats(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Set --test-postgres-url to sth like \"postgres://hexz:hexz@${HOST}:5432/hexz\" to run this test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	stats := &hexz.WASMStatsRequest{
		GameId:   "test_game_id",
		Move:     1,
		GameType: "Test",
		Stats: hexz.WASMStats{
			TreeSize:   42,
			Iterations: 1000,
		},
		UserInfo: hexz.UserInfo{
			UserAgent:  "Golang_Test",
			Language:   "en-US",
			Resolution: [2]int{800, 600},
		},
	}
	if err := db.InsertStats(ctx, stats); err != nil {
		t.Fatal(err)
	}
}
