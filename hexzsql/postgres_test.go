package hexzsql

import (
	"context"
	"flag"
	"testing"

	"github.com/dnswlt/hexz"
	pb "github.com/dnswlt/hexz/hexzpb"
)

var (
	testPostgresURL = flag.String("test-postgres-url", "postgres://hexz_test:hexz_test@nuc:5432/hexz_test", "PostgresSQL URL for testing")
)

func TestPostgresDatabase(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Set --test-postgres-url to sth like \"postgres://hexz_test:hexz_test@localhost:5432/hexz_test\" to run this test.")
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

func TestPostgresInsertStats(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Set --test-postgres-url to sth like \"postgres://hexz_test:hexz_test@localhost:5432/hexz_test\" to run this test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	stats := &hexz.WASMStatsRequest{
		GameId:   "TestPostgresInsertStats",
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

func TestPostgresUndo(t *testing.T) {
	if *testPostgresURL == "" {
		t.Skip("Set --test-postgres-url to sth like \"postgres://hexz_test:hexz_test@localhost:5432/hexz_test\" to run this test.")
	}
	ctx := context.Background()
	db, err := NewPostgresStore(ctx, *testPostgresURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	// Clean up from previous runs.
	if _, err := db.pool.ExecContext(ctx, "DELETE FROM game_history WHERE game_id = 'TestPostgresUndo'"); err != nil {
		t.Fatal("Failed to clean up database: ", err)
	}
	gs := &pb.GameState{
		GameInfo: &pb.GameInfo{
			Id:   "TestPostgresUndo",
			Type: "TestType",
			Host: "Player1",
		},
		Players: []*pb.Player{
			{
				Id:   "P1",
				Name: "Player1",
			},
		},
		Seqnum: 1,
	}
	if err := db.StoreGame(ctx, "P1", gs); err != nil {
		t.Fatal(err)
	}
	// Second player joins.
	gs.Players = append(gs.Players, &pb.Player{
		Id:   "P2",
		Name: "Player2",
	})
	gs.Seqnum++
	seqnumAferJoin := gs.Seqnum
	if err := db.InsertHistory(ctx, "join", gs.GameInfo.Id, gs); err != nil {
		t.Fatal(err)
	}
	// We shouldn't be able to undo anything so far.
	_, err = db.PreviousGameState(ctx, gs.GameInfo.Id)
	if err == nil {
		t.Fatal("Unexpected previous game state before first move")
	}
	// First move.
	gs.Seqnum++
	if err = db.InsertHistory(ctx, "move", gs.GameInfo.Id, gs); err != nil {
		t.Fatal("Failed to insert first move: ", err)
	}
	// Should be able to call undo once.
	prev, err := db.PreviousGameState(ctx, gs.GameInfo.Id)
	if err != nil {
		t.Fatal("Couldn't undo first move: ", err)
	}
	if prev.Seqnum != seqnumAferJoin {
		t.Fatalf("Expected seqnum %d, got %d", seqnumAferJoin, prev.Seqnum)
	}
	if err = db.InsertHistory(ctx, "undo", gs.GameInfo.Id, nil); err != nil {
		t.Fatal("Failed to insert undo: ", err)
	}
	// ... but not a second time. That would "unjoin" the second player.
	_, err = db.PreviousGameState(ctx, gs.GameInfo.Id)
	if err == nil {
		t.Fatal("Should be able to undo only once after first move")
	}
	// First move, second time.
	prev.Seqnum++
	seqnumFirstMove := prev.Seqnum
	if err = db.InsertHistory(ctx, "move", gs.GameInfo.Id, prev); err != nil {
		t.Fatal("Failed to insert first move the second time: ", err)
	}
	// And another move.
	prev.Seqnum++
	if err = db.InsertHistory(ctx, "move", gs.GameInfo.Id, prev); err != nil {
		t.Fatal("Failed to insert second move: ", err)
	}
	// Should be able to undo twice now.
	prev, err = db.PreviousGameState(ctx, gs.GameInfo.Id)
	if err != nil {
		t.Fatal("Couldn't undo second move: ", err)
	}
	if prev.Seqnum != seqnumFirstMove {
		t.Fatalf("Expected seqnum %d, got %d", seqnumFirstMove, prev.Seqnum)
	}
	if err = db.InsertHistory(ctx, "undo", gs.GameInfo.Id, nil); err != nil {
		t.Fatal("Failed to insert undo: ", err)
	}
	_, err = db.PreviousGameState(ctx, gs.GameInfo.Id)
	if err != nil {
		t.Fatal("Couldn't undo first move: ", err)
	}
}
