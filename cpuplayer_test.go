package hexz

import (
	"context"
	"errors"
	"math/rand"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRemoteCPUPlayer(t *testing.T) {
	if testing.Short() {
		return // Don't run http tests in -short mode.
	}
	cpuThinkTime := 1 * time.Second // Don't think for too long.
	cfg := &CPUPlayerServerConfig{
		CpuThinkTime: cpuThinkTime,
	}
	srv := NewCPUPlayerServer(cfg)
	testServer := httptest.NewServer(srv.createMux())
	defer testServer.Close()

	var cpu CPUPlayer = NewRemoteCPUPlayer("cpuPlayerId", testServer.URL, cpuThinkTime)
	ge := NewGameEngineFlagz(rand.NewSource(0))
	// Allow enough time to let the RPC succeed.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ev, err := cpu.SuggestMove(ctx, ge)
	if err != nil {
		t.Fatal("failed to get move suggestion: ", err)
	}
	mv, ok := ev.(ControlEventMove)
	if !ok {
		t.Fatalf("unexpected event type: %T", ev)
	}
	if !ge.MakeMove(GameEngineMove{
		PlayerNum: ge.Board().Turn,
		Move:      mv.moveRequest.Move,
		Row:       mv.moveRequest.Row,
		Col:       mv.moveRequest.Col,
		CellType:  mv.moveRequest.Type,
	}) {
		t.Fatal("failed to make move")
	}
}

func TestRemoteCPUPlayerDeadlineExceeded(t *testing.T) {
	if testing.Short() {
		return // Don't run http tests in -short mode.
	}
	cpuThinkTime := 1 * time.Second // Think longer than the deadline allows.
	rpcDeadline := 10 * time.Millisecond
	cfg := &CPUPlayerServerConfig{
		CpuThinkTime: cpuThinkTime,
	}
	srv := NewCPUPlayerServer(cfg)
	testServer := httptest.NewServer(srv.createMux())
	defer testServer.Close()

	var cpu CPUPlayer = NewRemoteCPUPlayer("cpuPlayerId", testServer.URL, cpuThinkTime)
	ge := NewGameEngineFlagz(rand.NewSource(0))
	// Very short deadline of 10ms. The RPC should fail with a timeout.
	ctx, cancel := context.WithTimeout(context.Background(), rpcDeadline)
	defer cancel()
	_, err := cpu.SuggestMove(ctx, ge)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Error("unexpected error: ", err)
	}
}
