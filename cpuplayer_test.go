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
	// This test checks that the RPC times out if the CPU player takes too long.
	if testing.Short() {
		return // Don't run http tests in -short mode.
	}
	// Allow to think longer than the deadline. cpuThinkTime determines the min runtime of this test,
	// b/c the server won't shut down before it is done thinking.
	cpuThinkTime := 100 * time.Millisecond
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
	if err == nil {
		t.Error("expected a deadline error, but request succeeded")
	}
	if err != nil && !errors.Is(err, context.DeadlineExceeded) {
		t.Error("unexpected error: ", err)
	}
}

func TestRemoteCPUPlayerDeadlineExceededWithPropagation(t *testing.T) {
	// This test checks that the RPC times out if the CPU player takes too long.
	// The client's deadline is passed on to the server in the X-Requested-Deadline header,
	// and the server should use this deadline to adjust the think time.
	// This test will time out if the server does not respect the deadline.
	if testing.Short() {
		return // Don't run http tests in -short mode.
	}
	cpuThinkTime := 1 * time.Hour // Allow CPU to think "forever" (unless adjusted by RPC deadline).
	rpcDeadline := 10 * time.Millisecond
	cfg := &CPUPlayerServerConfig{
		CpuThinkTime: cpuThinkTime,
	}
	srv := NewCPUPlayerServer(cfg)
	testServer := httptest.NewServer(srv.createMux())

	cpu := NewRemoteCPUPlayer("cpuPlayerId", testServer.URL, cpuThinkTime)
	// Crucially, enable deadline propagation, so server will adjust its think time.
	cpu.propagateRPCDeadline = true
	ge := NewGameEngineFlagz(rand.NewSource(0))
	// Very short deadline of 10ms. The RPC should fail with a timeout.
	ctx, cancel := context.WithTimeout(context.Background(), rpcDeadline)
	defer cancel()
	_, err := cpu.SuggestMove(ctx, ge)
	if err != nil && !errors.Is(err, context.DeadlineExceeded) {
		t.Error("unexpected error: ", err)
	}
	ch := make(chan struct{})
	go func() {
		testServer.Close() // This will take ~cpuThinkTime unless the server respects the deadline.
		ch <- struct{}{}
	}()
	select {
	case <-ch:
		// Good: server was closed.
	case <-time.After(5 * rpcDeadline):
		t.Fatal("Did not close the server fast enough")
	}
}
