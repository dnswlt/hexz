package hexz

// func TestRemoteCPUPlayer(t *testing.T) {
// 	if testing.Short() {
// 		t.Skip("Don't run http tests in -short mode.")
// 	}
// 	cpuThinkTime := 1 * time.Second // Don't think for too long.
// 	cfg := &CPUPlayerServerConfig{
// 		CpuThinkTime: cpuThinkTime,
// 	}
// 	srv := NewCPUPlayerServer(cfg)
// 	testServer := httptest.NewServer(srv.createMux())
// 	defer testServer.Close()

// 	var cpu CPUPlayer = NewRemoteCPUPlayer("cpuPlayerId", testServer.URL, cpuThinkTime)
// 	ge := NewGameEngineFlagz()
// 	// Allow enough time to let the RPC succeed.
// 	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
// 	defer cancel()
// 	mv, _, err := cpu.SuggestMove(ctx, ge)
// 	if err != nil {
// 		t.Fatal("failed to get move suggestion: ", err)
// 	}
// 	if !ge.MakeMove(*mv) {
// 		t.Fatal("failed to make move")
// 	}
// }

// func TestRemoteCPUPlayerDeadlineExceeded(t *testing.T) {
// 	// This test checks that the RPC times out if the CPU player takes too long.
// 	if testing.Short() {
// 		t.Skip("Don't run http tests in -short mode.")
// 	}
// 	// Allow to think longer than the deadline. cpuThinkTime determines the min runtime of this test,
// 	// b/c the server won't shut down before it is done thinking.
// 	cpuThinkTime := 100 * time.Millisecond
// 	rpcDeadline := 10 * time.Millisecond
// 	cfg := &CPUPlayerServerConfig{
// 		CpuThinkTime: cpuThinkTime,
// 	}
// 	srv := NewCPUPlayerServer(cfg)
// 	testServer := httptest.NewServer(srv.createMux())
// 	defer testServer.Close()

// 	var cpu CPUPlayer = NewRemoteCPUPlayer("cpuPlayerId", testServer.URL, cpuThinkTime)
// 	ge := NewGameEngineFlagz()
// 	// Very short deadline of 10ms. The RPC should fail with a timeout.
// 	ctx, cancel := context.WithTimeout(context.Background(), rpcDeadline)
// 	defer cancel()
// 	_, _, err := cpu.SuggestMove(ctx, ge)
// 	if err == nil {
// 		t.Error("expected a deadline error, but request succeeded")
// 	}
// 	if err != nil && !errors.Is(err, context.DeadlineExceeded) {
// 		t.Error("unexpected error: ", err)
// 	}
// }

// func TestRemoteCPUPlayerDeadlineExceededWithPropagation(t *testing.T) {
// 	// This test checks that the RPC times out if the CPU player takes too long.
// 	// The client's deadline is passed on to the server in the X-Requested-Deadline header,
// 	// and the server should use this deadline to adjust the think time.
// 	// This test will time out if the server does not respect the deadline.
// 	if testing.Short() {
// 		t.Skip("Don't run http tests in -short mode.")
// 	}
// 	cpuThinkTime := 1 * time.Hour // Allow CPU to think "forever" (unless adjusted by RPC deadline).
// 	rpcDeadline := 10 * time.Millisecond
// 	cfg := &CPUPlayerServerConfig{
// 		CpuThinkTime: cpuThinkTime,
// 	}
// 	srv := NewCPUPlayerServer(cfg)
// 	testServer := httptest.NewServer(srv.createMux())

// 	cpu := NewRemoteCPUPlayer("cpuPlayerId", testServer.URL, cpuThinkTime)
// 	// Crucially, enable deadline propagation, so server will adjust its think time.
// 	cpu.propagateRPCDeadline = true
// 	ge := NewGameEngineFlagz()
// 	// Very short deadline of 10ms. The RPC should fail with a timeout.
// 	ctx, cancel := context.WithTimeout(context.Background(), rpcDeadline)
// 	defer cancel()
// 	_, _, err := cpu.SuggestMove(ctx, ge)
// 	if err != nil && !errors.Is(err, context.DeadlineExceeded) {
// 		t.Error("unexpected error: ", err)
// 	}
// 	ch := make(chan struct{})
// 	go func() {
// 		testServer.Close() // This will take ~cpuThinkTime unless the server respects the deadline.
// 		ch <- struct{}{}
// 	}()
// 	select {
// 	case <-ch:
// 		// Good: server was closed.
// 	case <-time.After(5 * rpcDeadline):
// 		t.Fatal("Did not close the server fast enough")
// 	}
// }

// func TestRemoteCPUPlayerWrongURLPath(t *testing.T) {
// 	if testing.Short() {
// 		t.Skip("Don't run http tests in -short mode.")
// 	}
// 	cpuThinkTime := 10 * time.Millisecond
// 	cfg := &CPUPlayerServerConfig{
// 		CpuThinkTime: cpuThinkTime,
// 	}
// 	srv := NewCPUPlayerServer(cfg)
// 	testServer := httptest.NewServer(srv.createMux())
// 	defer testServer.Close()

// 	var cpu CPUPlayer = NewRemoteCPUPlayer("cpuPlayerId", testServer.URL+"/wrongpath", cpuThinkTime)
// 	ge := NewGameEngineFlagz()
// 	_, _, err := cpu.SuggestMove(context.Background(), ge)
// 	if err == nil {
// 		t.Error("request for wrong path succeeded")
// 	} else if !strings.Contains(err.Error(), "404") {
// 		t.Error("wanted 404 error, got: ", err)
// 	}
// }

// func TestRemoteCPUPlayerInvalidRequestData(t *testing.T) {
// 	s := NewCPUPlayerServer(&CPUPlayerServerConfig{})
// 	w := httptest.NewRecorder()
// 	// Create bogus request
// 	requestData := "{invalid json}"
// 	r := httptest.NewRequest(http.MethodPost, "/hexz/new", strings.NewReader(requestData))
// 	s.handleSuggestMove(w, r)
// 	if w.Code != http.StatusBadRequest {
// 		t.Errorf("expected status code 400, got %d", w.Code)
// 	}
// }

// func TestRemoteCPUPlayerIncompleteRequestData(t *testing.T) {
// 	// Tests that the handler defensively checks its inputs and does not crash on incomplete data.
// 	s := NewCPUPlayerServer(&CPUPlayerServerConfig{})
// 	w := httptest.NewRecorder()
// 	ge := NewGameEngineFlagz()
// 	ge.B.FlatFields = []Field{
// 		{Type: cellFlag, Owner: 1},
// 	} // Make board invalid.
// 	state, _ := ge.Encode()
// 	req, err := proto.Marshal(&pb.SuggestMoveRequest{
// 		MaxThinkTimeMs:  10,
// 		GameEngineState: state,
// 	})
// 	if err != nil {
// 		t.Fatal("marshal SuggestMoveRequest: ", err)
// 	}
// 	r := httptest.NewRequest(http.MethodPost, CpuSuggestMoveURLPath, bytes.NewReader(req))
// 	s.handleSuggestMove(w, r)
// 	if w.Code != http.StatusBadRequest {
// 		t.Errorf("expected status code 400, got %d", w.Code)
// 	}
// 	wantErr := "invalid game engine state"
// 	if !strings.Contains(w.Body.String(), wantErr) {
// 		t.Errorf("expected error message %q, got: %q", wantErr, w.Body.String())
// 	}
// }
