package hexz

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
	"google.golang.org/protobuf/proto"
)

type CPUPlayer interface {
	SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error)
	MaxThinkTime() time.Duration
}

type LocalCPUPlayer struct {
	playerId PlayerId
	// Channel over which moves are sent
	mcts         *MCTS
	thinkTime    time.Duration // Current think time (auto-adjusted based on confidence)
	maxThinkTime time.Duration // Maximum think time. Upper bound, independent of the context's deadline.
}

func NewLocalCPUPlayer(playerId PlayerId, maxThinkTime time.Duration) *LocalCPUPlayer {
	return &LocalCPUPlayer{
		playerId:     playerId,
		mcts:         NewMCTS(),
		thinkTime:    maxThinkTime,
		maxThinkTime: maxThinkTime,
	}
}

func (cpu *LocalCPUPlayer) MaxThinkTime() time.Duration {
	return cpu.maxThinkTime
}

// Calculates a suggested move (using MCTS) and sends a ControlEventMove to respCh.
// This method should be called in a separate goroutine.
// The GameEngineFlagz ge will not be modified.
func (cpu *LocalCPUPlayer) SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error) {
	t := cpu.thinkTime
	if t > cpu.maxThinkTime {
		t = cpu.maxThinkTime
	}
	mv, stats := cpu.mcts.SuggestMove(ge, t)
	if minQ := stats.MinQ(); minQ >= 0.98 || minQ <= 0.02 {
		// Speed up if we think we (almost) won or lost, but stop at 0.1% of maxThinkTime.
		if t > cpu.maxThinkTime/500 {
			cpu.thinkTime = t / 2
		}
	} else {
		cpu.thinkTime = cpu.maxThinkTime // use full time allowed.
	}
	moveEvals := make([]*pb.SuggestMoveStats_MoveEval, len(stats.Moves))
	for i, m := range stats.Moves {
		moveEvals[i] = &pb.SuggestMoveStats_MoveEval{
			Row:        int32(m.Row),
			Col:        int32(m.Col),
			Type:       pb.Field_CellType(m.CellType),
			Evaluation: float32(m.Iterations) / float32(stats.Iterations),
		}
	}
	return &GameEngineMove{
			PlayerNum: ge.B.Turn,
			Move:      mv.Move,
			Row:       mv.Row,
			Col:       mv.Col,
			CellType:  mv.CellType,
		}, &pb.SuggestMoveStats{
			Moves: moveEvals,
		}, nil
}

type RemoteCPUPlayer struct {
	playerId             PlayerId
	baseURL              string // Base URL of the remote CPU player server.
	maxThinkTime         time.Duration
	propagateRPCDeadline bool // Experimental: set to true to propagate the RPC deadline to the server.
}

func NewRemoteCPUPlayer(playerId PlayerId, baseURL string, maxThinkTime time.Duration) *RemoteCPUPlayer {
	return &RemoteCPUPlayer{
		playerId:             playerId,
		baseURL:              baseURL,
		maxThinkTime:         maxThinkTime,
		propagateRPCDeadline: false,
	}
}

func (cpu *RemoteCPUPlayer) MaxThinkTime() time.Duration {
	return cpu.maxThinkTime
}

const (
	// URL path used by the CPU player server.
	CpuSuggestMoveURLPath = "/hexz/cpu/suggest"
	// Used to propagate deadlines from clients to the server.
	HttpHeaderXRequestDeadline = "X-Request-Deadline"
)

func (cpu *RemoteCPUPlayer) SuggestMove(ctx context.Context, ge *GameEngineFlagz) (*GameEngineMove, *pb.SuggestMoveStats, error) {
	st, err := ge.Encode()
	if err != nil {
		return nil, nil, err
	}
	data, err := proto.Marshal(&pb.SuggestMoveRequest{
		MaxThinkTimeMs:  cpu.maxThinkTime.Milliseconds(),
		GameEngineState: st,
	})
	if err != nil {
		return nil, nil, err
	}
	remoteURL, err := url.JoinPath(cpu.baseURL, CpuSuggestMoveURLPath)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot build valid URL from %s and %s: %w", cpu.baseURL, CpuSuggestMoveURLPath, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, remoteURL, bytes.NewReader(data))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/x-protobuf")
	if requestDeadline, ok := ctx.Deadline(); ok && cpu.propagateRPCDeadline {
		// Propagate deadline to the server. It will use this to limit the time it spends thinking.
		req.Header.Set(HttpHeaderXRequestDeadline, requestDeadline.Format(time.RFC3339Nano))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("unexpected status code: %s", resp.Status)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("read response body: %w", err)
	}
	var response pb.SuggestMoveResponse
	if err := proto.Unmarshal(body, &response); err != nil {
		return nil, nil, fmt.Errorf("unmarshal response: %w", err)
	}
	var geMove GameEngineMove
	geMove.DecodeProto(response.Move)
	return &geMove, response.MoveStats, nil
}

type CPUPlayerServer struct {
	config *CPUPlayerServerConfig
}

type CPUPlayerServerConfig struct {
	Addr         string // Same format as http.Server.Addr, e.g. "localhost:8085".
	CpuThinkTime time.Duration
	CpuMaxFlags  int
	TlsCertChain string
	TlsPrivKey   string
}

func NewCPUPlayerServer(config *CPUPlayerServerConfig) *CPUPlayerServer {
	return &CPUPlayerServer{
		config: config,
	}
}

func (s *CPUPlayerServer) handleSuggestMove(w http.ResponseWriter, r *http.Request) {
	var req pb.SuggestMoveRequest
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "cannot read request", http.StatusInternalServerError)
		return
	}
	if err := proto.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid SuggestMoveRequest", http.StatusBadRequest)
		return
	}
	if req.GameEngineState == nil {
		http.Error(w, "missing game engine", http.StatusBadRequest)
		return
	}
	var ge *GameEngineFlagz
	switch req.GameEngineState.State.(type) {
	case *pb.GameEngineState_Flagz:
		ge = NewGameEngineFlagz()
		if err := ge.Decode(req.GameEngineState); err != nil {
			http.Error(w, "invalid game engine state", http.StatusBadRequest)
			return
		}
	default:
		http.Error(w, "unsupported game type", http.StatusBadRequest)
		return
	}
	mcts := NewMCTS()
	thinkTime := time.Duration(req.MaxThinkTimeMs) * time.Millisecond
	if s.config.CpuThinkTime > 0 && thinkTime > s.config.CpuThinkTime {
		thinkTime = s.config.CpuThinkTime
	}
	// If the request has a deadline, don't run longer than that.
	if deadline, ok := r.Context().Deadline(); ok {
		timeLeft := time.Until(deadline)
		if timeLeft < thinkTime {
			thinkTime = timeLeft
		}
	}
	mv, _ := mcts.SuggestMove(ge, thinkTime)

	w.Header().Set("Content-Type", "application/x-protobuf")
	data, err := proto.Marshal(&pb.SuggestMoveResponse{
		Move: mv.Proto(),
		// TODO: add Stats back
	})
	if err != nil {
		http.Error(w, "failed to marshal response", http.StatusInternalServerError)
		return
	}
	w.Write(data)
	if err != nil {
		errorLog.Printf("failed to encode response: %s", err)
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
	}
}

// requestDeadlineHandler sets the request deadline based on the X-Request-Deadline header, if present.
func requestDeadlineHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		d := r.Header.Get(HttpHeaderXRequestDeadline)
		if d == "" {
			next.ServeHTTP(w, r) // No deadline specified.
			return
		}
		deadline, err := time.Parse(time.RFC3339Nano, d)
		if err != nil {
			errorLog.Printf("invalid deadline: %s", err)
			http.Error(w, "invalid deadline", http.StatusBadRequest)
			return
		}
		ctx, cancel := context.WithDeadline(r.Context(), deadline)
		defer cancel()
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *CPUPlayerServer) createMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle(CpuSuggestMoveURLPath, requestDeadlineHandler(http.HandlerFunc(s.handleSuggestMove)))
	return mux
}

func (s *CPUPlayerServer) Serve() {
	srv := &http.Server{
		Addr:    s.config.Addr,
		Handler: s.createMux(),
	}
	infoLog.Print("Listening on ", srv.Addr)
	if s.config.TlsCertChain != "" && s.config.TlsPrivKey != "" {
		errorLog.Fatal(srv.ListenAndServeTLS(s.config.TlsCertChain, s.config.TlsPrivKey))
	}
	errorLog.Fatal(srv.ListenAndServe())
}
