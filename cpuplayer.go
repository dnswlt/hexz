package hexz

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	pb "github.com/dnswlt/hexz/hexzpb"
	"google.golang.org/protobuf/proto"
)

type CPUPlayer interface {
	SuggestMove(ctx context.Context, ge SinglePlayerGameEngine) (ControlEvent, error)
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
// The SinglePlayerGameEngine passed in will not be modified.
func (cpu *LocalCPUPlayer) SuggestMove(ctx context.Context, ge SinglePlayerGameEngine) (ControlEvent, error) {
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
	return ControlEventMove{
		playerId:  cpu.playerId,
		mctsStats: stats,
		moveRequest: &MoveRequest{
			Move: mv.Move,
			Row:  mv.Row,
			Col:  mv.Col,
			Type: mv.CellType,
		},
	}, nil
}

type RemoteCPUPlayer struct {
	playerId             PlayerId
	url                  string // Base URL of the remote CPU player server.
	maxThinkTime         time.Duration
	propagateRPCDeadline bool // Experimental: set to true to propagate the RPC deadline to the server.
}

func NewRemoteCPUPlayer(playerId PlayerId, url string, maxThinkTime time.Duration) *RemoteCPUPlayer {
	return &RemoteCPUPlayer{
		playerId:             playerId,
		url:                  url,
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

func (cpu *RemoteCPUPlayer) SuggestMove(ctx context.Context, spge SinglePlayerGameEngine) (ControlEvent, error) {
	// For now, only flagz engines are supported.
	ge, ok := spge.(*GameEngineFlagz)
	if !ok {
		return nil, fmt.Errorf("remote CPU player only supports flagz engines")
	}
	st, err := ge.Encode()
	if err != nil {
		return nil, err
	}
	geState, err := proto.Marshal(st)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(&SuggestMoveRequest{
		MaxThinkTime:    cpu.maxThinkTime,
		GameEngineState: geState,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cpu.url+CpuSuggestMoveURLPath, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if requestDeadline, ok := ctx.Deadline(); ok && cpu.propagateRPCDeadline {
		// Propagate deadline to the server. It will use this to limit the time it spends thinking.
		req.Header.Set(HttpHeaderXRequestDeadline, requestDeadline.Format(time.RFC3339Nano))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %s", resp.Status)
	}
	defer resp.Body.Close()
	var respData SuggestMoveResponse
	dec := json.NewDecoder(resp.Body)
	if err := dec.Decode(&respData); err != nil {
		return nil, err
	}
	return ControlEventMove{
		playerId:    cpu.playerId,
		moveRequest: respData.Move,
		mctsStats:   respData.Stats,
	}, nil
}

// API for remote CPU player.

type SuggestMoveRequest struct {
	MaxThinkTime time.Duration `json:"maxThinkTime"`
	// Encoded game engine state (obtained via SinglePlayerGameEngine.Encode).
	GameEngineState []byte `json:"gameEngineState"`
}

type SuggestMoveResponse struct {
	Move  *MoveRequest `json:"move"`
	Stats *MCTSStats   `json:"stats"`
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
	var req SuggestMoveRequest
	dec := json.NewDecoder(r.Body)
	err := dec.Decode(&req)
	if err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if req.GameEngineState == nil {
		http.Error(w, "missing game engine", http.StatusBadRequest)
		return
	}
	// Reconstruct game engine from encoded state.
	state := &pb.GameState{}
	if err := proto.Unmarshal(req.GameEngineState, state); err != nil {
		http.Error(w, "invalid game engine state", http.StatusBadRequest)
		return
	}
	var ge SinglePlayerGameEngine
	switch state.State.(type) {
	case *pb.GameState_Flagz:
		ge = NewGameEngineFlagz()
		if err := ge.Decode(state); err != nil {
			http.Error(w, "invalid game engine state", http.StatusBadRequest)
			return
		}
	default:
		http.Error(w, "unsupported game type", http.StatusBadRequest)
		return
	}
	mcts := NewMCTS()
	thinkTime := req.MaxThinkTime
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
	mv, stats := mcts.SuggestMove(ge, thinkTime)

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	err = enc.Encode(&SuggestMoveResponse{
		Move: &MoveRequest{
			Move: mv.Move,
			Row:  mv.Row,
			Col:  mv.Col,
			Type: mv.CellType,
		},
		Stats: stats,
	})
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
