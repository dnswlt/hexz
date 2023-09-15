package hexz

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"sync"
	"time"
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
	src          rand.Source
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
	// Use our own source of randomness. In the long term, this method will deal
	// with a deserialized game engine (sent via RPC), so the source will be nil.
	ge = ge.Clone(cpu.src)
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
	playerId     PlayerId
	url          string // Base URL of the remote CPU player server.
	maxThinkTime time.Duration
}

func NewRemoteCPUPlayer(playerId PlayerId, url string, maxThinkTime time.Duration) *RemoteCPUPlayer {
	return &RemoteCPUPlayer{
		playerId:     playerId,
		url:          url,
		maxThinkTime: maxThinkTime,
	}
}

func (cpu *RemoteCPUPlayer) MaxThinkTime() time.Duration {
	return cpu.maxThinkTime
}

const (
	cpuSuggestMoveURLPath = "/hexz/cpu/suggest"
)

func (cpu *RemoteCPUPlayer) SuggestMove(ctx context.Context, spge SinglePlayerGameEngine) (ControlEvent, error) {
	// For now, only flagz engines are supported.
	ge, ok := spge.(*GameEngineFlagz)
	if !ok {
		return nil, fmt.Errorf("remote CPU player only supports flagz engines")
	}
	data, err := json.Marshal(&SuggestMoveRequest{
		MaxThinkTime: cpu.maxThinkTime,
		GameEngine:   ge,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cpu.url+cpuSuggestMoveURLPath, bytes.NewReader(data))
	if err != nil {
		return nil, err
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
	MaxThinkTime time.Duration    `json:"maxThinkTime"`
	GameEngine   *GameEngineFlagz `json:"gameEngine"`
}

type SuggestMoveResponse struct {
	Move  *MoveRequest `json:"move"`
	Stats *MCTSStats   `json:"stats"`
}

type CPUPlayerServer struct {
	randSourcePool sync.Pool
	config         *CPUPlayerServerConfig
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
		randSourcePool: sync.Pool{
			New: func() any {
				return rand.NewSource(time.Now().UnixMicro())
			},
		},
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
	if req.GameEngine == nil {
		http.Error(w, "missing game engine", http.StatusBadRequest)
		return
	}
	src := s.randSourcePool.Get().(rand.Source)
	req.GameEngine.SetSource(src)
	mcts := NewMCTS()
	thinkTime := req.MaxThinkTime
	if s.config.CpuThinkTime > 0 && thinkTime > s.config.CpuThinkTime {
		thinkTime = s.config.CpuThinkTime
	}
	mv, stats := mcts.SuggestMove(req.GameEngine, thinkTime)
	s.randSourcePool.Put(src)
	src = nil // Don't use anymore

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

func (s *CPUPlayerServer) createMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc(cpuSuggestMoveURLPath, s.handleSuggestMove)
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
