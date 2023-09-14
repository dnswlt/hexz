package hexz

import (
	"context"
	"math/rand"
	"time"
)

type CPUPlayer interface {
	MakeMove(ctx context.Context, respCh chan<- ControlEvent, ge SinglePlayerGameEngine) bool
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

// Calculates a suggested move (using MCTS) and sends a ControlEventMove to respCh.
// This method should be called in a separate goroutine.
// The SinglePlayerGameEngine passed in will not be modified.
func (cpu *LocalCPUPlayer) MakeMove(ctx context.Context, respCh chan<- ControlEvent, ge SinglePlayerGameEngine) bool {
	// Use our own source of randomness. In the long term, this method will deal
	// with a deserialized game engine (sent via RPC), so the source will be nil.
	timeLeft := cpu.maxThinkTime
	if d, ok := ctx.Deadline(); ok {
		t := time.Until(d)
		if t < timeLeft {
			timeLeft = t
		}
	}
	ge = ge.Clone(cpu.src)
	t := cpu.thinkTime
	if t > timeLeft {
		t = timeLeft
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
	// Send move request
	select {
	case respCh <- ControlEventMove{
		playerId:  cpu.playerId,
		mctsStats: stats,
		moveRequest: &MoveRequest{
			Move: mv.Move,
			Row:  mv.Row,
			Col:  mv.Col,
			Type: mv.CellType,
		},
	}:
		return true
	case <-ctx.Done():
		return false
	}
}
