package hexz

import (
	"context"
	"math/rand"
	"time"
)

type CPUPlayer struct {
	playerId  PlayerId
	mcts      *MCTS
	thinkTime time.Duration // Current think time (auto-adjusted based on confidence)
	src       rand.Source
}

func NewCPUPlayer(playerId PlayerId) *CPUPlayer {
	return &CPUPlayer{
		playerId:  playerId,
		mcts:      NewMCTS(),
		thinkTime: time.Hour, // Will be adjusted by calls to SuggestMove.
	}
}

// Calculates a suggested move (using MCTS) and sends a ControlEventMove to respCh.
// This method should be called in a separate goroutine.
func (cpu *CPUPlayer) SuggestMove(ctx context.Context, respCh chan<- ControlEvent, ge SinglePlayerGameEngine, maxThinkTime time.Duration) bool {
	// Use our own source of randomness. In the long term, this method will deal
	// with a deserialized game engine (sent via RPC), so the source will be nil.
	ge.SetSource(cpu.src)
	if cpu.thinkTime > maxThinkTime {
		cpu.thinkTime = maxThinkTime
	}
	mv, stats := cpu.mcts.SuggestMove(ge, cpu.thinkTime)
	if minQ := stats.MinQ(); minQ >= 0.98 || minQ <= 0.02 {
		// Speed up if we think we (almost) won or lost, but stop at 0.1% of maxThinkTime.
		if cpu.thinkTime > maxThinkTime/500 {
			cpu.thinkTime = cpu.thinkTime / 2
		}
	} else {
		cpu.thinkTime = maxThinkTime // use full time allowed.
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
