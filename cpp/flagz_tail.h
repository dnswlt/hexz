#pragma once

#include <cstdint>
#include <optional>

#include "board.h"

namespace hexz {

enum class TailResolveStatus {
  // The separated-tail test is only valid once flag placement is finished.
  kFlagsRemaining,
  // Conservative reachability intersects, so the players may still interact.
  kTerritoriesOverlap,
  // At least one exact single-player solve exceeded its configured budget.
  kSolveLimit,
  // Both players' maximum independently attainable scores were computed.
  kSolved,
};

// Outcome and cost of one separated-tail attempt.
struct TailResolution {
  TailResolveStatus status = TailResolveStatus::kFlagsRemaining;
  int optimal_score[2] = {0, 0};
  // A score-maximizing legal move for each player at the input position.
  // This is absent when that player has no move or the tail was not solved.
  std::optional<Move> optimal_move[2];
  int solve_states = 0;
  int64_t solve_micros = 0;
  bool reachability_fallback = false;

  int ScoreMargin() const noexcept {
    return optimal_score[0] - optimal_score[1];
  }
  float Result() const noexcept {
    return ScoreMargin() > 0 ? 1.0f : (ScoreMargin() < 0 ? -1.0f : 0.0f);
  }
};

// Attempts to solve a Flagz position after both players have used all flags.
//
// The resolver first computes an over-approximation of every cell each player
// could still reach. If those sets are disjoint, neither player's move choices
// can change the other player's remaining territory. The two tails can then be
// solved independently as exact single-player score-maximization problems.
// These maxima determine the winner even though Flagz may terminate early
// when the player to move is already behind and has no move.
//
// kSolved is returned only if both exact solves finish. Every other status is
// deliberately conservative: the caller must continue normal MCTS play.
// max_states and max_micros apply separately to each player's exact solve.
TailResolution ResolveSeparatedTail(const Board& board, int max_states,
                                    int64_t max_micros);

const char* TailResolveStatusName(TailResolveStatus status);

}  // namespace hexz
