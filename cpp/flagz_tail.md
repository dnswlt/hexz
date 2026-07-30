# Flagz separated-tail solver

`flagz_tail.cc` ends self-play games once their winner is known and the
remaining moves only optimize already separated territory. It is deliberately
one-sided: uncertain or expensive positions fall back to ordinary MCTS.

## Safety condition

The solver runs only after both players have placed all flags. For each player,
it computes a conservative superset of cells that could still become playable.
If the two sets intersect, play continues normally.

When the sets are disjoint, a move by one player cannot occupy a cell, alter a
frontier value, or capture a grass cell that the other player can reach.
Consequently the two remaining score-maximization problems are independent.
The game result is the sign of:

```text
(P0 score now + P0 maximum remaining score)
-
(P1 score now + P1 maximum remaining score)
```

These are maximum *attainable* scores, not necessarily the scores printed by a
normally completed game. Flagz ends immediately when the player to move has no
move and is already behind. This early termination has the same winner after
separation: a stuck player's attainable score is fixed, and the other player
can overtake it if and only if that player's independent maximum is higher.
Equality remains a draw.

This argument depends on reachability being an over-approximation: false
overlap is harmless and merely misses an optimization, while false separation
would be incorrect.

## Conservative reachability

A future playable cell has a causal path from a cell that is playable now.
Values propagate from 1 through 5, a 5 stops propagation, and a capturable
grass cell restarts propagation at its grass value.

`OptimisticReachability` explores these paths while recording the cells already
used by each path. Reusing a cell is forbidden, which prevents artificial
cycles from extending a sequence forever. For a fixed `(cell, value)`, a path
whose used-cell set is a subset of another path dominates it and the larger
path can be discarded.

The number of non-dominated causal paths is capped at 100,000. On reaching that
cap, the code switches to `RelaxedReachableCells`, which forgets the one-use
constraint. That fallback can add impossible cells but cannot remove possible
ones, so it remains safe.

## Exact score search

After separation, each player is solved independently:

1. Split the reachable set into disconnected components.
2. Solve each component with memoized depth-first search.
3. Apply normal moves with the same value propagation, blocking-after-5, and
   automatic grass-capture behavior as `Board::MakeMove`.
4. Use relaxed value reachability as an admissible upper score bound for
   branch-and-bound pruning.
5. Add the independent component optima to the player's current score.

The hot state consists of a 105-bit blocked mask, 105 frontier values, and five
bits for the remaining grass cells. Board geometry and neighbor indexes are
precomputed once. No Torch operations occur inside the recursive search.

The state and wall-clock limits apply separately to each player. The state
limit is checked at every new state; the clock is sampled every 256 states to
keep timing overhead out of small solves. Exhausting either budget returns
`kSolveLimit` and normal MCTS continues.

## Self-play integration

`NeuralMCTS::PlayGame` invokes the resolver before searching a move. A solved
tail is used only when the absolute maximum-attainable-score margin reaches
`tail_solver_min_score_margin`, currently 5. The margin is an operational
caution gate, not part of the separation proof.

The controls are:

- `tail_solver_max_states` (`HEXZ_TAIL_SOLVER_MAX_STATES`)
- `tail_solver_max_micros` (`HEXZ_TAIL_SOLVER_MAX_MICROS`)
- `tail_solver_min_score_margin` (`HEXZ_TAIL_SOLVER_MIN_SCORE_MARGIN`)
- `tail_solver_shadow` (`HEXZ_TAIL_SOLVER_SHADOW`)

In shadow mode, the first qualifying result is logged but the game is played to
completion and its actual winner is compared with the predicted winner. In
active mode, the game stops immediately and all retained training examples are
labeled with the exact result. No example is produced for the omitted tail.

`flagz_tail_test.cc` covers flag gating, value propagation, blocking after 5
with grass restart, overlap rejection, component addition, and search limits.
`mcts_test.cc` also verifies that a qualifying tail terminates before MCTS
search begins.
