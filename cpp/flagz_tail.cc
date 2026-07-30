#include "flagz_tail.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hexz {
namespace {

constexpr int kNumCells = 105;
constexpr int kMaxValue = 5;
constexpr int kMaxGrassCells = 5;
// Causal-path reachability is normally small late in a game. This independent
// guard prevents an adversarial position from spending unbounded time proving
// separation; exceeding it switches to a cheaper, looser over-approximation.
constexpr int kReachabilityStateLimit = 100'000;

// Rows alternate between 10 and 9 valid columns. Removing the five missing
// cells from the usual row-major index packs the board into [0, 105).
constexpr int FlatIndex(int r, int c) { return 10 * r - r / 2 + c; }

// A fixed-size bit set is cheap to copy. Tail search copies masks on every
// branch, so keeping this at two machine words matters.
struct CellMask {
  std::array<uint64_t, 2> words{};

  bool Test(int cell) const noexcept {
    return words[cell / 64] & (uint64_t{1} << (cell % 64));
  }
  void Set(int cell) noexcept {
    words[cell / 64] |= uint64_t{1} << (cell % 64);
  }
  void Clear(int cell) noexcept {
    words[cell / 64] &= ~(uint64_t{1} << (cell % 64));
  }
  bool Intersects(const CellMask& other) const noexcept {
    return (words[0] & other.words[0]) || (words[1] & other.words[1]);
  }
  bool IsSubsetOf(const CellMask& other) const noexcept {
    return (words[0] & other.words[0]) == words[0] &&
           (words[1] & other.words[1]) == words[1];
  }
  bool operator==(const CellMask&) const = default;
};

struct FlatNeighbors {
  std::array<uint8_t, 6> cells{};
  uint8_t size = 0;
};

const std::array<FlatNeighbors, kNumCells>& Neighbors() {
  // Board geometry never changes. Build this once, then use flat integer
  // neighbors throughout the hot reachability and search loops.
  static const auto* neighbors = [] {
    auto* result = new std::array<FlatNeighbors, kNumCells>;
    for (int r = 0; r < 11; ++r) {
      for (int c = 0; c < 10 - r % 2; ++c) {
        auto& flat = (*result)[FlatIndex(r, c)];
        for (const auto& neighbor : internal::NeighborsOf({r, c})) {
          flat.cells[flat.size++] = FlatIndex(neighbor.r, neighbor.c);
        }
      }
    }
    return result;
  }();
  return *neighbors;
}

struct PlayerPosition {
  // blocked and next are player-specific. Grass is copied into both views
  // because either player may capture it until separation is proven.
  CellMask blocked;
  std::array<uint8_t, kNumCells> next{};
  std::array<uint8_t, kNumCells> grass{};
};

struct Position {
  PlayerPosition player[2];
  int score[2] = {0, 0};
};

Position CopyPosition(const Board& board) {
  // Tensor() is taken from player 0's view, whose channel layout is the
  // canonical P0/P1 ordering documented by Board. This snapshot also keeps
  // torch access out of the recursive solver.
  Position result;
  auto tensor = board.Tensor(0);
  auto values = tensor.accessor<float, 3>();
  for (int r = 0; r < 11; ++r) {
    for (int c = 0; c < 10 - r % 2; ++c) {
      const int cell = FlatIndex(r, c);
      const auto grass = static_cast<uint8_t>(values[10][r][c]);
      for (int player = 0; player < 2; ++player) {
        const int offset = 5 * player;
        if (values[2 + offset][r][c] != 0) {
          result.player[player].blocked.Set(cell);
        }
        result.player[player].next[cell] =
            static_cast<uint8_t>(values[3 + offset][r][c]);
        result.player[player].grass[cell] = grass;
        result.score[player] += static_cast<int>(values[1 + offset][r][c]);
      }
    }
  }
  return result;
}

CellMask RelaxedReachableCells(const PlayerPosition& position) {
  // Propagate (cell, value) possibilities while forgetting that a cell can be
  // occupied only once. This may invent cyclic paths, hence extra reachable
  // cells, but can never omit a legal path. It is safe for rejecting
  // separation and serves as the bounded fallback below.
  std::array<uint8_t, kNumCells> seen_values{};
  std::vector<std::pair<uint8_t, uint8_t>> pending;
  pending.reserve(kNumCells * kMaxValue);
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (position.next[cell]) {
      pending.emplace_back(cell, position.next[cell]);
    }
  }

  CellMask reachable;
  while (!pending.empty()) {
    const auto [cell, value] = pending.back();
    pending.pop_back();
    const uint8_t value_bit = uint8_t{1} << value;
    if (seen_values[cell] & value_bit) {
      continue;
    }
    seen_values[cell] |= value_bit;
    reachable.Set(cell);

    const auto& neighbors = Neighbors()[cell];
    for (int i = 0; i < neighbors.size; ++i) {
      const int neighbor = neighbors.cells[i];
      const uint8_t grass = position.grass[neighbor];
      if (grass && grass <= value) {
        pending.emplace_back(neighbor, grass);
      }
      if (value >= kMaxValue || position.blocked.Test(neighbor)) {
        continue;
      }
      uint8_t propagated = value + 1;
      if (position.next[neighbor]) {
        propagated = std::min(propagated, position.next[neighbor]);
      }
      pending.emplace_back(neighbor, propagated);
    }
  }
  return reachable;
}

struct ReachabilityResult {
  CellMask cells;
  bool used_relaxed_fallback = false;
};

struct PathState {
  // Cells consumed on this causal path. A future legal move always has such a
  // simple path from one of the currently playable frontier cells.
  CellMask used;
  uint8_t cell;
  uint8_t value;
};

ReachabilityResult OptimisticReachability(const PlayerPosition& position) {
  // For every (cell, value), retain only non-dominated used-cell masks. If path
  // A used a subset of path B, every continuation available to B is also
  // available to A, so B can be discarded without losing reachability.
  std::array<std::vector<CellMask>, kNumCells * (kMaxValue + 1)> masks;
  std::vector<PathState> pending;
  pending.reserve(1024);
  int path_states = 0;

  auto add = [&](int cell, int value, const CellMask& used) {
    auto& candidates = masks[cell * (kMaxValue + 1) + value];
    if (std::any_of(candidates.begin(), candidates.end(),
                    [&](const CellMask& existing) {
                      return existing.IsSubsetOf(used);
                    })) {
      return true;
    }
    std::erase_if(candidates, [&](const CellMask& existing) {
      return used.IsSubsetOf(existing);
    });
    candidates.push_back(used);
    pending.push_back(
        PathState{used, static_cast<uint8_t>(cell),
                  static_cast<uint8_t>(value)});
    // Falling back only enlarges the result, preserving the no-false-
    // separation property.
    return ++path_states <= kReachabilityStateLimit;
  };

  for (int cell = 0; cell < kNumCells; ++cell) {
    if (!position.next[cell]) {
      continue;
    }
    CellMask used;
    used.Set(cell);
    if (!add(cell, position.next[cell], used)) {
      return {RelaxedReachableCells(position), true};
    }
  }

  CellMask reachable;
  while (!pending.empty()) {
    const PathState current = pending.back();
    pending.pop_back();
    reachable.Set(current.cell);
    const auto& neighbors = Neighbors()[current.cell];
    for (int i = 0; i < neighbors.size; ++i) {
      const int neighbor = neighbors.cells[i];
      if (current.used.Test(neighbor)) {
        continue;
      }
      CellMask used = current.used;
      used.Set(neighbor);

      const uint8_t grass = position.grass[neighbor];
      if (grass && grass <= current.value &&
          !add(neighbor, grass, used)) {
        return {RelaxedReachableCells(position), true};
      }
      if (current.value >= kMaxValue ||
          position.blocked.Test(neighbor)) {
        continue;
      }
      uint8_t propagated = current.value + 1;
      if (position.next[neighbor]) {
        propagated = std::min(propagated, position.next[neighbor]);
      }
      if (!add(neighbor, propagated, used)) {
        return {RelaxedReachableCells(position), true};
      }
    }
  }
  return {reachable, false};
}

std::vector<CellMask> ReachableComponents(const CellMask& reachable) {
  // Disconnected reachable regions cannot create next-values in one another.
  // Their optimum scores can therefore be solved independently and added.
  CellMask visited;
  std::vector<CellMask> result;
  std::array<uint8_t, kNumCells> pending{};
  for (int seed = 0; seed < kNumCells; ++seed) {
    if (!reachable.Test(seed) || visited.Test(seed)) {
      continue;
    }
    CellMask component;
    int pending_size = 0;
    pending[pending_size++] = seed;
    visited.Set(seed);
    while (pending_size) {
      const int cell = pending[--pending_size];
      component.Set(cell);
      const auto& neighbors = Neighbors()[cell];
      for (int i = 0; i < neighbors.size; ++i) {
        const int neighbor = neighbors.cells[i];
        if (reachable.Test(neighbor) && !visited.Test(neighbor)) {
          visited.Set(neighbor);
          pending[pending_size++] = neighbor;
        }
      }
    }
    result.push_back(component);
  }
  return result;
}

struct SolverState {
  // blocked and next fully describe future normal moves for one player.
  // Official boards contain five grass cells, so remaining/captured status
  // fits in five bits; the grass values themselves are immutable.
  CellMask blocked;
  std::array<uint8_t, kNumCells> next{};
  uint8_t remaining_grass = 0;

  bool operator==(const SolverState&) const = default;
};

struct SolverStateHash {
  // FNV-1a over the compact state. Memoization saves substantial work because
  // many different move orders transpose into the same state.
  size_t operator()(const SolverState& state) const noexcept {
    size_t hash = 1469598103934665603ULL;
    auto mix = [&](uint8_t value) {
      hash ^= value;
      hash *= 1099511628211ULL;
    };
    for (uint64_t word : state.blocked.words) {
      for (int shift = 0; shift < 64; shift += 8) {
        mix(static_cast<uint8_t>(word >> shift));
      }
    }
    for (uint8_t value : state.next) {
      mix(value);
    }
    mix(state.remaining_grass);
    return hash;
  }
};

struct CacheEntry {
  int best = -1;
  int upper_bound = -1;
};

struct ExactResult {
  int additional_score = 0;
  int first_cell = -1;
};

class ExactSolver {
 public:
  ExactSolver(const PlayerPosition& position, int max_states,
              int64_t max_micros)
      : max_states_(max_states), max_micros_(max_micros) {
    grass_index_.fill(-1);
    initial_.blocked = position.blocked;
    initial_.next = position.next;
    for (int cell = 0; cell < kNumCells; ++cell) {
      if (!position.grass[cell]) {
        continue;
      }
      // A valid game starts with exactly five grass cells and may only lose
      // them. Fail conservatively on a synthetic board violating that rule.
      if (grass_count_ == kMaxGrassCells) {
        valid_ = false;
        continue;
      }
      const int index = grass_count_++;
      grass_index_[cell] = index;
      grass_value_[index] = position.grass[cell];
      initial_.remaining_grass |= uint8_t{1} << index;
    }
    cache_.reserve(max_states);
  }

  std::optional<ExactResult> Solve(const CellMask& reachable) {
    if (!valid_) {
      return std::nullopt;
    }
    started_ = std::chrono::steady_clock::now();
    deadline_ = started_ + std::chrono::microseconds(max_micros_);
    ExactResult result;
    for (const CellMask& component : ReachableComponents(reachable)) {
      SolverState state = RestrictToComponent(initial_, component);
      int first_cell = -1;
      const auto score = SolveState(state, &first_cell);
      if (!score.has_value()) {
        elapsed_micros_ = ElapsedMicros();
        return std::nullopt;
      }
      result.additional_score += *score;
      // Components are independent, so an optimal first move in any nonempty
      // component preserves the sum of their optimal scores.
      if (result.first_cell < 0 && first_cell >= 0) {
        result.first_cell = first_cell;
      }
    }
    elapsed_micros_ = ElapsedMicros();
    return result;
  }

  int states() const noexcept { return states_; }
  int64_t elapsed_micros() const noexcept { return elapsed_micros_; }

 private:
  SolverState RestrictToComponent(SolverState state,
                                  const CellMask& component) const {
    // Everything outside this component becomes permanently unavailable and
    // its frontier/grass metadata is dropped.
    for (int cell = 0; cell < kNumCells; ++cell) {
      if (component.Test(cell)) {
        continue;
      }
      state.blocked.Set(cell);
      state.next[cell] = 0;
      const int grass_index = grass_index_[cell];
      if (grass_index >= 0) {
        state.remaining_grass &=
            ~(uint8_t{1} << static_cast<uint8_t>(grass_index));
      }
    }
    return state;
  }

  int64_t ElapsedMicros() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - started_)
        .count();
  }

  bool BudgetAvailable() const {
    if (states_ >= max_states_) {
      return false;
    }
    // A clock read at every node is costly. The hard state cap is checked on
    // every node; wall time is sampled once per 256 newly visited states.
    return states_ % 256 != 0 ||
           std::chrono::steady_clock::now() <= deadline_;
  }

  uint8_t GrassValue(const SolverState& state, int cell) const {
    const int index = grass_index_[cell];
    if (index < 0 ||
        !(state.remaining_grass &
          (uint8_t{1} << static_cast<uint8_t>(index)))) {
      return 0;
    }
    return grass_value_[index];
  }

  int ApplyNormal(SolverState& state, int cell) const {
    // Mirrors Board::MakeMove for a normal move: score the cell, propagate the
    // next value (or block neighbors after a 5), then recursively auto-capture
    // eligible adjacent grass. Cross-player occupancy is absent because this
    // runs only after the conservative separation proof.
    const uint8_t value = state.next[cell];
    state.blocked.Set(cell);
    state.next[cell] = 0;
    const auto& neighbors = Neighbors()[cell];
    if (value < kMaxValue) {
      const uint8_t propagated = value + 1;
      for (int i = 0; i < neighbors.size; ++i) {
        const int neighbor = neighbors.cells[i];
        if (state.blocked.Test(neighbor)) {
          continue;
        }
        if (!state.next[neighbor] ||
            state.next[neighbor] > propagated) {
          state.next[neighbor] = propagated;
        }
      }
    } else {
      for (int i = 0; i < neighbors.size; ++i) {
        const int neighbor = neighbors.cells[i];
        state.blocked.Set(neighbor);
        state.next[neighbor] = 0;
      }
    }

    int gained = value;
    for (int i = 0; i < neighbors.size; ++i) {
      const int neighbor = neighbors.cells[i];
      const uint8_t grass = GrassValue(state, neighbor);
      if (!grass || grass > value) {
        continue;
      }
      const int grass_index = grass_index_[neighbor];
      state.remaining_grass &=
          ~(uint8_t{1} << static_cast<uint8_t>(grass_index));
      state.blocked.Clear(neighbor);
      state.next[neighbor] = grass;
      gained += ApplyNormal(state, neighbor);
    }
    return gained;
  }

  int UpperBound(const SolverState& state) {
    // Forget one-use path constraints and find the largest value with which
    // each cell might be reached. The sum can overestimate the remaining score
    // but cannot underestimate it, making it safe for branch-and-bound.
    auto entry = cache_.try_emplace(state).first;
    if (entry->second.upper_bound >= 0) {
      return entry->second.upper_bound;
    }

    std::array<uint8_t, kNumCells> seen_values{};
    std::vector<std::pair<uint8_t, uint8_t>> pending;
    pending.reserve(kNumCells * kMaxValue);
    for (int cell = 0; cell < kNumCells; ++cell) {
      if (state.next[cell]) {
        pending.emplace_back(cell, state.next[cell]);
      }
    }
    while (!pending.empty()) {
      const auto [cell, value] = pending.back();
      pending.pop_back();
      const uint8_t value_bit = uint8_t{1} << value;
      if (seen_values[cell] & value_bit) {
        continue;
      }
      seen_values[cell] |= value_bit;
      const auto& neighbors = Neighbors()[cell];
      for (int i = 0; i < neighbors.size; ++i) {
        const int neighbor = neighbors.cells[i];
        const uint8_t grass = GrassValue(state, neighbor);
        if (grass && grass <= value) {
          pending.emplace_back(neighbor, grass);
        }
        if (value >= kMaxValue || state.blocked.Test(neighbor)) {
          continue;
        }
        uint8_t propagated = value + 1;
        if (state.next[neighbor]) {
          propagated = std::min(propagated, state.next[neighbor]);
        }
        pending.emplace_back(neighbor, propagated);
      }
    }

    int upper_bound = 0;
    for (uint8_t values : seen_values) {
      for (int value = kMaxValue; value > 0; --value) {
        if (values & (uint8_t{1} << value)) {
          upper_bound += value;
          break;
        }
      }
    }
    // Recursive hash-table insertions may have rehashed and invalidated the
    // iterator obtained above.
    entry = cache_.find(state);
    entry->second.upper_bound = upper_bound;
    return upper_bound;
  }

  std::optional<int> SolveState(const SolverState& state,
                                int* best_first_cell = nullptr) {
    // Exact depth-first dynamic programming. Different move orders frequently
    // lead to the same state, while the relaxed bound eliminates branches that
    // cannot beat the best score already found.
    const auto cached = cache_.find(state);
    if (cached != cache_.end() && cached->second.best >= 0) {
      return cached->second.best;
    }
    if (!BudgetAvailable()) {
      return std::nullopt;
    }
    ++states_;

    struct Child {
      SolverState state;
      int gained;
      int cell;
    };
    std::vector<Child> children;
    for (int cell = 0; cell < kNumCells; ++cell) {
      if (!state.next[cell]) {
        continue;
      }
      SolverState child = state;
      const int gained = ApplyNormal(child, cell);
      children.push_back({std::move(child), gained, cell});
    }
    if (children.empty()) {
      cache_[state].best = 0;
      return 0;
    }
    std::sort(children.begin(), children.end(),
              [](const Child& lhs, const Child& rhs) {
                return std::pair(lhs.gained, lhs.cell) >
                       std::pair(rhs.gained, rhs.cell);
              });

    // High immediate gains first usually produce a strong incumbent early and
    // improve pruning without changing correctness.
    int best = 0;
    const int state_upper_bound = UpperBound(state);
    for (const Child& child : children) {
      if (child.gained + UpperBound(child.state) <= best) {
        continue;
      }
      const auto remaining = SolveState(child.state);
      if (!remaining.has_value()) {
        return std::nullopt;
      }
      const int candidate = child.gained + *remaining;
      if (candidate > best) {
        best = candidate;
        if (best_first_cell != nullptr) {
          *best_first_cell = child.cell;
        }
      }
      if (best >= state_upper_bound) {
        break;
      }
    }
    cache_[state].best = best;
    return best;
  }

  SolverState initial_;
  std::array<int8_t, kNumCells> grass_index_{};
  std::array<uint8_t, kMaxGrassCells> grass_value_{};
  int grass_count_ = 0;
  bool valid_ = true;
  int max_states_;
  int64_t max_micros_;
  int states_ = 0;
  int64_t elapsed_micros_ = 0;
  std::chrono::steady_clock::time_point started_;
  std::chrono::steady_clock::time_point deadline_;
  std::unordered_map<SolverState, CacheEntry, SolverStateHash> cache_;
};

}  // namespace

TailResolution ResolveSeparatedTail(const Board& board, int max_states,
                                    int64_t max_micros) {
  // Pipeline:
  //   1. reject positions where future flags can open arbitrary territory;
  //   2. conservatively prove that future territories do not overlap;
  //   3. maximize each player's remaining score independently and exactly.
  // Any uncertainty returns a non-solved status and leaves MCTS in charge.
  TailResolution result;
  if (board.Flags(0) || board.Flags(1)) {
    return result;
  }
  if (max_states <= 0 || max_micros <= 0) {
    result.status = TailResolveStatus::kSolveLimit;
    return result;
  }

  const Position position = CopyPosition(board);
  const ReachabilityResult reachable[2] = {
      OptimisticReachability(position.player[0]),
      OptimisticReachability(position.player[1]),
  };
  result.reachability_fallback =
      reachable[0].used_relaxed_fallback ||
      reachable[1].used_relaxed_fallback;
  if (reachable[0].cells.Intersects(reachable[1].cells)) {
    result.status = TailResolveStatus::kTerritoriesOverlap;
    return result;
  }

  for (int player = 0; player < 2; ++player) {
    ExactSolver solver(position.player[player], max_states, max_micros);
    const auto additional_score = solver.Solve(reachable[player].cells);
    result.solve_states += solver.states();
    result.solve_micros += solver.elapsed_micros();
    if (!additional_score.has_value()) {
      result.status = TailResolveStatus::kSolveLimit;
      return result;
    }
    result.optimal_score[player] =
        position.score[player] + additional_score->additional_score;
    if (additional_score->first_cell >= 0) {
      const int cell = additional_score->first_cell;
      int row = 0;
      while (row < 10 && FlatIndex(row + 1, 0) <= cell) {
        ++row;
      }
      const int col = cell - FlatIndex(row, 0);
      result.optimal_move[player] =
          Move{Move::Typ::kNormal, row, col,
               static_cast<float>(position.player[player].next[cell])};
    }
  }
  result.status = TailResolveStatus::kSolved;
  return result;
}

const char* TailResolveStatusName(TailResolveStatus status) {
  switch (status) {
    case TailResolveStatus::kFlagsRemaining:
      return "flags_remaining";
    case TailResolveStatus::kTerritoriesOverlap:
      return "territories_overlap";
    case TailResolveStatus::kSolveLimit:
      return "solve_limit";
    case TailResolveStatus::kSolved:
      return "solved";
  }
  return "unknown";
}

}  // namespace hexz
