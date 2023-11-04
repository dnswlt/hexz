#include "board.h"

#include <absl/log/absl_check.h>
#include <torch/torch.h>

#include <cstdlib>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>

#include "base.h"
#include "perfm.h"

namespace hexz {

using internal::Idx;

namespace internal {
// InitializeNeighborsMap returns a map that yields the valid neighbor cell
// indices for all valid indices of a hexz board.
std::unordered_map<Idx, std::vector<Idx>>* InitializeNeighborsMap() {
  auto* map = new std::unordered_map<Idx, std::vector<Idx>>();
  for (int r = 0; r < 11; r++) {
    int s = r % 2;
    for (int c = 0; c < 10 - s; c++) {
      Idx k{r, c};
      std::vector<Idx> ns{
          Idx{r, c + 1}, Idx{r - 1, c + s},     Idx{r - 1, c - 1 + s},
          Idx{r, c - 1}, Idx{r + 1, c - 1 + s}, Idx{r + 1, c + s},
      };
      for (const auto& n : ns) {
        if (n.IsValid()) {
          (*map)[k].push_back(n);
        }
      }
    }
  }
  return map;
}

// NeighborsOf returns the valid neighbor indices of k.
const std::vector<Idx>& NeighborsOf(const Idx& k) {
  static const std::unordered_map<Idx, std::vector<Idx>>* map =
      InitializeNeighborsMap();
  return (*map).at(k);
}

}  // namespace internal

Board::Board() : nflags_{0, 0} {
  b_ = torch::zeros({9, 11, 10}, torch::dtype(torch::kFloat32));
}

// Copy c'tor.
Board::Board(const Board& other) {
  b_ = other.b_.clone();
  nflags_[0] = other.nflags_[0];
  nflags_[1] = other.nflags_[1];
}

// Torch representation of a hexz board.
// A board is represented by an (9, 11, 10) tensor. Each 11x10 channel is
// a one-hot encoding of the presence of specific type of piece/obstacle/etc.
// The channels are:
//
// * 0: flags by P0
// * 1: cell value 1-5 for P0
// * 2: cells blocked for P0 (any occupied cell or a cell next to a 5)
// * 3: next value for P0
// * 4: flags by P1
// * 5: cell value 1-5 for P1
// * 6: cells blocked for P1
// * 7: next value for P1
// * 8: grass cells with value 1-5
//
// An action is specified by a (2, 11, 10) numpy array. The first 11x10 channel
// represents a flag move, the second one represents a regular cell move. A
// flag move must have a single 1 set, a normal move must have a single value
// 1-5 set.
Board Board::RandomBoard() {
  Board b;
  b.nflags_[0] = 3;
  b.nflags_[1] = 3;
  // Even rows have 10 cells, odd rows only 9, so mark the last cell in odd
  // rows as blocked for P1+P2.
  for (int i = 1; i <= 9; i += 2) {
    b.b_.index_put_({2, i, 9}, 1);
    b.b_.index_put_({6, i, 9}, 1);
  }
  // 15 randomly placed stones.
  std::uniform_int_distribution<> rnd_row(0, 10);
  std::uniform_int_distribution<> rnd_col(0, 9);
  for (int n_stones = 0; n_stones < 15;) {
    int r = rnd_row(internal::rng);
    int c = rnd_col(internal::rng);
    if (b.b_.index({2, r, c}).item<float>() != 0) {
      continue;  // Already occupied.
    }
    b.b_.index_put_({2, r, c}, 1);
    b.b_.index_put_({6, r, c}, 1);
    n_stones++;
  }
  // 5 randomly placed grass cells.
  for (int n_grass = 0; n_grass < 5;) {
    int r = rnd_row(internal::rng);
    int c = rnd_col(internal::rng);
    if (b.b_.index({2, r, c}).item<float>() != 0) {
      continue;  // Already occupied.
    }
    b.b_.index_put_({8, r, c}, n_grass + 1);
    b.b_.index_put_({2, r, c}, 1);
    b.b_.index_put_({6, r, c}, 1);
    n_grass++;
  }

  return b;
}

torch::Tensor Board::Tensor(int player) const {
  if (player == 0) {
    return b_;
  }
  // Swap channels for player 0 and player 1. Leave grass unchanged.
  // index_select returns a new Tensor that uses its own storage.
  return b_.index_select(0, torch::tensor({4, 5, 6, 7, 0, 1, 2, 3, 8}));
}

std::pair<float, float> Board::Score() const {
  return {b_.index({1}).sum().item<float>(), b_.index({5}).sum().item<float>()};
}

float Board::Result() const {
  auto score = Score();
  if (score.first > score.second) {
    return 1.0;
  }
  if (score.first < score.second) {
    return -1.0;
  }
  return 0.0;
}

int Board::Flags(int player) const { return nflags_[player]; }

void Board::MakeMove(int player, const Move& move) {
  Perfm::Scope ps(Perfm::MakeMove);
  auto b_acc = b_.accessor<float, 3>();
  b_acc[move.typ + player * 4][move.r][move.c] = move.value;
  bool played_flag = move.typ == 0;
  // Occupy cell for both players
  b_acc[2][move.r][move.c] = 1;
  b_acc[6][move.r][move.c] = 1;
  // Zero next value.
  b_acc[3][move.r][move.c] = 0;
  b_acc[7][move.r][move.c] = 0;
  float next_val = 1;
  if (played_flag) {
    ABSL_DCHECK(nflags_[player] > 0)
        << "Move " << move.DebugString() << " by " << player
        << " without flags left: " << DebugString();
    nflags_[player]--;
  } else {
    next_val = move.value + 1;
  }
  for (const auto& nb : NeighborsOf(Idx{move.r, move.c})) {
    if (next_val <= 5) {
      if (b_acc[2 + player * 4][nb.r][nb.c] == 0) {
        // Neighbor cell is not blocked.
        if (b_acc[3 + player * 4][nb.r][nb.c] == 0) {
          // Neighbor cell did not have a next value yet.
          b_acc[3 + player * 4][nb.r][nb.c] = next_val;
        } else if (b_acc[3 + player * 4][nb.r][nb.c] > next_val) {
          // Neighbor cell's value was larger: decrease.
          b_acc[3 + player * 4][nb.r][nb.c] = next_val;
        }
      }
    } else {
      // Played a 5: block neighboring cells and clear next value.
      b_acc[2 + player * 4][nb.r][nb.c] = 1;
      b_acc[3 + player * 4][nb.r][nb.c] = 0;
    }
  }
  if (!played_flag) {
    // Occupy grass fields
    for (const auto& nb : NeighborsOf(Idx{move.r, move.c})) {
      float grass_val = b_acc[8][nb.r][nb.c];
      if (grass_val > 0 && grass_val <= b_acc[1 + player * 4][move.r][move.c]) {
        // Occupy grass cell: remove grass value and add it to player's value.
        b_acc[8][nb.r][nb.c] = 0;
        MakeMove(player, Move{1, nb.r, nb.c, grass_val});
      }
    }
  }
}

std::vector<Move> Board::NextMoves(int player) const {
  Perfm::Scope ps(Perfm::NextMoves);
  std::vector<Move> moves;
  auto b_acc = b_.accessor<float, 3>();
  bool flag = nflags_[player] > 0;
  // Flag in any unoccupied cell.
  for (int r = 0; r < 11; r++) {
    int s = r % 2;
    for (int c = 0; c < 10 - s; c++) {
      if (flag && b_acc[2 + player * 4][r][c] == 0) {
        moves.push_back(Move{0, r, c, 0.0});
      }
      float next_val = b_acc[3 + player * 4][r][c];
      if (next_val > 0) {
        moves.push_back(Move{1, r, c, next_val});
      }
    }
  }
  return moves;
}

std::string Board::DebugString() const {
  std::ostringstream os;
  os << "Board(" <<                                            //
      "flags:(" << nflags_[0] << ", " << nflags_[1] << ")" <<  //
      " score:" << Score() <<                                  //
      ")";
  return os.str();
}

}  // namespace hexz
