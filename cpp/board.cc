#include "board.h"

#include <absl/log/absl_check.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <torch/torch.h>

#include <cstdlib>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>

#include "base.h"
#include "perfm.h"

#define CHANNELS_PER_PLAYER 5

// Accessor macros for the different channels.
#define I_FLAG(p) (0 + CHANNELS_PER_PLAYER * p)
#define I_VALUE(p) (1 + CHANNELS_PER_PLAYER * p)
#define I_BLOCKED(p) (2 + CHANNELS_PER_PLAYER * p)
#define I_NEXTVAL(p) (3 + CHANNELS_PER_PLAYER * p)
#define I_NFLAGS(p) (4 + CHANNELS_PER_PLAYER * p)
#define I_GRASS 10

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

Board::Board() {
  b_ = torch::zeros({11, 11, 10}, torch::dtype(torch::kFloat32));
}

// Copy c'tor.
Board::Board(const Board& other) { b_ = other.b_.clone(); }

Board Board::RandomBoard() {
  Board b;
  b.b_.index_put_({I_NFLAGS(0)}, 3.0f);
  b.b_.index_put_({I_NFLAGS(1)}, 3.0f);
  // Even rows have 10 cells, odd rows only 9, so mark the last cell in odd
  // rows as blocked for P1+P2.
  for (int i = 1; i <= 9; i += 2) {
    b.b_.index_put_({I_BLOCKED(0), i, 9}, 1);
    b.b_.index_put_({I_BLOCKED(1), i, 9}, 1);
  }
  // 15 randomly placed stones.
  std::uniform_int_distribution<> rnd_row(0, 10);
  std::uniform_int_distribution<> rnd_col(0, 9);
  for (int n_stones = 0; n_stones < 15;) {
    int r = rnd_row(internal::rng);
    int c = rnd_col(internal::rng);
    if (b.b_.index({I_BLOCKED(0), r, c}).item<float>() != 0) {
      continue;  // Already occupied.
    }
    b.b_.index_put_({I_BLOCKED(0), r, c}, 1);
    b.b_.index_put_({I_BLOCKED(1), r, c}, 1);
    n_stones++;
  }
  // 5 randomly placed grass cells.
  for (int n_grass = 0; n_grass < 5;) {
    int r = rnd_row(internal::rng);
    int c = rnd_col(internal::rng);
    if (b.b_.index({I_BLOCKED(0), r, c}).item<float>() != 0) {
      continue;  // Already occupied.
    }
    b.b_.index_put_({I_GRASS, r, c}, n_grass + 1);
    b.b_.index_put_({I_BLOCKED(0), r, c}, 1);
    b.b_.index_put_({I_BLOCKED(1), r, c}, 1);
    n_grass++;
  }

  return b;
}

absl::StatusOr<Board> Board::FromProto(const hexzpb::Board& board) {
  const int iFlag = static_cast<int>(hexzpb::Field::FLAG);
  if (board.resources_size() != 2 ||
      board.resources(0).num_pieces_size() <= iFlag ||
      board.resources(1).num_pieces_size() <= iFlag) {
    return absl::InvalidArgumentError("Invalid resources.num_pieces shape");
  }
  Board b;
  b.b_.index_put_({I_NFLAGS(0)},
                  static_cast<float>(board.resources(0).num_pieces(iFlag)));
  b.b_.index_put_({I_NFLAGS(1)},
                  static_cast<float>(board.resources(1).num_pieces(iFlag)));
  auto a = b.b_.accessor<float, 3>();
  int r = 0;
  int c = 0;
  if (board.flat_fields_size() != 105) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expecting board with exactly 105 fields, got: ",
                     board.flat_fields_size()));
  }
  for (const auto& f : board.flat_fields()) {
    if (f.owner() < 0 || f.owner() > 2) {
      // Owner is 1-based.
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid owner: ", f.owner()));
    }
    int p = f.owner() - 1;
    switch (f.type()) {
      case hexzpb::Field::NORMAL:
        if (f.value() > 0) {
          a[I_VALUE(p)][r][c] = f.value();
          a[I_BLOCKED(0)][r][c] = 1;
          a[I_BLOCKED(1)][r][c] = 1;
          break;
        }
        if (f.next_val_size() > 0 && f.next_val(0) > 0) {
          a[I_NEXTVAL(0)][r][c] = f.next_val(0);
        }
        if (f.next_val_size() > 1 && f.next_val(1) > 0) {
          a[I_NEXTVAL(1)][r][c] = f.next_val(1);
        }
        if ((f.blocked() & 1) > 0) {
          a[I_BLOCKED(0)][r][c] = 1;
        }
        if ((f.blocked() & 2) > 0) {
          a[I_BLOCKED(1)][r][c] = 1;
        }
        break;
      case hexzpb::Field::FLAG:
        a[I_FLAG(p)][r][c] = 1;
        a[I_BLOCKED(0)][r][c] = 1;
        a[I_BLOCKED(1)][r][c] = 1;
        break;
      case hexzpb::Field::GRASS:
        a[I_GRASS][r][c] = f.value();
        a[I_BLOCKED(0)][r][c] = 1;
        a[I_BLOCKED(1)][r][c] = 1;
        break;
      case hexzpb::Field::ROCK:
        a[I_BLOCKED(0)][r][c] = 1;
        a[I_BLOCKED(1)][r][c] = 1;
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Board has invalid field of type ", f.type()));
    }
    c++;
    if (c == 10 - r % 2) {
      c = 0;
      r += 1;
    }
  }
  return b;
}

torch::Tensor Board::Tensor(int player) const {
  if (player == 0) {
    return b_;
  }
  // Swap channels for player 0 and player 1. Leave grass unchanged.
  // index_select returns a new Tensor that uses its own storage.
  return b_.index_select(0, torch::tensor({5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10}));
}

std::pair<float, float> Board::Score() const {
  return {b_.index({I_VALUE(0)}).sum().item<float>(),
          b_.index({I_VALUE(1)}).sum().item<float>()};
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

int Board::Flags(int player) const {
  // The whole NFLAGS channel has identical numbers. Just pick (0, 0).
  return b_.index({I_NFLAGS(player), 0, 0}).item<float>();
}

void Board::MakeMove(int player, const Move& move) {
  Perfm::Scope ps(Perfm::MakeMove);
  auto b_acc = b_.accessor<float, 3>();
  ABSL_DCHECK_EQ(b_acc[I_BLOCKED(player)][move.r][move.c], 0)
      << "MakeMove on blocked field";
  ABSL_DCHECK(move.typ == Move::Typ::kFlag && move.value == 1 ||
              move.typ == Move::Typ::kNormal &&
                  b_acc[I_NEXTVAL(player)][move.r][move.c] == move.value)
      << "MakeMove: wrong value: move: " << move.DebugString()
      << " board: " << DebugString();
  if (move.typ == Move::Typ::kFlag) {
    b_acc[I_FLAG(player)][move.r][move.c] = move.value;
  } else {
    // NORMAL move.
    b_acc[I_VALUE(player)][move.r][move.c] = move.value;
  }
  bool played_flag = move.typ == Move::Typ::kFlag;
  // Occupy cell for both players
  b_acc[I_BLOCKED(0)][move.r][move.c] = 1;
  b_acc[I_BLOCKED(1)][move.r][move.c] = 1;
  // Zero next value.
  b_acc[I_NEXTVAL(0)][move.r][move.c] = 0;
  b_acc[I_NEXTVAL(1)][move.r][move.c] = 0;
  float next_val = 1;
  if (played_flag) {
    ABSL_DCHECK(Flags(player) > 0)
        << "Move " << move.DebugString() << " by " << player
        << " without flags left: " << DebugString();
    // Decrement number of flags.
    b_.index_put_({I_NFLAGS(player)}, static_cast<float>(Flags(player) - 1));
  } else {
    next_val = move.value + 1;
  }
  // Update neighboring cells.
  for (const auto& nb : NeighborsOf(Idx{move.r, move.c})) {
    if (next_val <= 5) {
      if (b_acc[I_BLOCKED(player)][nb.r][nb.c] == 0) {
        // Neighbor cell is not blocked.
        if (b_acc[I_NEXTVAL(player)][nb.r][nb.c] == 0) {
          // Neighbor cell did not have a next value yet.
          b_acc[I_NEXTVAL(player)][nb.r][nb.c] = next_val;
        } else if (b_acc[I_NEXTVAL(player)][nb.r][nb.c] > next_val) {
          // Neighbor cell's value was larger: decrease.
          b_acc[I_NEXTVAL(player)][nb.r][nb.c] = next_val;
        }
      }
    } else {
      // Played a 5: block neighboring cells and clear next value.
      b_acc[I_BLOCKED(player)][nb.r][nb.c] = 1;
      b_acc[I_NEXTVAL(player)][nb.r][nb.c] = 0;
    }
  }
  if (!played_flag) {
    // Occupy neighboring grass fields.
    for (const auto& nb : NeighborsOf(Idx{move.r, move.c})) {
      float grass_val = b_acc[I_GRASS][nb.r][nb.c];
      if (grass_val > 0 &&
          grass_val <= b_acc[I_VALUE(player)][move.r][move.c]) {
        // Occupy grass cell: remove grass value and add it as player's value.
        b_acc[I_GRASS][nb.r][nb.c] = 0;
        b_acc[I_BLOCKED(player)][nb.r][nb.c] = 0;
        b_acc[I_NEXTVAL(player)][nb.r][nb.c] = grass_val;
        MakeMove(player, Move{Move::Typ::kNormal, nb.r, nb.c, grass_val});
      }
    }
  }
}

std::vector<Move> Board::NextMoves(int player) const {
  Perfm::Scope ps(Perfm::NextMoves);
  std::vector<Move> moves;
  auto b_acc = b_.accessor<float, 3>();
  bool flag = Flags(player) > 0;
  // Flag in any unoccupied cell.
  for (int r = 0; r < 11; r++) {
    int cols = 10 - r % 2;
    for (int c = 0; c < cols; c++) {
      if (flag && b_acc[I_BLOCKED(player)][r][c] == 0) {
        moves.push_back(Move::Flag(r, c));
      }
      float next_val = b_acc[I_NEXTVAL(player)][r][c];
      if (next_val > 0) {
        moves.push_back(Move{Move::Typ::kNormal, r, c, next_val});
      }
    }
  }
  return moves;
}

float Board::CellValue(int player, Channel ch, int r, int c) const {
  int ch_idx =
      static_cast<int>(ch == kGrass ? ch : ch + CHANNELS_PER_PLAYER * player);
  return b_.index({ch_idx, r, c}).item<float>();
}

void Board::SetCellValue(int player, Channel ch, int r, int c, float value) {
  int ch_idx =
      static_cast<int>(ch == kGrass ? ch : ch + CHANNELS_PER_PLAYER * player);
  b_.index_put_({ch_idx, r, c}, value);
}

void Board::SetRemainingFlags(int player, int n_flags) {
  b_.index_put_({I_NFLAGS(player)}, static_cast<float>(n_flags));
}

std::string Board::ShortDebugString() const {
  const auto [s0, s1] = Score();
  return absl::StrFormat("Board(flags: %d-%d score: %.0f-%.0f)", Flags(0),
                         Flags(1), s0, s1);
}

std::string Board::DebugString() const {
  std::ostringstream os;
  os << "Board(\n";
  os << "  flags:(" << Flags(0) << ", " << Flags(1) << ")\n";
  os << "  score:" << Score() << "\n";
  os << "  fields: [\n";
  auto b_acc = b_.accessor<float, 3>();
  for (int r = 0; r < 11; r++) {
    int cols = 10 - r % 2;
    for (int c = 0; c < cols; c++) {
      // Output row/col indices and values of all 11 channels
      // Example row: (0, 1): 0 0 1 0 0 1 1 1 0
      os << "    (" << r << ", " << c << "): ";
      for (int i = 0; i < 11; i++) {
        int iv = static_cast<int>(b_acc[i][r][c]);
        if (i > 0) {
          os << "|";
        }
        os << iv;
      }
      if (b_acc[I_BLOCKED(0)][r][c] > 0 || b_acc[I_BLOCKED(1)][r][c] > 0) {
        os << "|b";
      }
      if (b_acc[I_FLAG(0)][r][c] > 0 || b_acc[I_FLAG(1)][r][c] > 0) {
        os << "|f";
      }
      if (b_acc[I_VALUE(0)][r][c] > 0 || b_acc[I_VALUE(1)][r][c] > 0) {
        os << "|v";
      }
      if (b_acc[I_NEXTVAL(0)][r][c] > 0 || b_acc[I_NEXTVAL(1)][r][c] > 0) {
        os << "|n";
      }
      if (b_acc[I_GRASS][r][c] > 0) {
        os << "|g";
      }
      os << "\n";
    }
  }
  os << "  ]\n";
  os << ")";
  return os.str();
}

namespace {

struct RolloutCell {
  int8_t val;
  int8_t next;
  int8_t blocked;
};

class FastGameState {
 public:
  // Initializes this state using the data from board.
  explicit FastGameState(const Board& board) {
    auto t = board.Tensor(0);
    auto acc = t.accessor<float, 3>();
    nflags[0] = acc[I_NFLAGS(0)][0][0];
    nflags[1] = acc[I_NFLAGS(1)][0][0];
    nmoves[0] = 0;
    nmoves[1] = 0;
    free_cells[0] = 0;
    free_cells[1] = 0;
    for (int r = 0; r < 11; r++) {
      for (int c = 0; c < 10 - (r & 1); c++) {
        for (int i = 0; i < 2; i++) {
          b[i][r][c].val = acc[I_VALUE(i)][r][c];
          b[i][r][c].next = acc[I_NEXTVAL(i)][r][c];
          if (b[i][r][c].next > 0) {
            nmoves[i]++;
          }
          b[i][r][c].blocked = acc[I_BLOCKED(i)][r][c];
          if (b[i][r][c].blocked == 0) {
            free_cells[i]++;
          }
        }
        grass[r][c] = acc[I_GRASS][r][c];
      }
    }
  }

  bool FastNextMove(int turn, Move& m) const {
    int moves_cnt = 0;
    float j = 0;
    bool can_place_flag = nflags[turn] > 0 && free_cells[turn] > 0;
    bool pick_flag =
        can_place_flag &&
        (nmoves[turn] == 0 ||
         internal::UnitRandom() <=
             nflags[turn] / static_cast<float>(nmoves[turn] + nflags[turn]));
    if (pick_flag) {
      int n = internal::RandomInt(0, free_cells[turn] - 1);
      int k = 0;
      for (int r = 0; r < 11; r++) {
        for (int c = 0; c < 10 - (r & 1); c++) {
          if (b[turn][r][c].blocked == 0) {
            if (k == n) {
              m = Move{Move::Typ::kFlag, r, c, 1.0};
              return true;
            }
            k++;
          }
        }
      }
    }
    if (nmoves[turn] > 0) {
      // No flag picked. Try a normal move.
      int n = internal::RandomInt(0, nmoves[turn] - 1);
      int k = 0;
      for (int r = 0; r < 11; r++) {
        for (int c = 0; c < 10 - (r & 1); c++) {
          float next_val = b[turn][r][c].next;
          if (next_val > 0) {
            if (k == n) {
              m = Move{Move::Typ::kNormal, r, c, next_val};
              return true;
            }
            k++;
          }
        }
      }
    }
    return false;
  }

  void FastMakeMove(int turn, const Move& m) {
    if (m.typ == Move::Typ::kNormal) {
      b[turn][m.r][m.c].val = m.value;
    }
    // Occupy cell for both players.
    b[0][m.r][m.c].blocked = 1;
    b[1][m.r][m.c].blocked = 1;
    free_cells[0]--;
    free_cells[1]--;
    // Update remaining moves.
    if (b[turn][m.r][m.c].next > 0) {
      nmoves[turn]--;
    }
    if (b[1 - turn][m.r][m.c].next > 0) {
      nmoves[1 - turn]--;
    }
    // Zero next value.
    b[0][m.r][m.c].next = 0;
    b[1][m.r][m.c].next = 0;
    int next_val = 1;
    if (m.typ == Move::Typ::kFlag) {
      nflags[turn]--;
    } else {
      next_val = m.value + 1;
    }
    for (const auto& nb : NeighborsOf(Idx{m.r, m.c})) {
      if (next_val <= 5) {
        if (b[turn][nb.r][nb.c].blocked == 0) {
          // Neighbor cell is not blocked.
          if (b[turn][nb.r][nb.c].next == 0) {
            // Neighbor cell did not have a next value yet.
            nmoves[turn]++;
            b[turn][nb.r][nb.c].next = next_val;
          } else if (b[turn][nb.r][nb.c].next > next_val) {
            // Neighbor cell's value was larger: decrease.
            b[turn][nb.r][nb.c].next = next_val;
          }
        }
      } else {
        // Played a 5: block neighboring cells and clear next value.
        b[turn][nb.r][nb.c].blocked = 1;
        free_cells[turn]--;
        if (b[turn][nb.r][nb.c].next > 0) {
          nmoves[turn]--;
          b[turn][nb.r][nb.c].next = 0;
        }
      }
    }
    if (m.typ == Move::Typ::kNormal) {
      for (const auto& nb : NeighborsOf(Idx{m.r, m.c})) {
        float grass_val = grass[nb.r][nb.c];
        if (grass_val > 0 && grass_val <= b[turn][m.r][m.c].val) {
          // Occupy grass cell: remove grass value and add it as player's value.
          grass[nb.r][nb.c] = 0;
          b[turn][nb.r][nb.c].blocked = 0;
          b[turn][nb.r][nb.c].next = grass_val;
          nmoves[turn]++;  // Will be removed in the recursive call.
          FastMakeMove(turn, Move{Move::Typ::kNormal, nb.r, nb.c, grass_val});
        }
      }
    }
  }

  float Result() const {
    int score[2] = {0, 0};
    for (int r = 0; r < 11; r++) {
      for (int c = 0; c < 10 - (r & 1); c++) {
        score[0] += b[0][r][c].val;
        score[1] += b[1][r][c].val;
      }
    }
    if (score[0] < score[1]) return -1;
    if (score[0] == score[1]) return 0;
    return 1;
  }

 private:
  RolloutCell b[2][11][10];
  int grass[11][10];
  int nflags[2];
  int nmoves[2];
  int free_cells[2];
};

}  // namespace

float FastRandomPlayout(int turn, const Board& board) {
  FastGameState s(board);
  int n_moves = 0;
  while (n_moves < 200) {
    // Find random next move.
    Move m;
    if (!s.FastNextMove(turn, m)) {
      turn = 1 - turn;
      if (!s.FastNextMove(turn, m)) {
        // Game over.
        return s.Result();
      }
    }
    s.FastMakeMove(turn, m);
    turn = 1 - turn;
    n_moves++;
  }
  ABSL_LOG(FATAL) << "FastRandomPlayout error: did not finish the game";
  return 0.0;  // Never reached
}

}  // namespace hexz
