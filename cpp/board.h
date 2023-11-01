#ifndef __HEXZ_BOARD_H__
#define __HEXZ_BOARD_H__
#include <torch/torch.h>

#include <random>
#include <utility>
#include <vector>

namespace hexz {

struct Move {
  int typ;
  int r;
  int c;
  float value;
};

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
class Board {
 public:
  static Board RandomBoard();

  // Copy c'tor.
  Board(const Board& other);

  std::pair<float, float> Score() const;
  float Result() const;

  torch::Tensor Tensor(int player) const;

  int Flags(int player) const;

  void MakeMove(int player, const Move& move);
  std::vector<Move> NextMoves(int player) const;

 private:
  Board();

  static std::mt19937 rng_;
  torch::Tensor b_;
  int nflags_[2];
};

}  // namespace hexz
#endif  // __HEXZ_BOARD_H__
