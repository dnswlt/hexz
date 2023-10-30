#include "board.h"

#include <algorithm>
#include <gtest/gtest.h>

namespace hexz {

std::mt19937 rng{std::random_device{}()};

TEST(BoardTest, PlayFullGame) {
  Board b = Board::RandomBoard();
  int player = 0;
  int n_moves = 0;
  while (n_moves < 200) {
    auto moves = b.NextMoves(player);
    if (moves.empty()) {
      player = 1 - player;
      moves = b.NextMoves(player);
      if (moves.empty()) {
        break;
      }
    }
    std::shuffle(moves.begin(), moves.end(), rng);
    b.MakeMove(player, moves[0]);
    n_moves++;
  }
  EXPECT_GT(n_moves, 0);
  EXPECT_LT(n_moves, 200);
  EXPECT_EQ(b.Flags(0), 0);
  EXPECT_EQ(b.Flags(1), 0);
}

TEST(TorchTest, TensorIsRef) {
  // Shows that copying a tensor does not copy the underlying data.
  torch::Tensor t1 = torch::ones({2, 2});
  torch::Tensor t2 = t1;
  t2.index_put_({0, 0}, 2);
  EXPECT_EQ(t1.index({0, 0}).item<float>(), 2);
}

}  // namespace hexz
