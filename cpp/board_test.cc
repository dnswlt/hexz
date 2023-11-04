#include "board.h"

#include <gtest/gtest.h>

#include <algorithm>

#include "base.h"

namespace hexz {

namespace {

using internal::Idx;
using internal::NeighborsOf;

Board EmptyBoard(int n_flags) {
    Board b;
    b.SetRemainingFlags(0, n_flags);
    b.SetRemainingFlags(1, n_flags);
    return b;
}

TEST(BoardTest, PlayFullGame) {
  Board b = Board::RandomBoard();
  ASSERT_EQ(b.Flags(0), 3);
  ASSERT_EQ(b.Flags(1), 3);

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
    std::shuffle(moves.begin(), moves.end(), internal::rng);
    b.MakeMove(player, moves[0]);
    n_moves++;
  }
  EXPECT_GT(n_moves, 0);
  EXPECT_LT(n_moves, 200);
  // I'd like to expect here that both players played all their flags,
  // but it's actually possible to occupy the whole board without
  // playing all flags. So we just expect that both players played
  // at least one flag.
  EXPECT_LT(b.Flags(0), 3);
  EXPECT_LT(b.Flags(1), 3);
}

TEST(BoardTest, MakeMoveSetsNextValue) {
  int player = 0;
  Board b;
  b.SetRemainingFlags(player, 1);
  b.MakeMove(player, Move{0, 0, 0, 0});
  for (const auto& n : NeighborsOf(Idx{0, 0})) {
    EXPECT_EQ(b.CellValue(player, Board::kNextValue, n.r, n.c), 1);
  }
}

TEST(BoardTest, NoValidNextMoves) {
  // Player zero has one flag, which is surrounded by rocks.
  int player = 0;
  Board b = EmptyBoard(/*n_flags=*/1);
  // Place rocks first:
  for (const auto& n : NeighborsOf(Idx{0, 0})) {
    b.SetCellValue(player, Board::kBlocked, n.r, n.c, 1);
  }
  // Now make the (silly) flag move:
  b.MakeMove(player, Move{0, 0, 0, 0});

  EXPECT_TRUE(b.NextMoves(player).empty());
  EXPECT_FALSE(b.NextMoves(1 - player).empty());
}

TEST(BoardTest, GrassPropagation) {
  // Put grass cells "1" in (3, 3) and "2" in (3, 4).
  // Make a move with value 3 in (3, 5) and expect to occupy both grass cells.
  int player = 0;
  Board b;
  b.SetCellValue(player, Board::kGrass, 3, 3, 1);
  b.SetCellValue(player, Board::kGrass, 3, 4, 2);
  b.MakeMove(player, Move{1, 3, 5, 2});

  EXPECT_EQ(b.CellValue(player, Board::kNextValue, 3, 2), 2);
  EXPECT_EQ(b.CellValue(player, Board::kValue, 3, 3), 1);
  EXPECT_EQ(b.CellValue(player, Board::kGrass, 3, 3), 0);
  EXPECT_EQ(b.CellValue(player, Board::kBlocked, 3, 3), 1);
  EXPECT_EQ(b.CellValue(1 - player, Board::kBlocked, 3, 3), 1);
  EXPECT_EQ(b.CellValue(player, Board::kValue, 3, 4), 2);
  EXPECT_EQ(b.CellValue(player, Board::kGrass, 3, 4), 0);
  EXPECT_EQ(b.CellValue(player, Board::kBlocked, 3, 4), 1);
  EXPECT_EQ(b.CellValue(1 - player, Board::kBlocked, 3, 4), 1);
}

TEST(BoardTest, Score) {
  int p0 = 0, p1 = 1;
  Board b = EmptyBoard(/*n_flags=*/3);
  b.MakeMove(p0, Move{0, 3, 3, 0});
  b.MakeMove(p1, Move{0, 7, 3, 0});
  b.MakeMove(p0, Move{1, 3, 4, 1});
  b.MakeMove(p1, Move{0, 7, 4, 0});
  b.MakeMove(p0, Move{1, 3, 5, 2});
  EXPECT_EQ(b.Score(), std::make_pair(3.0f, 0.0f));
}

TEST(TorchTest, TensorIsRef) {
  // Shows that copying a tensor does not copy the underlying data.
  torch::Tensor t1 = torch::ones({2, 2});
  torch::Tensor t2 = t1;
  t2.index_put_({0, 0}, 2);
  EXPECT_EQ(t1.index({0, 0}).item<float>(), 2);
}

TEST(TorchTest, TensorAccessorWrites) {
  // Shows that writing to a TensorAccessor modifies the original tensor.
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor t = torch::ones({2, 2}, opts);
  auto t_acc = t.accessor<float, 2>();
  t_acc[1][1] = 101;
  EXPECT_EQ(t.index({1, 1}).item<float>(), 101);
}

}  // namespace
}  // namespace hexz
