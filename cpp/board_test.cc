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
    player = 1 - player;
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

TEST(BoardTest, MakeMoveFlag) {
  // Making a move should update all channels appropriately.
  int player = 0;
  Board b;
  b.SetRemainingFlags(player, 1);
  int r = 4;
  int c = 4;
  b.MakeMove(player, Move{0, r, c, 1});
  // Flag should be set.
  EXPECT_EQ(b.CellValue(player, Board::kFlag, r, c), 1);
  EXPECT_EQ(b.CellValue(1 - player, Board::kFlag, r, c), 0);
  // Cell should not have a value, since it's a flag.
  EXPECT_EQ(b.CellValue(player, Board::kValue, r, c), 0);
  EXPECT_EQ(b.CellValue(1 - player, Board::kValue, r, c), 0);
  // Cell should be blocked for both players.
  EXPECT_EQ(b.CellValue(player, Board::kBlocked, r, c), 1);
  EXPECT_EQ(b.CellValue(1 - player, Board::kBlocked, r, c), 1);
  // All neighbors should have next value.
  for (const auto& n : NeighborsOf(Idx{r, c})) {
    EXPECT_EQ(b.CellValue(player, Board::kNextValue, n.r, n.c), 1);
    EXPECT_EQ(b.CellValue(1 - player, Board::kNextValue, n.r, n.c), 0);
  }
}

TEST(BoardTest, MakeMoveNormal) {
  // Making a move should update all channels appropriately.
  int player = 1;
  Board b;
  b.SetRemainingFlags(player, 1);
  int r = 4;
  int c = 4;
  b.MakeMove(player, Move{0, r, c - 1, 1});
  b.MakeMove(player, Move{1, r, c, 1});
  // Cell should not have a value, since it's a flag.
  EXPECT_EQ(b.CellValue(player, Board::kValue, r, c), 1);
  EXPECT_EQ(b.CellValue(1 - player, Board::kValue, r, c), 0);
  // Cell should be blocked for both players.
  EXPECT_EQ(b.CellValue(player, Board::kBlocked, r, c), 1);
  EXPECT_EQ(b.CellValue(1 - player, Board::kBlocked, r, c), 1);
  // Flag
  ASSERT_EQ(b.CellValue(player, Board::kFlag, r, c - 1), 1);
  EXPECT_EQ(b.CellValue(player, Board::kNextValue, r, c - 1), 0);
  EXPECT_EQ(b.CellValue(1 - player, Board::kNextValue, r, c - 1), 0);
  // Neighbors that don't also touch the flag.
  EXPECT_EQ(b.CellValue(player, Board::kNextValue, r, c + 1), 2);
  EXPECT_EQ(b.CellValue(1 - player, Board::kNextValue, r, c + 1), 0);
  // Neighbors touching the flag.
  EXPECT_EQ(b.CellValue(player, Board::kNextValue, r - 1, c - 1), 1);
  EXPECT_EQ(b.CellValue(1 - player, Board::kNextValue, r - 1, c - 1), 0);
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

TEST(BoardTest, MakeMove) {
  int player = 0;
  Board b;
  b.SetCellValue(player, Board::kGrass, 3, 3, 1);
}

TEST(BoardTest, GrassPropagation) {
  // Put grass cells "1" in (3, 3) and "2" in (3, 4).
  // Make a move with value 3 in (3, 5) and expect to occupy both grass cells.
  int player = 0;
  Board b;
  b.SetCellValue(player, Board::kGrass, 3, 3, 1);
  b.SetCellValue(player, Board::kGrass, 3, 4, 2);
  // Ensure it's OK to make a move on (3, 5) by setting next value:
  b.SetCellValue(player, Board::kNextValue, 3, 5, 2);
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

hexzpb::Board EmptyProtoBoard(int n_flags) {
  hexzpb::Board pb;
  // Add resources
  const int iFlag = static_cast<int>(hexzpb::Field::FLAG);
  auto* r1 = pb.add_resources();
  auto* r2 = pb.add_resources();
  for (int i = 0; i <= iFlag; i++) {
    r1->add_num_pieces(i == iFlag ? n_flags : 0);
    r2->add_num_pieces(i == iFlag ? n_flags : 0);
  }
  // Add empty fields
  for (int i = 0; i < 105; i++) {
    auto& f = *pb.add_flat_fields();
    f.set_type(hexzpb::Field::NORMAL);
  }
  return pb;
}

TEST(BoardTest, FromProtoValidEmpty) {
  auto pb = EmptyProtoBoard(/*n_flags=*/0);
  const auto b = Board::FromProto(pb);
  EXPECT_TRUE(b.ok()) << b.status();
}

TEST(BoardTest, FromProtoInvalid) {
  {
    // Missing resources.
    hexzpb::Board pb = EmptyProtoBoard(0);
    pb.clear_resources();
    const auto b = Board::FromProto(pb);
    ASSERT_FALSE(b.ok());
  }
  {
    // Wrong number of fields.
    hexzpb::Board pb = EmptyProtoBoard(0);
    pb.add_flat_fields();
    const auto b = Board::FromProto(pb);
    ASSERT_FALSE(b.ok());
  }
  {
    // Invalid owner.
    hexzpb::Board pb = EmptyProtoBoard(0);
    pb.mutable_flat_fields(0)->set_owner(3);
    const auto b = Board::FromProto(pb);
    ASSERT_FALSE(b.ok());
  }
}

TEST(BoardTest, FromProtoValid) {
  auto pb = EmptyProtoBoard(/*n_flags=*/1);
  auto& f0 = *pb.mutable_flat_fields(0);
  f0.set_owner(1);
  f0.set_value(3);
  auto& f1 = *pb.mutable_flat_fields(1);
  f1.set_owner(2);
  f1.set_value(1);
  auto& f2 = *pb.mutable_flat_fields(2);
  f2.add_next_val(2);
  f2.add_next_val(3);
  auto& f3 = *pb.mutable_flat_fields(3);
  f3.set_type(hexzpb::Field::ROCK);
  auto& f4 = *pb.mutable_flat_fields(4);
  f4.set_type(hexzpb::Field::FLAG);
  f4.set_owner(2);
  auto& f5 = *pb.mutable_flat_fields(5);
  f5.set_type(hexzpb::Field::GRASS);
  f5.set_value(5);
  const auto b = Board::FromProto(pb);
  ASSERT_TRUE(b.ok());
  // f0
  EXPECT_EQ(b->CellValue(0, Board::kValue, 0, 0), 3.0);
  EXPECT_EQ(b->CellValue(1, Board::kValue, 0, 0), 0.0);
  EXPECT_EQ(b->CellValue(0, Board::kBlocked, 0, 0), 1.0);
  EXPECT_EQ(b->CellValue(1, Board::kBlocked, 0, 0), 1.0);
  // f1
  EXPECT_EQ(b->CellValue(1, Board::kValue, 0, 1), 1.0);
  // f2
  EXPECT_EQ(b->CellValue(0, Board::kNextValue, 0, 2), 2.0);
  EXPECT_EQ(b->CellValue(1, Board::kNextValue, 0, 2), 3.0);
  // f3
  EXPECT_EQ(b->CellValue(0, Board::kBlocked, 0, 3), 1.0);
  EXPECT_EQ(b->CellValue(1, Board::kBlocked, 0, 3), 1.0);
  // f4
  EXPECT_EQ(b->CellValue(0, Board::kFlag, 0, 4), 0.0);
  EXPECT_EQ(b->CellValue(1, Board::kFlag, 0, 4), 1.0);
  // f5
  EXPECT_EQ(b->CellValue(1, Board::kGrass, 0, 5), 5.0);
  EXPECT_EQ(b->CellValue(0, Board::kBlocked, 0, 5), 1.0);
  EXPECT_EQ(b->CellValue(1, Board::kBlocked, 0, 5), 1.0);
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
