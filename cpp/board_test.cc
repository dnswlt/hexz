#include "board.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>

#include "base.h"
#include "perfm.h"

namespace hexz {

namespace {

using internal::Idx;
using internal::NeighborsOf;
using testing::UnorderedElementsAre;

TEST(BoardTest, NeighborsOf) {
  EXPECT_THAT(NeighborsOf(Idx{0, 0}),
              UnorderedElementsAre(Idx{0, 1}, Idx{1, 0}));
  EXPECT_THAT(NeighborsOf(Idx{0, 4}),
              UnorderedElementsAre(Idx{0, 3}, Idx{0, 5}, Idx{1, 3}, Idx{1, 4}));
  EXPECT_THAT(NeighborsOf(Idx{0, 9}),
              UnorderedElementsAre(Idx{0, 8}, Idx{1, 8}));
  EXPECT_THAT(NeighborsOf(Idx{4, 0}),
              UnorderedElementsAre(Idx{4, 1}, Idx{3, 0}, Idx{5, 0}));
  EXPECT_THAT(NeighborsOf(Idx{4, 4}),
              UnorderedElementsAre(Idx{3, 3}, Idx{3, 4}, Idx{5, 3}, Idx{5, 4},
                                   Idx{4, 3}, Idx{4, 5}));
  size_t max_size = 0;
  for (int r = 0; r < 11; r++) {
    for (int c = 0; c < 10 - (r & 1); c++) {
      max_size = std::max(max_size, NeighborsOf(Idx{r, c}).size());
    }
  }
  EXPECT_LE(max_size, 6);
}

TEST(BoardTest, EnumValues) {
  // If we change the dimensions of the board, we should adjust the public enum
  // values.
  Board b;
  auto t = b.Tensor(0);
  EXPECT_EQ(t.sizes(), b.Tensor(1).sizes());
  auto s = t.sizes();
  // kGrass is the last enum value (adjust as necessary).
  EXPECT_EQ(s[0] - 1, Board::Channel::kGrass);
  // We expect boards to be 11x10:
  EXPECT_EQ(s[1], 11);
  EXPECT_EQ(s[2], 10);
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
  b.MakeMove(player, Move::Flag(r, c));
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
  b.MakeMove(player, Move::Flag(r, c - 1));
  b.MakeMove(player, Move{Move::Typ::kNormal, r, c, 1});
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
  Board b = Board::EmptyBoard(/*n_flags=*/1);
  // Place rocks first:
  for (const auto& n : NeighborsOf(Idx{0, 0})) {
    b.SetCellValue(player, Board::kBlocked, n.r, n.c, 1);
  }
  // Now make the (silly) flag move:
  b.MakeMove(player, Move::Flag(0, 0));

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
  b.MakeMove(player, Move{Move::Typ::kNormal, 3, 5, 2});

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
  Board b = Board::EmptyBoard(/*n_flags=*/3);
  b.MakeMove(p0, Move::Flag(3, 3));
  b.MakeMove(p1, Move::Flag(7, 3));
  b.MakeMove(p0, Move{Move::Typ::kNormal, 3, 4, 1});
  b.MakeMove(p1, Move::Flag(7, 4));
  b.MakeMove(p0, Move{Move::Typ::kNormal, 3, 5, 2});
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

TEST(RolloutTest, FastRandomPlayout) {
    internal::RNG rng;
  hexz::Perfm::InitScope perfm;
  for (int i = 0; i < 1000; i++) {
    Board b = Board::RandomBoard();
    auto result = FastRandomPlayout(0, b, rng);
    EXPECT_TRUE(result == 0 || result == -1 || result == 1);
  }
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

TEST(TorchTest, Broadcast) {
  // We use this broadcasting for the kRemainingFlags channels,
  // make sure it works as we expect.
  auto t = torch::ones({11, 11, 10});
  t.index_put_({Board::Channel::kRemainingFlags}, 0);
  // Look at an arbitrary index in the middle:
  EXPECT_EQ(t.index({Board::Channel::kValue, 4, 7}).item<float>(), 1);
  EXPECT_EQ(t.index({Board::Channel::kRemainingFlags, 4, 7}).item<float>(), 0);
  // Check that all are set:
  EXPECT_TRUE(
      torch::all(t.index({Board::Channel::kRemainingFlags}) == 0).item<bool>());
}

TEST(TorchTest, StackTest) {
  // torch::stack is the function to use to create batches of tensors to pass to
  // the model.
  auto t = torch::ones({3, 4, 5});
  auto t2 = torch::stack({t, t});
  EXPECT_THAT(t2.sizes(), testing::ElementsAre(2, 3, 4, 5));
}

}  // namespace
}  // namespace hexz
