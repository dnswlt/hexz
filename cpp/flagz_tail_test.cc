#include "flagz_tail.h"

#include <gtest/gtest.h>

#include <initializer_list>
#include <utility>

#include "board.h"

namespace hexz {
namespace {

using Cell = std::pair<int, int>;

void BlockAllCells(Board& board) {
  for (int r = 0; r < 11; ++r) {
    for (int c = 0; c < 10 - r % 2; ++c) {
      board.SetCellValue(0, Board::kBlocked, r, c, 1);
      board.SetCellValue(1, Board::kBlocked, r, c, 1);
    }
  }
}

void OpenCells(Board& board, int player,
               std::initializer_list<Cell> cells) {
  for (const auto& [r, c] : cells) {
    board.SetCellValue(player, Board::kBlocked, r, c, 0);
  }
}

TEST(FlagzTailTest, FlagsRemaining) {
  Board board = Board::EmptyBoard(/*flags=*/1);

  const TailResolution result =
      ResolveSeparatedTail(board, /*max_states=*/1000,
                           /*max_micros=*/1'000'000);

  EXPECT_EQ(result.status, TailResolveStatus::kFlagsRemaining);
}

TEST(FlagzTailTest, ForcedChain) {
  Board board = Board::EmptyBoard(/*flags=*/0);
  BlockAllCells(board);
  OpenCells(board, 0, {{0, 0}, {0, 1}, {0, 2}});
  board.SetCellValue(0, Board::kNextValue, 0, 0, 1);

  const TailResolution result =
      ResolveSeparatedTail(board, /*max_states=*/1000,
                           /*max_micros=*/1'000'000);

  ASSERT_EQ(result.status, TailResolveStatus::kSolved);
  EXPECT_EQ(result.optimal_score[0], 6);
  EXPECT_EQ(result.optimal_score[1], 0);
  EXPECT_EQ(result.Result(), 1);
}

TEST(FlagzTailTest, GrassResetsPropagationAfterFive) {
  Board board = Board::EmptyBoard(/*flags=*/0);
  BlockAllCells(board);
  OpenCells(board, 0, {{0, 0}, {0, 2}, {0, 3}});
  board.SetCellValue(0, Board::kNextValue, 0, 0, 5);
  board.SetCellValue(0, Board::kGrass, 0, 1, 1);

  const TailResolution result =
      ResolveSeparatedTail(board, /*max_states=*/1000,
                           /*max_micros=*/1'000'000);

  ASSERT_EQ(result.status, TailResolveStatus::kSolved);
  EXPECT_EQ(result.optimal_score[0], 11);
}

TEST(FlagzTailTest, DetectsOverlappingReachability) {
  Board board = Board::EmptyBoard(/*flags=*/0);
  BlockAllCells(board);
  OpenCells(board, 0, {{0, 0}, {0, 1}});
  OpenCells(board, 1, {{0, 1}, {0, 2}});
  board.SetCellValue(0, Board::kNextValue, 0, 0, 1);
  board.SetCellValue(1, Board::kNextValue, 0, 2, 1);

  const TailResolution result =
      ResolveSeparatedTail(board, /*max_states=*/1000,
                           /*max_micros=*/1'000'000);

  EXPECT_EQ(result.status, TailResolveStatus::kTerritoriesOverlap);
}

TEST(FlagzTailTest, AddsIndependentComponents) {
  Board board = Board::EmptyBoard(/*flags=*/0);
  BlockAllCells(board);
  OpenCells(board, 0, {{0, 0}, {0, 1}, {10, 0}, {10, 1}});
  board.SetCellValue(0, Board::kNextValue, 0, 0, 1);
  board.SetCellValue(0, Board::kNextValue, 10, 0, 3);

  const TailResolution result =
      ResolveSeparatedTail(board, /*max_states=*/1000,
                           /*max_micros=*/1'000'000);

  ASSERT_EQ(result.status, TailResolveStatus::kSolved);
  EXPECT_EQ(result.optimal_score[0], 10);
}

TEST(FlagzTailTest, FallsBackAtStateLimit) {
  Board board = Board::EmptyBoard(/*flags=*/0);
  BlockAllCells(board);
  OpenCells(board, 0,
            {{4, 4}, {4, 3}, {4, 5}, {3, 3}, {3, 4}, {5, 3}, {5, 4}});
  board.SetCellValue(0, Board::kNextValue, 4, 4, 1);

  const TailResolution result =
      ResolveSeparatedTail(board, /*max_states=*/1,
                           /*max_micros=*/1'000'000);

  EXPECT_EQ(result.status, TailResolveStatus::kSolveLimit);
}

}  // namespace
}  // namespace hexz
