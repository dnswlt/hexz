#include "batch.h"

#include <absl/log/absl_check.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>

#include "base.h"

namespace hexz {

namespace {

class Moveable {
 public:
  explicit Moveable(int value) : value_{value} {}
  Moveable(const Moveable& other) : value_{other.value_} { copies_++; }
  Moveable(Moveable&& other) : value_{other.value_} {
    other.value_ = 0;
    moves_++;
  }
  int value() { return value_; }

  static void PassByRValueRef(Moveable&& m) { m.value_++; }
  static void PassByRValueRefAndMove(Moveable&& m) {
    Moveable unused(std::move(m));
  }
  static int moves_;
  static int copies_;

 private:
  int value_ = 0;
};

int Moveable::moves_ = 0;
int Moveable::copies_ = 0;

TEST(MoveTest, MoveMadness) {
  // Just to confirm my intuition about C++ rvalues and moves.
  Moveable s(1);
  Moveable& t = s;
  Moveable u = std::move(t);
  EXPECT_EQ(s.value(), 0);
  EXPECT_EQ(u.value(), 1);
  EXPECT_EQ(Moveable::moves_, 1);
  EXPECT_EQ(Moveable::copies_, 0);
  Moveable::PassByRValueRef(std::move(s));
  EXPECT_EQ(s.value(), 1);
  EXPECT_EQ(Moveable::moves_, 1);
  EXPECT_EQ(Moveable::copies_, 0);
  Moveable::PassByRValueRefAndMove(std::move(s));
  EXPECT_EQ(s.value(), 0);
  EXPECT_EQ(Moveable::moves_, 2);
}

class IntCompute {
 public:
  using input_t = int;
  using result_t = int;

  explicit IntCompute(int& n_calls) : n_calls_{n_calls} {}

  std::vector<int> ComputeAll(std::vector<int>&& inputs) {
    n_calls_++;
    // Identity function.
    std::vector<int> results = std::move(inputs);

    // Pretend the computation takes a while
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return results;
  }

  int n_calls() { return n_calls_; }

 private:
  // Using a reference here so we can view the updates from the "outside".
  int& n_calls_;
};

TEST(BatchTest, BatchSizeEqNumThreads) {
  constexpr int kNumThreads = 8;
  constexpr int kMaxBatchSize = kNumThreads;
  constexpr int64_t kTimeoutMicros = 1'000'000;
  constexpr int kNumRounds = 10;
  std::vector<std::thread> ts;
  int n_calls = 0;
  Batcher<IntCompute> batcher(std::make_unique<IntCompute>(n_calls),
                              kMaxBatchSize, kTimeoutMicros);
  for (int i = 0; i < kNumThreads; i++) {
    ts.emplace_back([&batcher, i] {
      int sum = 0;
      for (int j = 0; j < kNumRounds; j++) {
        sum += batcher.ComputeValue(j);
      }
      ASSERT_EQ(sum, (kNumRounds * (kNumRounds - 1)) / 2);
    });
  }
  for (auto& t : ts) {
    if (t.joinable()) {
      t.join();
    }
  }
  // If the batch size is equal to the number of threads,
  // ComputeAll should have been called exactly once each round.
  EXPECT_EQ(n_calls, kNumRounds);
}

TEST(BatchTest, BatchSizeOne) {
  // While it is not advisable to use a Batcher if the batch size is just one,
  // it still should work.
  constexpr int kNumThreads = 8;
  constexpr int kMaxBatchSize = 1;
  constexpr int64_t kTimeoutMicros = 1'000'000;
  constexpr int kNumRounds = 10;
  std::vector<std::thread> ts;
  int n_calls = 0;
  Batcher<IntCompute> batcher(std::make_unique<IntCompute>(n_calls),
                              kMaxBatchSize, kTimeoutMicros);
  for (int i = 0; i < kNumThreads; i++) {
    ts.emplace_back([&batcher, i] {
      int sum = 0;
      for (int j = 0; j < kNumRounds; j++) {
        sum += batcher.ComputeValue(j);
      }
      ASSERT_EQ(sum, (kNumRounds * (kNumRounds - 1)) / 2);
    });
  }
  for (auto& t : ts) {
    if (t.joinable()) {
      t.join();
    }
  }
  EXPECT_EQ(n_calls, kNumRounds * kNumThreads);
}

TEST(BatchTest, BatchSizeHalf) {
  // If the batch size is half the number of threads, there will be threads
  // that wait to enter the batch, but are blocked b/c the batch is full.
  // Moreover, towards the end of the run, there will likely be timeouts,
  // because once fewer threads than the batch size are alive, they cannot fill
  // the batch entirely anymore.
  constexpr int kNumThreads = 8;
  constexpr int kMaxBatchSize = kNumThreads / 2;
  constexpr int64_t kTimeoutMicros = 100'000;
  constexpr int kNumRounds = 10;
  std::vector<std::thread> ts;
  int n_calls = 0;
  Batcher<IntCompute> batcher(std::make_unique<IntCompute>(n_calls),
                              kMaxBatchSize, kTimeoutMicros);
  for (int i = 0; i < kNumThreads; i++) {
    ts.emplace_back([&batcher, i] {
      int sum = 0;
      for (int j = 0; j < kNumRounds; j++) {
        sum += batcher.ComputeValue(j);
      }
      ASSERT_EQ(sum, (kNumRounds * (kNumRounds - 1)) / 2);
    });
  }
  for (auto& t : ts) {
    if (t.joinable()) {
      t.join();
    }
  }
  EXPECT_GE(n_calls, kNumRounds * 2);
  // In the worst case, each of the last kMaxBatchSize-1 threads has to
  // rely on timeouts to get their remaining computations done.
  // This is a very conservative upper bound.
  EXPECT_LE(n_calls, kNumRounds * 2 + (kMaxBatchSize - 1) * kNumRounds);
}

}  // namespace
}  // namespace hexz
