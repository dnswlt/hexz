#include "batch.h"

#include <absl/log/absl_check.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <latch>
#include <thread>

#include "base.h"

namespace hexz {

namespace {

using testing::ElementsAre;

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

  explicit IntCompute(int& n_calls, std::vector<int>* input_sizes = nullptr)
      : n_calls_{n_calls}, input_sizes_{input_sizes} {}

  std::vector<int> ComputeAll(std::vector<int>&& inputs) {
    n_calls_++;
    if (input_sizes_ != nullptr) {
      input_sizes_->push_back(inputs.size());
    }
    // Identity function.
    std::vector<int> results = std::move(inputs);

    // Pretend the computation takes a while
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return results;
  }

 private:
  // Using a reference here so we can view the updates from the "outside".
  int& n_calls_;
  std::vector<int>* input_sizes_;
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
  // Synchronize threads: ensure they first all register.
  std::latch reg_sync{kNumThreads};
  for (int i = 0; i < kNumThreads; i++) {
    ts.emplace_back([&] {
      auto token = batcher.RegisterThread();
      reg_sync.count_down();
      reg_sync.wait();
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
  // Synchronize threads: ensure they first all register.
  std::latch reg_sync{kNumThreads};
  for (int i = 0; i < kNumThreads; i++) {
    ts.emplace_back([&] {
      auto token = batcher.RegisterThread();
      reg_sync.count_down();
      reg_sync.wait();
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
  // Synchronize threads: ensure they first all register.
  std::latch reg_sync{kNumThreads};
  for (int i = 0; i < kNumThreads; i++) {
    ts.emplace_back([&] {
      auto token = batcher.RegisterThread();
      reg_sync.count_down();
      reg_sync.wait();
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

TEST(BatchTest, VaryingCallCounts) {
  // N threads use the same batcher, but need different numbers of computations.
  // Batcher's thread registration count should handle this.
  constexpr int kNumThreads = 8;
  constexpr int64_t kTimeoutMicros = 3'600'000'000;  // 1h
  std::vector<std::thread> ts;
  int n_calls = 0;
  std::vector<int> input_sizes;
  Batcher<IntCompute> batcher(
      std::make_unique<IntCompute>(n_calls, &input_sizes), kNumThreads,
      kTimeoutMicros);
  std::latch reg_sync{kNumThreads};
  for (int i = 0; i < kNumThreads; i++) {
    ts.emplace_back([&, call_count = i + 1] {
      auto token = batcher.RegisterThread();
      reg_sync.count_down();
      reg_sync.wait();
      int sum = 0;
      for (int j = 0; j < call_count; j++) {
        sum += batcher.ComputeValue(j);
      }
      ASSERT_EQ(sum, (call_count * (call_count - 1)) / 2);
    });
  }
  for (auto& t : ts) {
    if (t.joinable()) {
      t.join();
    }
  }
  EXPECT_EQ(n_calls, kNumThreads);
  EXPECT_THAT(input_sizes, ElementsAre(8, 7, 6, 5, 4, 3, 2, 1));
}

}  // namespace
}  // namespace hexz
