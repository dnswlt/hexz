#include "batch.h"

#include <absl/log/absl_check.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>

#include "base.h"

namespace hexz {

namespace {

class IntCompute {
 public:
  using input_t = int;
  using result_t = int;

  int AddInput(input_t v) {
    inputs_.push_back(v);
    return inputs_.size() - 1;
  }

  void ComputeAll() {
    n_calls_++;
    for (const auto& inp : inputs_) {
      results_.push_back(ComputeOne(inp));
    }
    // Pretend the computation takes a while
    int64_t t_before = hexz::UnixMicros();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    acc_time_ += hexz::UnixMicros() - t_before;
  }

  result_t GetResult(int idx) {
    ABSL_CHECK(idx >= 0 && idx < results_.size())
        << "idx: " << idx << ", results_.size(): " << results_.size();
    return results_[idx];
  }

  void Reset() {
    inputs_.clear();
    results_.clear();
  }

  int64_t acc_time() { return acc_time_; }
  int n_calls() { return n_calls_; }

 private:
  // The identify function.
  result_t ComputeOne(const input_t& v) { return v; }

  int64_t acc_time_ = 0;
  int n_calls_ = 0;
  std::vector<input_t> inputs_;
  std::vector<result_t> results_;
};

TEST(BatchTest, BatchSizeEqNumThreads) {
  constexpr int kNumThreads = 8;
  constexpr int kMaxBatchSize = kNumThreads;
  constexpr int64_t kTimeoutMicros = 1'000'000;
  constexpr int kNumRounds = 10;
  std::vector<std::thread> ts;
  IntCompute comp;
  Batcher<IntCompute> batcher(comp, kMaxBatchSize, kTimeoutMicros);
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
  EXPECT_EQ(comp.n_calls(), kNumRounds);
}

TEST(BatchTest, BatchSizeOne) {
  // While it is not advisable to use a Batcher if the batch size is just one,
  // it still should work.
  constexpr int kNumThreads = 8;
  constexpr int kMaxBatchSize = 1;
  constexpr int64_t kTimeoutMicros = 1'000'000;
  constexpr int kNumRounds = 10;
  std::vector<std::thread> ts;
  IntCompute comp;
  Batcher<IntCompute> batcher(comp, kMaxBatchSize, kTimeoutMicros);
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
  EXPECT_EQ(comp.n_calls(), kNumRounds * kNumThreads);
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
  IntCompute comp;
  Batcher<IntCompute> batcher(comp, kMaxBatchSize, kTimeoutMicros);
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
  EXPECT_GE(comp.n_calls(), kNumRounds * 2);
  // In the worst case, each of the last kMaxBatchSize-1 threads has to
  // rely on timeouts to get their remaining computations done.
  // This is a very conservative upper bound.
  EXPECT_LE(comp.n_calls(), kNumRounds * 2 + (kMaxBatchSize - 1) * kNumRounds);
}

}  // namespace
}  // namespace hexz
