#pragma once

#include <absl/base/no_destructor.h>

#include <cassert>
#include <chrono>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace hexz {

class Perfm {
 public:
  enum Label {
    Predict = 0,
    FindLeaf = 1,
    MakeMove = 2,
    PlayGame = 3,
    MaxPuctChild = 4,
    Puct = 5,
    NextMoves = 6,
    NeuralMCTS_Run = 7,
    RandomPlayout = 8,
    PredictBatch = 9,
    // Add new labels here. But don't remove StatsSize and increment its value:
    // It indicates the number of labels we have and defines array sizes below.
    //
    // Also add the new enum to the LabelName method below.
    StatsSize = 10,
  };

  // Helper struct to use RAII for initialization and printing final results.
  struct InitScope {
    InitScope() { Perfm::Init(); }
    ~InitScope() {
      Perfm::AccumulateThreadLocalStats();
      Perfm::PrintStats();
    }
  };
  struct ThreadScope {
    ThreadScope() = default;
    ~ThreadScope() { Perfm::AccumulateThreadLocalStats(); }
  };

  struct Scope {
    Perfm::Label label;
    std::chrono::high_resolution_clock::time_point started;
    Scope(Perfm::Label label)
        : label{label}, started{std::chrono::high_resolution_clock::now()} {}
    ~Scope() {
      Perfm::stats_[label].count++;
      Perfm::stats_[label].elapsed_nanos +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now() - started)
              .count();
    }
  };

  struct CumulativeStats {
    Perfm::Label label;
    int64_t count = 0;
    int64_t elapsed_nanos = 0;
  };

  static const std::string& LabelName(Perfm::Label label) {
    static std::string names[StatsSize] = {
        "Predict",       "FindLeaf",     "MakeMove",  "PlayGame",
        "MaxPuctChild",  "Puct",         "NextMoves", "NeuralMCTS::Run",
        "RandomPlayout", "PredictBatch",
    };
    assert(label < StatsSize);
    return names[label];
  }

  static void Init();

  static void PrintStats();

  // Adds the thread-local stats into the static accumulated stats.
  static void AccumulateThreadLocalStats() {
    std::unique_lock<std::mutex> lk(Perfm::mut_);
    for (int i = 0; i < StatsSize; i++) {
      cum_stats_[i].count += stats_[i].count;
      cum_stats_[i].elapsed_nanos += stats_[i].elapsed_nanos;
      stats_[i] = CumulativeStats{};  // Reset thread local values to 0;
    }
  }

 private:
  static CumulativeStats cum_stats_[StatsSize];
  static thread_local CumulativeStats stats_[StatsSize];
  static std::mutex mut_;
};

// APM is used to measure application performance per wall-time. It can be used
// to measure work item throughput per wall-time.
// This is in contrast to Perfm, which measures individual execution times of
// code blocks and provides averages for these. In other words, APM measures
// "how much work gets done over time", whereas Perfm measures "how long does a
// specific code block take to run".
template <typename Clock = std::chrono::high_resolution_clock>
class APM {
 public:
  // Reserve twice the capacity needed to retain max_window_seconds.
  // We maintain a sliding window that moves in max_window_seconds steps.
  explicit APM(std::string name, int64_t max_window_seconds)
      : name_(std::move(name)),
        max_window_seconds_(max_window_seconds),
        counts_(max_window_seconds * 2, 0),
        t_start_(Clock::now()) {}

  // Increment the count of processed work items for the current second by n.
  void Increment(int n) {
    int64_t d = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() -
                                                                 t_start_)
                    .count();
    std::scoped_lock<std::mutex> lk(mut_);
    d = AlignCounts(d);
    counts_[d] += n;
  }

  // Rate returns the rate of items processed per second, averaged over the
  // given window (in seconds). The "current" second (the one for which stats
  // are still being collected), will be added to the window, so that this
  // method returns the rate over a window_seconds +
  // time-passed-in-current-second window.
  double Rate(int window_seconds) {
    int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
                     Clock::now() - t_start_)
                     .count();
    int64_t d = us / 1'000'000;
    int64_t rem = us % 1'000'000;
    if (d == 0 && rem == 0) {
      // Rate was called right after this instance was created.
      return 0;
    }

    std::scoped_lock<std::mutex> lk(mut_);
    d = AlignCounts(d);
    int64_t w = std::min(static_cast<int64_t>(window_seconds), d);
    // Accumulate counts_[d - w .. d+1]
    int64_t sum = std::accumulate(counts_.begin() + (d - w),
                                  counts_.begin() + (d + 1), 0);
    double t = static_cast<double>(w) + static_cast<double>(rem) / 1e6;
    return static_cast<double>(sum) / t;
  }

  std::vector<int64_t> CountsForTesting() {
    std::scoped_lock<std::mutex> lk(mut_);
    return counts_;
  }

  Clock::time_point t_start() const { 
    std::scoped_lock<std::mutex> lk(mut_);
    return t_start_;
  }

 private:
  // Realign counts_ such that d (an offset in seconds from the t_start_)
  // is a valid index. Callers MUST use the returned value as the new d,
  // since this method shifts data in counts_ and adjusts t_start_ accordingly.
  //
  // Must only be called while holding a lock on mut_.
  int64_t AlignCounts(int64_t d) {
    if (d >= counts_.size()) {
      // Realign s.t. the latest points [d-max_window_seconds..d] are at the
      // front.
      size_t offset = d - max_window_seconds_;
      int i = 0;
      for (int j = offset; j < counts_.size(); j++) {
        counts_[i++] = counts_[j];
      }
      for (; i < counts_.size(); i++) {
        counts_[i] = 0;
      }
      t_start_ = t_start_ + std::chrono::seconds(offset);
      return d - offset;
    }
    return d;
  }

  const std::string name_;
  const int64_t max_window_seconds_;
  mutable std::mutex mut_;
  std::vector<int64_t> counts_;
  Clock::time_point t_start_;
};

// Used by workers and MCTS code to count the throughput w.r.t. predictions
// generated.
inline APM<>& APMPredictions() noexcept {
  static absl::NoDestructor<APM<>> apm("/predictions", 3600);
  return *apm;
}
// Used by workers and MCTS code to count the throughput w.r.t. examples
// generated.
inline APM<>& APMExamples() noexcept {
  static absl::NoDestructor<APM<>> apm("/examples", 3600);
  return *apm;
}
// Used by workers and MCTS code to count the throughput w.r.t. games generated.
inline APM<>& APMGames() noexcept {
  static absl::NoDestructor<APM<>> apm("/games", 3600);
  return *apm;
}

}  // namespace hexz
